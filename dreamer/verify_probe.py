"""Component 3a, rigorous gate — is the player/enemy info IN the latent?

A blurry reconstruction can mean either (a) the latent is blind to sprites, or
(b) the decoder is just lazy (no MSE incentive to render an 8px soldier sharply)
while the encoder embedding still encodes its position. For Dreamer only (a)
would be fatal — the RSSM/actor read the latent, not the pixels.

This settles it with a LINEAR probe: train the autoencoder, freeze the encoder,
then least-squares-fit embedding → {player_x, player_y, n_enemies} from RAM
ground truth (contra.game_state). If a *linear* map recovers player position to
within a few pixels, the latent is not blind and we may proceed to the RSSM.
We also probe a randomly-initialized encoder as a control (random conv features
can leak some position), so the number that matters is trained − random.

    python -m dreamer.verify_probe --frames 5000 --steps 6000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer.models import ConvDecoder, ConvEncoder


def _collect(n: int, size: int, seed: int):
    """Return frames (N,size,size,3) uint8 and targets (N,3): px, py, n_enemies."""
    from dreamer.envs import make_contra_env, ACTION_NAMES
    from contra.game_state import state_from_ram

    rng = np.random.default_rng(seed)
    RF = ACTION_NAMES.index("RF")
    env = make_contra_env(level=1, size=size)
    frames = np.zeros((n, size, size, 3), dtype=np.uint8)
    targets = np.zeros((n, 3), dtype=np.float32)
    try:
        obs, _ = env.reset(seed=seed)
        for i in range(n):
            ram = env.unwrapped.get_ram()
            s = state_from_ram(ram)
            px, py = s[3], s[4]
            enemies = s[26:90].reshape(16, 4)
            n_en = float((enemies[:, 0] != 0).sum())
            frames[i] = obs
            targets[i] = (px, py, n_en)
            a = RF if rng.random() < 0.9 else int(rng.integers(len(ACTION_NAMES)))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset()
    finally:
        env.close()
    return frames, targets


def _embed(enc, frames_u8, device, bs=256):
    enc.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(frames_u8), bs):
            x = torch.as_tensor(frames_u8[i:i + bs], dtype=torch.float32, device=device) / 255.0
            outs.append(enc(x.permute(0, 3, 1, 2)).cpu())
    return torch.cat(outs)  # (N, embed_dim)


def _ridge_probe(emb_tr, y_tr, emb_ev, y_ev, lam=10.0):
    """Closed-form ridge regression; return per-target RMSE on eval."""
    X = torch.cat([emb_tr, torch.ones(len(emb_tr), 1)], 1)   # bias column
    Xe = torch.cat([emb_ev, torch.ones(len(emb_ev), 1)], 1)
    A = X.T @ X + lam * torch.eye(X.shape[1])
    W = torch.linalg.solve(A, X.T @ y_tr)                    # (D+1, K)
    pred = Xe @ W
    rmse = torch.sqrt(((pred - y_ev) ** 2).mean(0))
    return rmse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=4096)
    p.add_argument("--depth", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"[probe] device={device} collecting {args.frames} frames + RAM state…")
    frames, targets = _collect(args.frames, args.size, args.seed)

    n_eval = max(64, args.frames // 10)
    fr_tr, fr_ev = frames[:-n_eval], frames[-n_eval:]
    y_tr = torch.as_tensor(targets[:-n_eval])
    y_ev = torch.as_tensor(targets[-n_eval:])
    names = ["player_x(px)", "player_y(px)", "n_enemies"]
    base = torch.sqrt(((y_ev - y_tr.mean(0)) ** 2).mean(0))   # predict-mean baseline

    enc = ConvEncoder(args.size, depth=args.depth, embed_dim=args.embed_dim).to(device)

    # control: probe the RANDOM (untrained) encoder first
    rnd = _ridge_probe(_embed(enc, fr_tr, device), y_tr, _embed(enc, fr_ev, device), y_ev)

    # train the autoencoder (reconstruction only — the 3a objective)
    dec = ConvDecoder(args.size, depth=args.depth, feat_dim=args.embed_dim).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=args.lr)
    enc.train()
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(fr_tr), args.batch)
        x = torch.as_tensor(fr_tr[idx], dtype=torch.float32, device=device) / 255.0
        x = x.permute(0, 3, 1, 2)
        loss = F.mse_loss(dec(enc(x)), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  ae step {step:5d}  mse {loss.item():.5f}")

    trn = _ridge_probe(_embed(enc, fr_tr, device), y_tr, _embed(enc, fr_ev, device), y_ev)

    print("\n  linear-probe RMSE on held-out frames (lower = info is in the latent):")
    print(f"    {'target':14s} {'baseline':>10s} {'random-enc':>11s} {'trained-enc':>12s}")
    for k, name in enumerate(names):
        print(f"    {name:14s} {base[k].item():10.2f} {rnd[k].item():11.2f} {trn[k].item():12.2f}")
    print("\n  player_x/y range is 0..255 px; a trained-enc RMSE of a few px means")
    print("  the latent localizes the player. If it's near the baseline, it's blind.")


if __name__ == "__main__":
    main()
