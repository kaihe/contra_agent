"""Component 3a verification gate — does the conv encoder/decoder represent
Contra frames through a compact latent?

We collect real frames, train ConvEncoder→ConvDecoder as a plain autoencoder on
a train split, then report PSNR on a held-out split and dump a side-by-side
input/reconstruction grid. The gate: held-out reconstructions match the inputs by
eye (terrain, player, enemies recognizable) and PSNR climbs well above the
blur-everything baseline.

    python -m dreamer.verify_ae --frames 3000 --steps 4000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer.buffer import ReplayBuffer, _fill_from_env
from dreamer.models import ConvDecoder, ConvEncoder


def _collect_frames(n: int, size: int, seed: int) -> np.ndarray:
    buf = ReplayBuffer(capacity=n, obs_shape=(size, size, 3), num_actions=21, seq_len=2)
    _fill_from_env(buf, n, seed)
    return buf.image[: buf.size].copy()      # (N,H,W,3) uint8


def _to_chw(frames_u8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.as_tensor(frames_u8, dtype=torch.float32, device=device) / 255.0
    return t.permute(0, 3, 1, 2).contiguous()   # (N,3,H,W)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"[ae] device={device} collecting {args.frames} frames…")
    frames = _collect_frames(args.frames, args.size, args.seed)

    n_eval = max(16, args.frames // 10)
    train_u8, eval_u8 = frames[:-n_eval], frames[-n_eval:]
    eval_x = _to_chw(eval_u8, device)
    print(f"[ae] train={len(train_u8)} eval={len(eval_u8)} frames")

    enc = ConvEncoder(args.size, depth=args.depth, embed_dim=args.embed_dim).to(device)
    dec = ConvDecoder(args.size, depth=args.depth, feat_dim=args.embed_dim).to(device)
    n_params = sum(q.numel() for q in [*enc.parameters(), *dec.parameters()])
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=args.lr)
    print(f"[ae] params={n_params/1e6:.1f}M  bottleneck={args.embed_dim}")

    # baseline: PSNR of predicting the mean frame (the "blur everything" floor)
    mean_frame = _to_chw(train_u8, device).mean(0, keepdim=True)
    base_psnr = _psnr(mean_frame.expand_as(eval_x), eval_x)

    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train_u8), args.batch)
        x = _to_chw(train_u8[idx], device)
        recon = dec(enc(x))
        loss = F.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                ev = _psnr(dec(enc(eval_x)), eval_x)
            print(f"  step {step:5d}  train_mse {loss.item():.5f}  eval_psnr {ev:5.2f} dB")

    # Final held-out comparison grid: inputs (top) vs reconstructions (bottom).
    with torch.no_grad():
        k = min(8, len(eval_u8))
        x = eval_x[:k]
        recon = dec(enc(x)).clamp(0, 1)
        final_psnr = _psnr(recon, x)
        top = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        bot = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        grid = np.concatenate([np.concatenate(list(top), axis=1),
                               np.concatenate(list(bot), axis=1)], axis=0)
    import imageio
    from dreamer import out_path
    out = out_path("ae_recon.png")
    imageio.imwrite(out, grid)
    print(f"[ae] eval PSNR {final_psnr:.2f} dB  (mean-frame baseline {base_psnr:.2f} dB)")
    print(f"[ae] grid (top=input, bottom=recon) → {out}")


if __name__ == "__main__":
    main()
