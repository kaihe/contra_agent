"""Component 3a verification gate — does the conv encoder/decoder represent
Contra frames through a compact latent?

Two modes:

* train (default): collect real frames, train ConvEncoder→ConvDecoder as a plain
  autoencoder, report held-out PSNR, dump an input/reconstruction grid. The gate:
  reconstructions match the inputs by eye and PSNR beats the blur-everything floor.

      python -m dreamer.verify_ae --frames 3000 --steps 4000

* --ckpt: skip training, load a trained model (e.g. dreamer.pretrain_ae's
  ae_pretrained.pt) and dump reconstructions on 20 frames sampled uniformly
  (evenly spaced) along one MC trace — start→end of the level. Each frame is a row
  of [input | recon | entity-heatmap overlay], rows stacked vertically. The
  overlay draws the EntityHead's four predicted heatmaps (player/enemies and their
  bullets) onto the frame, so you can see the entities are localized in the latent
  even where the recon column blurs the small sprites.

      python -m dreamer.verify_ae --ckpt tmp/dreamer/ae_pretrained.pt --level 1
      python -m dreamer.verify_ae --ckpt tmp/dreamer/ae_pretrained.pt \
          --trace tmp/mc_trace/level1/win_level1_20260630190411_i0.npz --k 20
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer.buffer import ReplayBuffer, _fill_from_env
from dreamer.collect import iter_trace_frames, trace_paths
from dreamer.models import ConvDecoder, ConvEncoder, EntityHead


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


def _pick_trace(level: int, seed: int) -> str:
    """Choose one trace file for a level (random, seeded for reproducibility)."""
    paths = trace_paths(level)
    if not paths:
        raise SystemExit(f"no traces at tmp/mc_trace/level{level}/")
    return paths[int(np.random.default_rng(seed).integers(len(paths)))]


def _sample_trace_frames(path: str, size: int, k: int) -> np.ndarray:
    """Uniformly sample `k` frames evenly spaced along a single trace (start→end).

    Even spacing covers the whole level playthrough; frame count is known from the
    action length up-front, so only the `k` target frames are kept in RAM.
    """
    n = len(np.load(path, allow_pickle=True)["actions"])          # frames at stride 1
    targets = set(np.linspace(0, n - 1, min(k, n)).astype(int).tolist())
    frames = [frame for i, (frame, _r, _t) in
              enumerate(iter_trace_frames([path], size=size, stride=1)) if i in targets]
    return np.asarray(frames)


def _recon_grid(x: torch.Tensor, recon: torch.Tensor) -> np.ndarray:
    """Per-frame [input | reconstruction] pairs stacked vertically (tall, not wide)."""
    inp = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    rec = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    pairs = [np.concatenate([inp[i], rec[i]], axis=1) for i in range(len(inp))]
    return np.concatenate(pairs, axis=0)


# entity heatmap overlay colors (RGB), matching contra.entities.annotate
_HEAT_COLORS = {
    0: (0, 255, 0),      # player         → green
    1: (0, 255, 255),    # player_bullets → cyan
    2: (255, 0, 0),      # enemies        → red
    3: (255, 255, 0),    # enemy_bullets  → yellow
}


def _overlay_heatmaps(frame_u8: np.ndarray, heat: np.ndarray, size: int) -> np.ndarray:
    """Alpha-blend the 4 predicted entity heatmaps onto a frame, colored by class."""
    import cv2
    out = frame_u8.astype(np.float32)
    for c, color in _HEAT_COLORS.items():
        m = np.clip(cv2.resize(heat[c], (size, size), interpolation=cv2.INTER_LINEAR),
                    0, 1)[..., None]
        out = out * (1 - m) + np.array(color, dtype=np.float32) * m
    return out.astype(np.uint8)


def _visualize_ckpt(args, device: str) -> None:
    """Load a trained model; dump per-frame [input | recon | entity-heatmap overlay].

    The overlay comes from the checkpoint's EntityHead — the four occupancy
    heatmaps (player, player_bullets, enemies, enemy_bullets) decoded from the same
    embedding the decoder reads. Sharp blobs here despite blurry pixels in the
    recon column is the point: entity positions live in the latent even though pixel
    MSE won't render the small sprites crisply.
    """
    import imageio
    from dreamer import out_path

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    size, embed_dim, depth, grid = cfg["size"], cfg["embed_dim"], cfg["depth"], cfg["grid"]
    enc = ConvEncoder(size, depth=depth, embed_dim=embed_dim).to(device).eval()
    dec = ConvDecoder(size, depth=depth, feat_dim=embed_dim).to(device).eval()
    head = EntityHead(embed_dim, n_classes=4, grid=grid, depth=depth).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    head.load_state_dict(ckpt["entity_head"])
    print(f"[ae] loaded {args.ckpt}  (size={size} embed={embed_dim} depth={depth} grid={grid})")

    path = args.trace or _pick_trace(args.level, args.seed)
    print(f"[ae] sampling {args.k} frames uniformly along {path.split('/')[-1]}…")
    frames = _sample_trace_frames(path, size, args.k)
    x = _to_chw(frames, device)
    with torch.no_grad():
        emb = enc(x)
        recon = dec(emb).clamp(0, 1)
        heat = head(emb).cpu().numpy()                          # (N,4,grid,grid)
        psnr = _psnr(recon, x)

    inp = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    rec = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    rows = [np.concatenate([inp[i], rec[i], _overlay_heatmaps(inp[i], heat[i], size)], axis=1)
            for i in range(len(frames))]                        # each row: 3 tiles wide

    out = out_path("ae_ckpt_recon.png")
    imageio.imwrite(out, np.concatenate(rows, axis=0))
    print(f"[ae] {len(frames)} frames  encode→decode PSNR {psnr:.2f} dB")
    print("[ae] grid rows=frames, cols = input | recon | entity heatmaps "
          "(player=green, p_bullet=cyan, enemy=red, e_bullet=yellow)")
    print(f"[ae] → {out}")


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
    p.add_argument("--ckpt", type=str, default=None,
                   help="visualize a trained model's reconstructions instead of training")
    p.add_argument("--trace", type=str, default=None,
                   help="specific trace .npz to sample from (--ckpt); default: random for --level")
    p.add_argument("--level", type=int, default=1, help="trace level for --ckpt sampling")
    p.add_argument("--k", type=int, default=20, help="frames to sample from the trace (--ckpt)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.ckpt:
        _visualize_ckpt(args, device)
        return

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

    # Final held-out grid: per-frame [input | recon] pairs, stacked tall.
    with torch.no_grad():
        k = min(8, len(eval_u8))
        x = eval_x[:k]
        recon = dec(enc(x)).clamp(0, 1)
        final_psnr = _psnr(recon, x)
        grid = _recon_grid(x, recon)
    import imageio
    from dreamer import out_path
    out = out_path("ae_recon.png")
    imageio.imwrite(out, grid)
    print(f"[ae] eval PSNR {final_psnr:.2f} dB  (mean-frame baseline {base_psnr:.2f} dB)")
    print(f"[ae] grid (rows=frames, left=input, right=recon) → {out}")


if __name__ == "__main__":
    main()
