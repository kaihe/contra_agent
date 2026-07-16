"""Pretrain a frozen encoder from game traces, with a 4-class entity aux loss.

Motivation
----------
In the joint Dreamer loss the reward/KL/continue gradients are what pull small,
task-critical entities (bullets are ~2px) into the latent — pixel reconstruction
alone ignores them. If we want to PRETRAIN the encoder on traces and FREEZE it,
we lose that gradient, so a recon-only frozen encoder goes entity-blind (the
death-blindness failure). This script replaces the missing signal with an
explicit supervised one: alongside frame reconstruction, an EntityHead must
decode four occupancy heatmaps from the *embedding* —

    player · player_bullets · enemies · enemy_bullets   (contra.entities)

Ground truth comes free from the aligned RAM captured while replaying the winning
MC traces (whole-level coverage). Frames are taken at native NES fidelity (default
256×256; the stable_retro screen is a 240×224 overscan crop of the 256×240 PPU) so
a 2px bullet survives the resize.

Data pipeline
-------------
The full level-1 trace set is ~650k frames = 128GB at native res — it can't live
in RAM. So we replay the traces ONCE through the single shared emulator, streaming
frames into a disk-backed uint8 memmap (materialize_traces). Training then reads
random batches from that memmap via a DataLoader with worker processes; the tiny
RAM snapshots stay in memory and heatmap targets are computed on the fly (so they
never hit disk and --grid/--sigma stay tunable without re-collecting). The
emulator is only touched during materialization, never in a worker — respecting
stable_retro's one-instance-per-process rule.

    python -m dreamer.pretrain_ae --level 1 --steps 8000
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from contra.entities import HEATMAP_CLASSES, entity_heatmaps
from dreamer import OUT_DIR, out_path
from dreamer.collect import iter_trace_frames, trace_paths
from dreamer.models import ConvDecoder, ConvEncoder, EntityHead


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


# ── data ──────────────────────────────────────────────────────────────────────

def materialize_traces(level: int, size: int, stride: int, max_traces: int | None):
    """Replay all `level` traces once → disk memmap of frames + in-RAM rams/tids.

    Cached by (level, size, stride, n_traces): a re-run with matching params skips
    the emulator entirely. Returns (frames_path, frames_memmap, rams, tids).
    """
    paths = trace_paths(level)
    if max_traces:
        paths = paths[:max_traces]
    if not paths:
        raise SystemExit(f"no traces at tmp/mc_trace/level{level}/")

    # N is known from action lengths without replaying (see iter_trace_frames).
    N = int(sum(math.ceil(len(np.load(p, allow_pickle=True)["actions"]) / stride)
                for p in paths))

    cache = os.path.join(OUT_DIR, "ae_cache")
    os.makedirs(cache, exist_ok=True)
    tag = f"L{level}_s{size}_st{stride}_n{len(paths)}"
    fpath = os.path.join(cache, f"frames_{tag}.npy")
    rpath = os.path.join(cache, f"rams_{tag}.npy")
    tpath = os.path.join(cache, f"tids_{tag}.npy")

    if all(os.path.exists(q) for q in (fpath, rpath, tpath)):
        frames = np.lib.format.open_memmap(fpath, mode="r")
        if frames.shape == (N, size, size, 3):
            print(f"[pretrain] cache hit: {frames.shape} frames at {fpath}")
            return fpath, frames, np.load(rpath), np.load(tpath)

    gb = N * size * size * 3 / 1e9
    print(f"[pretrain] materializing {N} frames → {gb:.1f}GB memmap at {fpath} …")
    frames = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.uint8,
                                       shape=(N, size, size, 3))
    rams = np.zeros((N, 2048), dtype=np.uint8)
    tids = np.zeros(N, dtype=np.int64)
    i = 0
    for frame, ram, ti in iter_trace_frames(paths, size, stride, max_traces):
        frames[i], rams[i], tids[i] = frame, ram, ti
        i += 1
        if i % 10000 == 0:
            print(f"  {i}/{N} frames…")
    frames.flush()
    np.save(rpath, rams)
    np.save(tpath, tids)
    print(f"[pretrain] materialized {i} frames from {len(np.unique(tids))} traces")
    return fpath, frames, rams, tids


class TraceDataset(Dataset):
    """Frames from the disk memmap + on-the-fly entity heatmaps from stored RAM.

    The memmap is opened lazily per worker (each mmaps the shared file — no copy);
    `rams` is small and inherited copy-on-write via fork.
    """

    def __init__(self, fpath, rams, indices, grid, sigma):
        self.fpath, self.rams = fpath, rams
        self.indices = indices
        self.grid, self.sigma = grid, sigma
        self._frames = None

    def _frames_mm(self):
        if self._frames is None:
            self._frames = np.lib.format.open_memmap(self.fpath, mode="r")
        return self._frames

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x = torch.from_numpy(self._frames_mm()[i].astype(np.float32) / 255.0).permute(2, 0, 1)
        h = entity_heatmaps(self.rams[i], grid=self.grid, sigma=self.sigma)
        return x.contiguous(), torch.from_numpy(h)


class StreamDataset(torch.utils.data.IterableDataset):
    """Replay train traces through one emulator (in the worker process) and yield
    (frame_chw, heatmap) through a reservoir shuffle buffer — no disk, no full-set
    RAM. Single emulator per process → use num_workers=1. The buffer holds uint8
    frames + RAM (~194KB each) and decorrelates the otherwise-sequential replay.
    """

    def __init__(self, paths, size, stride, grid, sigma, buffer=8192, seed=0):
        self.paths, self.size, self.stride = paths, size, stride
        self.grid, self.sigma, self.buffer, self.seed = grid, sigma, buffer, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        buf: list = []
        while True:                                     # epochs, forever
            order = list(self.paths)
            rng.shuffle(order)
            for frame, ram, _ in iter_trace_frames(order, self.size, self.stride):
                if len(buf) < self.buffer:
                    buf.append((frame, ram))
                    continue
                j = int(rng.integers(len(buf)))
                fr, rm = buf[j]
                buf[j] = (frame, ram)
                x = torch.from_numpy(fr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                yield x, torch.from_numpy(entity_heatmaps(rm, self.grid, self.sigma))


def _print_presence(rams_u8, grid, sigma):
    occ = np.stack([entity_heatmaps(r, grid, sigma) for r in rams_u8]
                   ).reshape(len(rams_u8), 4, -1).max(-1).mean(0)
    print("[pretrain] class presence: " +
          "  ".join(f"{n}={occ[c]:.3f}" for c, n in enumerate(HEATMAP_CLASSES)))


def _to_device_eval(frames_u8, rams_u8, args, device):
    x = torch.from_numpy(frames_u8.astype(np.float32) / 255.0
                         ).permute(0, 3, 1, 2).contiguous().to(device)
    h = torch.from_numpy(np.stack([entity_heatmaps(r, args.grid, args.sigma)
                                   for r in rams_u8])).to(device)
    return x, h


def _memmap_data(args, device, rng):
    """Default path: materialize to disk memmap, random-access via workers."""
    fpath, frames, rams, tids = materialize_traces(args.level, args.size, args.stride,
                                                   args.max_traces)
    uniq = np.unique(tids)
    eval_traces = set(rng.choice(uniq, size=min(args.n_eval_traces, len(uniq) - 1),
                                 replace=False).tolist())
    ev = np.array([t in eval_traces for t in tids])
    train_idx, eval_idx = np.where(~ev)[0], np.where(ev)[0]
    print(f"[pretrain] train={len(train_idx)} eval={len(eval_idx)} frames "
          f"({len(eval_traces)} eval traces held out)")
    _print_presence(rams[rng.choice(train_idx, min(2000, len(train_idx)), replace=False)],
                    args.grid, args.sigma)
    ds = TraceDataset(fpath, rams, train_idx, args.grid, args.sigma)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    esel = eval_idx[np.linspace(0, len(eval_idx) - 1, min(512, len(eval_idx))).astype(int)]
    eval_x, eval_h = _to_device_eval(frames[esel], rams[esel], args, device)
    return loader, eval_x, eval_h


def _stream_data(args, device, rng):
    """--stream path: replay traces on the fly, no disk. Eval traces held in RAM."""
    paths = trace_paths(args.level)
    if args.max_traces:
        paths = paths[:args.max_traces]
    if not paths:
        raise SystemExit(f"no traces at tmp/mc_trace/level{args.level}/")
    eval_ti = set(rng.choice(len(paths), size=min(args.n_eval_traces, len(paths) - 1),
                             replace=False).tolist())
    eval_paths = [p for i, p in enumerate(paths) if i in eval_ti]
    train_paths = [p for i, p in enumerate(paths) if i not in eval_ti]
    ef, er = [], []
    for frame, ram, _ in iter_trace_frames(eval_paths, args.size, args.stride):
        ef.append(frame); er.append(ram)
    ef, er = np.asarray(ef), np.asarray(er)
    print(f"[pretrain] stream: {len(train_paths)} train traces on the fly; "
          f"{len(eval_paths)} eval traces → {len(ef)} eval frames in RAM")
    _print_presence(er, args.grid, args.sigma)
    if args.workers != 1:
        print("[pretrain] stream uses num_workers=1 (one emulator per process)")
    ds = StreamDataset(train_paths, args.size, args.stride, args.grid, args.sigma,
                       buffer=args.buffer, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=1, pin_memory=True)
    esel = np.linspace(0, len(ef) - 1, min(512, len(ef))).astype(int)
    eval_x, eval_h = _to_device_eval(ef[esel], er[esel], args, device)
    return loader, eval_x, eval_h


def _heat_to_rgb(hm: np.ndarray, size: int) -> np.ndarray:
    """(grid,grid) in [0,1] → (size,size,3) uint8 grayscale (nearest upscale)."""
    import cv2
    up = cv2.resize((hm * 255).astype(np.uint8), (size, size),
                    interpolation=cv2.INTER_NEAREST)
    return np.repeat(up[..., None], 3, axis=2)


# ── train ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--steps", type=int, default=8000, help="gradient updates (not epochs)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--size", type=int, default=256, help="native-ish square input")
    p.add_argument("--stride", type=int, default=8,
                   help="record every Nth decision frame (traces are highly redundant)")
    p.add_argument("--workers", type=int, default=16, help="DataLoader worker processes")
    p.add_argument("--stream", action="store_true",
                   help="stream frames from the emulator on the fly (no disk memmap)")
    p.add_argument("--buffer", type=int, default=8192, help="stream shuffle-buffer size")
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--grid", type=int, default=32, help="heatmap resolution")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian blob width (cells)")
    p.add_argument("--pos_weight", type=float, default=10.0, help="entity loss weight")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_traces", type=int, default=None)
    p.add_argument("--n_eval_traces", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ae_pretrained.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(args.seed)
    loader, eval_x, eval_h = (_stream_data if args.stream else _memmap_data)(args, device, rng)

    enc = ConvEncoder(args.size, depth=args.depth, embed_dim=args.embed_dim).to(device)
    dec = ConvDecoder(args.size, depth=args.depth, feat_dim=args.embed_dim).to(device)
    head = EntityHead(args.embed_dim, n_classes=4, grid=args.grid, depth=args.depth).to(device)
    params = [*enc.parameters(), *dec.parameters(), *head.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    print(f"[pretrain] params={sum(q.numel() for q in params)/1e6:.1f}M  "
          f"embed={args.embed_dim}  pos_weight={args.pos_weight}  workers={args.workers}")

    def _run_eval():
        enc.eval(); dec.eval(); head.eval()
        with torch.no_grad():
            emb = enc(eval_x)
            psnr = _psnr(dec(emb), eval_x)
            hp = head(emb)
            per_cls = ((hp - eval_h) ** 2).mean(dim=(0, 2, 3))     # MSE per class, all 4
        enc.train(); dec.train(); head.train()
        return psnr, per_cls

    step = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            x, h_gt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, h_gt = next(data_iter)
        x = x.to(device, non_blocking=True)
        h_gt = h_gt.to(device, non_blocking=True)
        embed = enc(x)
        recon_loss = F.mse_loss(dec(embed), x)
        pos_loss = F.mse_loss(head(embed), h_gt)
        loss = recon_loss + args.pos_weight * pos_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 500 == 0 or step == 1:
            psnr, per_cls = _run_eval()
            cls = " ".join(f"{n[:4]}={per_cls[c]:.4f}" for c, n in enumerate(HEATMAP_CLASSES))
            print(f"  step {step:5d}  recon {recon_loss.item():.4f}  pos {pos_loss.item():.4f}"
                  f"  | eval PSNR {psnr:5.2f}dB  heatmap MSE [{cls}]")

    # ── viz + save ──────────────────────────────────────────────────────────────
    import imageio
    enc.eval(); dec.eval(); head.eval()
    with torch.no_grad():
        k = min(8, eval_x.shape[0])
        recon = dec(enc(eval_x[:k])).clamp(0, 1)
        top = (eval_x[:k].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        bot = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        # per-frame [input | recon] pairs stacked vertically (tall, not wide).
        recon_grid = np.concatenate([np.concatenate([top[i], bot[i]], 1)
                                     for i in range(k)], 0)
        m = min(4, eval_x.shape[0])
        hp = head(enc(eval_x[:m])).cpu().numpy()
    recon_out = out_path("pretrain_recon.png")
    imageio.imwrite(recon_out, recon_grid)

    rows = []
    for i in range(m):
        inp = (eval_x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        tiles = [inp]
        for c in range(4):
            tiles.append(_heat_to_rgb(eval_h[i, c].cpu().numpy(), args.size))
            tiles.append(_heat_to_rgb(hp[i, c], args.size))
        rows.append(np.concatenate(tiles, 1))
    heat_out = out_path("pretrain_heatmaps.png")
    imageio.imwrite(heat_out, np.concatenate(rows, 0))

    ckpt_out = out_path(args.out)
    torch.save({
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "entity_head": head.state_dict(),
        "config": {"size": args.size, "embed_dim": args.embed_dim, "depth": args.depth,
                   "grid": args.grid, "classes": list(HEATMAP_CLASSES)},
    }, ckpt_out)

    print(f"\n[pretrain] recon grid  → {recon_out}")
    print(f"[pretrain] heatmap panel (cols: input, then GT|pred per class "
          f"{HEATMAP_CLASSES}) → {heat_out}")
    print(f"[pretrain] frozen weights → {ckpt_out}")


if __name__ == "__main__":
    main()
