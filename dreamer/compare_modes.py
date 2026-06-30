"""Three-way rollout comparison at random anchors — the C3b (RSSM) gate visual.

At each of N random anchor points in a held-out trace, build a `horizon`-step
rollout under three regimes and stack them as a TALL, scrollable grid:

  columns  = REAL (game engine) | CLOSED-loop (posterior) | OPEN-loop (prior)
  rows     = timestep 0 (top) → horizon-1 (bottom)

All three share the same `context`-frame warmup, so they start from the same
state. Read it by scrolling down: CLOSED should stay glued to REAL the whole way;
OPEN should track then visibly drift/slow (the freeze).

`run_comparison` is imported by dreamer.verify_rssm (which trains + saves the
model first). Standalone, this loads tmp/dreamer/world_model.pt — no retraining:

    python -m dreamer.compare_modes --n_anchors 10
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from dreamer.buffer import ReplayBuffer
from dreamer.collect import fill_buffer_from_traces, trace_paths
from dreamer.world_model import WorldModel
from dreamer import out_path


def _load_trace(trace_path, size, device):
    n = len(np.load(trace_path, allow_pickle=True)["actions"])
    buf = ReplayBuffer(capacity=n + 4, obs_shape=(size, size, 3),
                       num_actions=21, seq_len=2, device=device)
    fill_buffer_from_traces(buf, [trace_path], verbose=False)
    N = buf.size
    frames = torch.as_tensor(buf.image[:N], dtype=torch.float32, device=device) / 255.0
    onehot = torch.nn.functional.one_hot(
        torch.as_tensor(buf.action[:N], dtype=torch.long, device=device), 21).float()
    return frames, onehot, buf.is_terminal[:N].copy()


def _pick_anchors(frames, terminal, context, H, n_anchors, seed):
    """Random anchors with full context+horizon room, no terminal in the window,
    and real motion above the median (active gameplay)."""
    N = len(frames)
    motion = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    rng = np.random.default_rng(seed)
    valid = [t0 for t0 in range(context, N - H)
             if not terminal[t0 - context: t0 + H].any()
             and motion[t0: t0 + H - 1].mean() >= np.median(motion)]
    rng.shuffle(valid)
    return sorted(valid[:n_anchors])


def _rollout(wm, frames, onehot, t0, context, H, device):
    """Return (real, closed, open) each (H,3,h,w), sharing the context warmup."""
    fwin = frames[t0 - context: t0 + H].unsqueeze(0)
    awin = onehot[t0 - context: t0 + H].unsqueeze(0)
    first = torch.zeros(1, context + H, device=device)
    with torch.no_grad():
        posts, _ = wm.observe(fwin, awin, first)                  # closed-loop over whole window
        closed = wm.decode(wm.rssm.get_feat(posts))[0, context:context + H]
        state = {k: posts[k][:, context - 1] for k in posts}      # state at t0-1 (end of warmup)
        ia = onehot[t0 - 1: t0 - 1 + H].unsqueeze(0)
        openl = wm.decode(wm.rssm.imagine(state, ia))[0]          # open-loop from same warmup
    real = frames[t0: t0 + H].permute(0, 3, 1, 2)
    return real.cpu(), closed.clamp(0, 1).cpu(), openl.clamp(0, 1).cpu()


def _grid_vertical(real, closed, openl):
    """Tall grid: 3 columns (REAL/CLOSED/OPEN), one timestep per row, top→bottom.
    Left gutter shows the step index; a header labels the columns."""
    def u8(x):
        return (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    R, C, O = u8(real), u8(closed), u8(openl)
    H, h, w, _ = R.shape
    gut, sep = 34, 2
    vsep = np.zeros((h, sep, 3), np.uint8)
    rows = []
    for t in range(H):
        cells = np.concatenate([R[t], vsep, C[t], vsep, O[t]], axis=1)
        g = np.zeros((h, gut, 3), np.uint8)
        cv2.putText(g, str(t), (4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        rows.append(np.concatenate([g, cells], axis=1))
    body = np.concatenate(rows, axis=0)
    W = body.shape[1]
    header = np.zeros((24, W, 3), np.uint8)
    for k, name in enumerate(["REAL", "CLOSED", "OPEN"]):
        cv2.putText(header, name, (gut + k * (w + sep) + 6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def run_comparison(wm, trace_path, *, n_anchors=10, context=5, horizon=15,
                   size=128, seed=0, device="cuda"):
    """Generate the 3-way vertical comparison grids + per-anchor metric table."""
    import imageio
    frames, onehot, terminal = _load_trace(trace_path, size, device)
    anchors = _pick_anchors(frames, terminal, context, horizon, n_anchors, seed)
    print(f"\n[compare] {len(anchors)} anchors on {trace_path.split('/')[-1]}; "
          f"rows=time↓, cols = REAL / CLOSED-loop / OPEN-loop")
    print(f"  {'anchor':>6} {'closed_mse':>11} {'open_mse':>9} {'real_motion':>12} {'open_motion':>12}")
    for i, t0 in enumerate(anchors):
        real, closed, openl = _rollout(wm, frames, onehot, t0, context, horizon, device)
        closed_mse = ((closed - real) ** 2).mean().item()
        open_mse = ((openl - real) ** 2).mean().item()
        rm = (real[1:] - real[:-1]).abs().mean().item() * 255
        om = (openl[1:] - openl[:-1]).abs().mean().item() * 255
        path = out_path(f"compare_anchor{i:02d}_t{t0}.png")
        imageio.imwrite(path, _grid_vertical(real, closed, openl))
        print(f"  {t0:6d} {closed_mse:11.4f} {open_mse:9.4f} {rm:12.2f} {om:12.2f}   → {path}")
    print("  GATE: closed_mse≈0 (matches REAL); open tracks REAL with open_motion≈real_motion,")
    print("        then drifts. open_motion«real_motion = the freeze.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="tmp/dreamer/world_model.pt")
    p.add_argument("--trace", default=None, help="held-out trace npz (default: level1 #8)")
    p.add_argument("--n_anchors", type=int, default=10)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--deter", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trace = args.trace or trace_paths(1)[8]
    wm = WorldModel(size=args.size, deter=args.deter).to(device)
    wm.load_state_dict(torch.load(args.ckpt, map_location=device))
    wm.eval()
    print(f"[compare] loaded {args.ckpt}")
    run_comparison(wm, trace, n_anchors=args.n_anchors, context=args.context,
                   horizon=args.horizon, size=args.size, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
