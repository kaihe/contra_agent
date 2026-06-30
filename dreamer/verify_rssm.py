"""Component 3b gate — RSSM dynamics.

Trains the WorldModel (encoder + RSSM + decoder, recon + KL) on whole-level
traces, saves the checkpoint, then runs the three-way rollout comparison
(`dreamer.compare_modes`) at random anchors of a held-out trace:

  REAL (game engine)  vs  CLOSED-loop (posterior)  vs  OPEN-loop (prior)

Gate: CLOSED-loop matches REAL everywhere (closed_mse≈0); OPEN-loop tracks REAL
over the imagination horizon with open_motion ≈ real_motion before drifting. A
frozen open-loop (open_motion « real_motion) fails — that's the failure the old
MSE-only gate hid.

    python -m dreamer.verify_rssm --train_traces 8 --steps 6000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dreamer.buffer import ReplayBuffer
from dreamer.collect import fill_buffer_from_traces, trace_paths
from dreamer.compare_modes import run_comparison
from dreamer.world_model import WorldModel
from dreamer import out_path


def _buffer_from(paths, size, seq_len, device):
    lengths = [len(np.load(p, allow_pickle=True)["actions"]) for p in paths]
    buf = ReplayBuffer(capacity=sum(lengths) + 16, obs_shape=(size, size, 3),
                       num_actions=21, seq_len=seq_len, device=device)
    fill_buffer_from_traces(buf, paths, verbose=False)
    return buf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_traces", type=int, default=8)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--n_anchors", type=int, default=10)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--deter", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    all_paths = trace_paths(1)
    train_paths = all_paths[: args.train_traces]
    eval_trace = all_paths[args.train_traces]                    # held-out, for the comparison
    print(f"[rssm] device={device}  train_traces={len(train_paths)}  eval={eval_trace.split('/')[-1]}")
    train_buf = _buffer_from(train_paths, args.size, args.seq_len, device)
    print(f"[rssm] train={train_buf.size} frames")

    wm = WorldModel(size=args.size, deter=args.deter).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
    print(f"[rssm] params={sum(q.numel() for q in wm.parameters())/1e6:.1f}M  "
          f"deter={args.deter} stoch=32x32  context={args.context} horizon={args.horizon}")

    for step in range(1, args.steps + 1):
        loss, m, _ = wm.loss(train_buf.sample(args.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d}  loss {m['loss']:.1f}  recon {m['recon']:.1f}  "
                  f"kl {m['kl']:.2f} (dyn {m['dyn']:.2f} rep {m['rep']:.2f})")

    ckpt = out_path("world_model.pt")
    torch.save(wm.state_dict(), ckpt)
    print(f"\n  saved world model → {ckpt}")

    wm.eval()
    run_comparison(wm, eval_trace, n_anchors=args.n_anchors, context=args.context,
                   horizon=args.horizon, size=args.size, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
