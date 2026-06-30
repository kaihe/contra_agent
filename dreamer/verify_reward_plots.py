"""Visualize the step-reward profile leading into every terminal.

Rebuilds the same C3c training buffer (traces + env deaths) as verify_reward,
finds all terminals, and draws one line plot per terminal: per-step reward vs.
step number over the last `window` steps of that episode (step 0 = the terminal).
WIN (levelup, big +reward) vs DEATH (negative penalty) are colored differently.

This shows what the reward head must predict: tiny progress rewards most steps,
then a large spike at the terminal — and how rare/imbalanced that spike is.

    python -m dreamer.verify_reward_plots --train_traces 8 --env_steps 100
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from dreamer.collect import trace_paths
from dreamer.verify_reward import _build_buffer
from dreamer import out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_traces", type=int, default=8)
    p.add_argument("--env_steps", type=int, default=6000)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--window", type=int, default=1800,
                   help="steps before terminal to show (default shows the full episode)")
    p.add_argument("--clip", type=float, default=0.0,
                   help="clip y-axis to ±clip so the terminal spike goes off-scale "
                        "and the tiny dense reward becomes visible (e.g. 2.0)")
    p.add_argument("--symlog", action="store_true",
                   help="symlog y-axis: shows tiny progress (~0.3) AND big spikes "
                        "(enemy +1, pickup +20, terminal) on one axis")
    p.add_argument("--value", action="store_true",
                   help="overlay the critic's TARGET: discounted return-to-go "
                        "V_t = Σ γ^(k-t) r_k (bold line), with reward faint behind")
    p.add_argument("--gamma", type=float, default=0.997, help="discount for --value")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = trace_paths(1)[: args.train_traces]
    buf = _build_buffer(paths, args.env_steps, args.size, 2, "cpu", args.seed)
    N = buf.size
    reward = buf.reward[:N]
    is_term = buf.is_terminal[:N]
    is_first = buf.is_first[:N]
    term_idx = np.where(is_term)[0]
    print(f"[reward-plots] {N} frames, {len(term_idx)} terminals")

    # Is the per-step reward really ~0, or just dwarfed by the terminal spikes?
    body = reward[~is_term & ~is_first]
    nz = np.abs(body) > 1e-6
    print(f"  non-terminal steps: {len(body)}  nonzero: {nz.mean()*100:.1f}%  "
          f"mean|r|={np.abs(body).mean():.3f}  max|r|={np.abs(body).max():.2f}  "
          f"(p50|r|={np.percentile(np.abs(body),50):.3f}, p90={np.percentile(np.abs(body),90):.3f})")

    # classify each terminal by its reward spike: big positive = WIN (levelup),
    # negative = DEATH.
    def kind(r):
        return ("WIN", "tab:green") if r > 0.5 else ("DEATH", "tab:red") if r < -1 else ("?", "tab:gray")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 4
    nrow = math.ceil(len(term_idx) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), squeeze=False)
    n_win = n_death = 0
    for ax in axes.flat:
        ax.axis("off")
    for k, ti in enumerate(term_idx):
        # walk back to episode start (most recent is_first) or `window`, whichever closer
        start = ti
        while start > 0 and not is_first[start] and ti - start < args.window:
            start -= 1
        rel = np.arange(start - ti, 1)               # negative .. 0 (terminal at 0)
        seg = reward[start: ti + 1]
        label, color = kind(reward[ti])
        n_win += label == "WIN"
        n_death += label == "DEATH"

        ax = axes.flat[k]
        ax.axis("on")
        ax.axhline(0, color="0.8", lw=0.6)
        if args.value:
            # discounted return-to-go (the critic's target), computed backward:
            # V_t = r_t + γ V_{t+1}; V_T = r_T (episode ends at the terminal).
            V = np.empty_like(seg)
            V[-1] = seg[-1]
            for t in range(len(seg) - 2, -1, -1):
                V[t] = seg[t] + args.gamma * V[t + 1]
            ax.plot(rel, V, color=color, lw=1.4, zorder=3)     # VALUE = bold line
            ax2 = ax.twinx()                                   # reward faint behind
            ax2.plot(rel, seg, color="0.6", lw=0.5, alpha=0.6)
            ax2.tick_params(labelsize=5, colors="0.6")
            ax.set_title(f"{label}  V0={V[0]:.1f}  termR={reward[ti]:.1f}",
                         fontsize=8, color=color)
        else:
            ax.plot(rel, seg, color=color, lw=0.6)
            ax.scatter([0], [reward[ti]], color=color, s=18, zorder=3)
            if args.symlog:
                ax.set_yscale("symlog", linthresh=0.1)
            elif args.clip > 0:
                ax.set_ylim(-args.clip, args.clip)
            ax.set_title(f"{label}  term r={reward[ti]:.1f}", fontsize=8, color=color)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Step reward into each terminal  ({n_win} WIN / {n_death} DEATH / "
                 f"{len(term_idx)-n_win-n_death} other)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_path("reward_terminals.png")
    fig.savefig(out, dpi=90)
    print(f"  {n_win} WIN, {n_death} DEATH terminals")
    print(f"  reward at terminals: min {reward[term_idx].min():.1f}  "
          f"max {reward[term_idx].max():.1f}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
