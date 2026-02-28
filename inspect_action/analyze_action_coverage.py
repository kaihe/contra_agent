"""
Action Table Coverage Analysis
================================

Picks one random win trace, runs best-effort matching against the current
ACTION_TABLE, then plots the human vs encoded time series for each of the
6 main buttons with the per-button difference ratio.

Usage:
    python analyze_action_coverage.py
    python analyze_action_coverage.py --seed 42
    python analyze_action_coverage.py --skip 3
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
from contra_wrapper import ACTION_TABLE, ACTION_NAMES

TRACE_DIR  = os.path.join(os.path.dirname(__file__), "human_trace")
SKIP       = 4
N_BUTTONS  = 9
BTN        = ["B", "NULL", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
MAIN_BTNS  = [0, 4, 5, 6, 7, 8]   # B, UP, DOWN, LEFT, RIGHT, A
SMOOTH_WIN = 30                    # rolling-average window for display (frames)


def expand_action(action_vec: np.ndarray, skip: int) -> np.ndarray:
    """Expand 9-element action into (9, skip) matrix; release B on last frame."""
    mat = np.tile(action_vec[:, np.newaxis], (1, skip))
    mat[0, skip - 1] = 0
    return mat


def best_match(human_chunk: np.ndarray, action_mats: np.ndarray) -> tuple[int, int]:
    """Find lowest-cost action for a (9, skip) human chunk. Returns (idx, bit_diff)."""
    diffs = np.bitwise_xor(action_mats, human_chunk[np.newaxis])  # (n, 9, skip)
    costs = diffs.sum(axis=(1, 2))
    best_idx = int(costs.argmin())
    return best_idx, int(costs[best_idx])


def encode_trace(actions: np.ndarray, action_mats: np.ndarray, skip: int) -> np.ndarray:
    """
    For every skip-frame chunk find the best match and reconstruct the full
    encoded frame sequence.

    Returns encoded: (n_chunks * skip, 9)
    """
    n_chunks = len(actions) // skip
    encoded  = np.empty((n_chunks * skip, N_BUTTONS), dtype=np.int8)
    for i in range(n_chunks):
        chunk = actions[i * skip : (i + 1) * skip].T   # (9, skip)
        best_idx, _ = best_match(chunk, action_mats)
        encoded[i * skip : (i + 1) * skip] = action_mats[best_idx].T
    return encoded


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Simple causal rolling mean."""
    kernel = np.ones(w) / w
    return np.convolve(x.astype(float), kernel, mode="same")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=SKIP)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for trace selection")
    args = parser.parse_args()

    skip = args.skip

    # Pre-expand all actions
    table       = np.array(ACTION_TABLE, dtype=np.int8)
    action_mats = np.stack([expand_action(table[i], skip) for i in range(len(table))])

    # Pick one random win trace
    win_files = sorted(glob.glob(os.path.join(TRACE_DIR, "win_*.npz")))
    if not win_files:
        print("No win traces found.")
        return
    rng      = np.random.default_rng(args.seed)
    fpath    = win_files[int(rng.integers(len(win_files)))]
    basename = os.path.basename(fpath)
    print(f"Trace: {basename}")

    # Load and encode
    d       = np.load(fpath)
    actions = d["actions"][1:].astype(np.int8)      # (N_frames, 9)
    encoded = encode_trace(actions, action_mats, skip)
    n       = len(encoded)
    human   = actions[:n]                            # trim to match encoded length

    # Per-button difference ratio
    diff    = np.bitwise_xor(human, encoded)         # (n, 9)

    print(f"\nFrames: {n}  |  skip={skip}  |  smooth window={SMOOTH_WIN}")
    print(f"\n{'Button':8s}  {'Human%':>7s}  {'Encod%':>7s}  {'Diff%':>7s}  "
          f"{'FP%':>7s}  {'FN%':>7s}")
    print("-" * 55)
    for j in MAIN_BTNS:
        h   = human[:, j].astype(float)
        e   = encoded[:, j].astype(float)
        d_r = diff[:, j].astype(float)
        fp  = ((encoded[:, j] == 1) & (human[:, j] == 0)).astype(float)
        fn  = ((encoded[:, j] == 0) & (human[:, j] == 1)).astype(float)
        print(f"{BTN[j]:8s}  {h.mean()*100:>6.1f}%  {e.mean()*100:>6.1f}%  "
              f"{d_r.mean()*100:>6.1f}%  {fp.mean()*100:>6.1f}%  {fn.mean()*100:>6.1f}%")

    # Plot
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    t = np.arange(n)
    fig, axes = plt.subplots(len(MAIN_BTNS), 1,
                             figsize=(14, 2.2 * len(MAIN_BTNS)), sharex=True)
    fig.suptitle(
        f"{basename}\n"
        f"Blue = human   Orange dashed = encoded   Grey fill = difference",
        fontsize=9,
    )

    for ax, j in zip(axes, MAIN_BTNS):
        h_sm = rolling_mean(human[:, j],   SMOOTH_WIN)
        e_sm = rolling_mean(encoded[:, j], SMOOTH_WIN)
        d_sm = rolling_mean(diff[:, j],    SMOOTH_WIN)

        diff_ratio = diff[:, j].mean()

        ax.plot(t, h_sm, color="steelblue",  lw=1.2, label="human")
        ax.plot(t, e_sm, color="darkorange", lw=1.2, linestyle="--", label="encoded")
        ax.fill_between(t, 0, d_sm, color="gray", alpha=0.35, label="diff")

        ax.set_ylabel(BTN[j], fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks([0, 0.5, 1])
        ax.legend(loc="upper right", fontsize=7, ncol=3,
                  title=f"diff={diff_ratio*100:.1f}%", title_fontsize=7)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
