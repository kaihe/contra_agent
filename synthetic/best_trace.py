"""
Find the most efficient (fewest actions) winning mc_trace for each level,
and plot the action-count distribution per level.

Usage:
    python synthetic/best_trace.py [--trace-dir synthetic/mc_trace]
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=os.path.join(os.path.dirname(__file__), "mc_trace"))
    args = parser.parse_args()

    pattern = re.compile(r"win_level(\d+)_")
    best: dict[int, tuple[int, str]] = {}   # level -> (n_actions, path)
    counts: dict[int, list[int]] = defaultdict(list)  # level -> [n_actions, ...]

    for fname in sorted(os.listdir(args.trace_dir)):
        if not fname.endswith(".npz"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        level = int(m.group(1))
        path = os.path.join(args.trace_dir, fname)
        data = np.load(path, allow_pickle=True)
        n = len(data["actions"])
        counts[level].append(n)
        if level not in best or n < best[level][0]:
            best[level] = (n, path)

    if not best:
        print("No winning traces found.")
        return

    print(f"{'Level':>5}  {'Actions':>7}  File")
    print("-" * 60)
    for level in sorted(best):
        n, path = best[level]
        print(f"{level:>5}  {n:>7}  {os.path.basename(path)}")

    levels = sorted(counts)
    ncols = 4
    nrows = (len(levels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for ax, level in zip(axes, levels):
        vals = counts[level]
        ax.hist(vals, bins=max(5, len(vals) // 2), color="steelblue", edgecolor="white")
        ax.axvline(best[level][0], color="tomato", linestyle="--", label=f"best={best[level][0]}")
        ax.set_title(f"Level {level}")
        ax.set_xlabel("# actions")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    for ax in axes[len(levels):]:
        ax.set_visible(False)

    fig.suptitle("Action-count distribution of winning traces per level", y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
