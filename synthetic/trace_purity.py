"""
trace_purity.py — Measure how clean / economical a winning trace is.

A clean trace reaches the win with as little button activity as possible. In a
forward level Right is held almost every frame by necessity, so the cleanliness
signal is the *non-R* buttons (F, J, U, D, L — the tactical "extra" inputs). We
report two views, both "lower is better" among winning traces:

  * non-R button-frames — every frame a non-R button is held, i.e. how *long* you
    hold the extras. This is the main cleanliness signal: sustained gratuitous
    input (e.g. holding aim-up at empty sky) shows up here.
  * non-R press events  — 0->1 rising edges, i.e. how many *distinct* times you
    decide to jump / fire / aim (a held button counts once). This tracks the
    level's intrinsic tactical demand and barely moves between clean and noisy
    traces — what differs is how long each press is held.

(The old RAM-purity check that ran prune_actions was dropped: the stateful
sampler already guarantees no inert presses, so it saturated at ~100% and told us
nothing the button economy doesn't.)

Usage
-----
    python synthetic/trace_purity.py --input "tmp/mc_trace/level1/*.npz"
    python synthetic/trace_purity.py --input "tmp/mc_trace_old/level1/*.npz"
"""

import argparse
import glob
from collections import Counter

import numpy as np

from synthetic.action_configs.search_reward import BUTTON_BITS  # nickname (F/U/D/L/R/J) -> bit

# Combo label order: directions, then fire, then jump (matches the action table:
# UR, URF, RJ, …); '_' is the no-op.
_COMBO_ORDER = ["U", "D", "L", "R", "F", "J"]


def combo_label(row: np.ndarray) -> str:
    """Action-combo label for a 9-bit row, e.g. 'UR', 'RF', '_' for the no-op."""
    on = [n for n in _COMBO_ORDER if row[BUTTON_BITS[n]]]
    return "".join(on) if on else "_"

# Non-Right tactical bits: Right is the canonical forward action, excluded so the
# economy measures the "extra" inputs rather than just trace length.
_NONR_BITS = [bit for nick, bit in BUTTON_BITS.items() if nick != "R"]
_ALL_BITS = list(BUTTON_BITS.values())


def _rising_edges(col: np.ndarray) -> int:
    """Count 0->1 transitions in a boolean column (a leading 1 counts as a press)."""
    return int((col[1:] & ~col[:-1]).sum()) + int(col[0]) if len(col) else 0


def button_economy(actions: np.ndarray) -> dict:
    """Button-activity economy for one trace (pure action arithmetic).

    Returns held-frame counts and rising-edge press-event counts; non-R figures
    exclude the Right bit. Lower is cleaner among winning traces.
    """
    a = np.asarray(actions, dtype=np.uint8)
    return {
        "steps": len(a),
        "nonr_held": int(a[:, _NONR_BITS].sum()),
        "all_held": int(a[:, _ALL_BITS].sum()),
        "nonr_events": sum(_rising_edges(a[:, b].astype(bool)) for b in _NONR_BITS),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Trace NPZ or glob")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit(f"no trace matches {args.input!r}")

    rows = []
    combos = Counter()        # action-combo frequency pooled across all traces
    total_steps = 0
    for p in paths:
        actions = np.load(p, allow_pickle=True)["actions"]
        e = button_economy(actions)
        rows.append(e)
        for r in actions:
            combos[combo_label(r)] += 1
        total_steps += len(actions)
        print(f"  nonR {e['nonr_held']:>4} held / {e['nonr_events']:>3} events  "
              f"{e['steps']:>4} steps  {p.split('/')[-1]}")

    mean = lambda k: float(np.mean([r[k] for r in rows]))
    msteps = mean("steps")
    print(f"\n{len(paths)} trace(s)")
    print("  BUTTON ECONOMY (mean per win, lower = cleaner):")
    print(f"    steps to win:         {msteps:8.0f}")
    print(f"    non-R button-frames:  {mean('nonr_held'):8.0f}   (per step {mean('nonr_held')/msteps:.3f})")
    print(f"    non-R press events:   {mean('nonr_events'):8.0f}   (per step {mean('nonr_events')/msteps:.3f})")
    print(f"    all button-frames:    {mean('all_held'):8.0f}   (per step {mean('all_held')/msteps:.3f})")

    print(f"\n  ACTION DISTRIBUTION ({total_steps} steps, % of steps):")
    print(f"    {'combo':<6}{'count':>9}{'%':>7}")
    for combo, n in combos.most_common():
        print(f"    {combo:<6}{n:>9}{100*n/total_steps:>6.1f}")


if __name__ == "__main__":
    main()
