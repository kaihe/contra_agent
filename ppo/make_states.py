"""Generate multi-state training anchors from winning Contra traces.

For each winning trace, replay it through the emulator and write anchors as
gzipped emulator savestates named:

    level<N>_<timestamp>_<x|s><coord:04d>.state

Anchor placement is level-aware:
  - "forward" levels (side-scroll): N anchors evenly along the approach, the
    last (boss-wall) one dropped.
  - "inside"/"up" levels (indoor/climb): one anchor at the entry of each room
    (first step at each screen number), the boss room included.

The progress coordinate is level-aware (same source of truth as the reward):
  - "forward" levels (side-scroll): horizontal xscroll  -> label "x"
  - "inside"/"up" levels (indoor/climb): screen number   -> label "s"
The level is read from RAM, so the same code handles every level — point it at
the right traces and it picks the right coordinate automatically.

Usage:
    python ppo/make_states.py --dry-run                 # level 1 defaults, preview
    python ppo/make_states.py --level 2                 # auto-pick level-2 traces
    python ppo/make_states.py --traces a.npz b.npz      # custom trace list
"""
import argparse
import glob
import gzip
import os
import re
import sys

import numpy as np

import contra  # noqa: F401  registers the custom ROM integration
import stable_retro as retro
from contra.events import ADDR_LEVEL, ADDR_XSCROLL_HI, level_advance_style

sys.path.insert(0, os.path.dirname(__file__))
from contra_wrapper import xscroll  # noqa: E402

GAME = "Contra-Nes"
SKIP = 3                       # traces were recorded at 60/3 = 20 fps
TRACE_DIR = "synthetic/mc_trace"
OUT_DIR = "ppo/states"

# Curated default trace lists per level (varied recording sessions). Levels not
# listed here fall back to evenly sampling `--num-traces` from all win traces.
DEFAULT_TRACES_BY_LEVEL = {
    1: [
        "win_level1_202603301145.npz",
        "win_level1_202603301703.npz",
        "win_level1_202604021858.npz",
        "win_level1_202604081140.npz",
        "win_level1_202604081539.npz",
    ],
}


def progress_coord(ram, style):
    """Level-aware progress coordinate used for anchor placement + naming."""
    if style == "forward":
        return xscroll(ram)
    return int(ram[ADDR_XSCROLL_HI])  # screen/room number for indoor & climb


def replay_snapshots(trace_path):
    """Replay a trace, returning (states, coords, level, style): the emulator
    savestate and the progress coordinate at every step, plus the final
    post-trace state, the 0-indexed level, and its advancement style."""
    d = np.load(trace_path, allow_pickle=True)
    actions = d["actions"]
    initial = bytes(d["initial_state"])

    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE, render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    env.em.set_state(initial)
    env.data.update_ram()

    level = int(env.unwrapped.get_ram()[ADDR_LEVEL])
    style = level_advance_style(level)

    states, coords = [], []
    for act in actions:
        states.append(env.em.get_state())              # state *before* this action
        coords.append(progress_coord(env.unwrapped.get_ram(), style))
        a = np.asarray(act, dtype=np.uint8)
        for _ in range(SKIP):
            env.step(a.copy())
    states.append(env.em.get_state())                  # final (post-levelup) state
    coords.append(progress_coord(env.unwrapped.get_ram(), style))
    env.close()
    return states, coords, level, style


def trace_timestamp(trace_path):
    base = os.path.basename(trace_path)
    m = re.match(r"win_level\d+_(.+)\.npz$", base)
    return m.group(1) if m else base.replace(".npz", "")


def gen_anchors(trace_path, n_anchors, drop_last, out_dir, dry_run):
    states, coords, level, style = replay_snapshots(trace_path)
    ts = trace_timestamp(trace_path)
    level_1 = level + 1
    label = "x" if style == "forward" else "s"
    # The progress coordinate increases over the approach and peaks when the
    # player reaches the boss (boss wall for "forward", boss room for "inside"
    # /"up"); after the levelup it resets, so cap sampling at boss-arrival.
    coords_arr = np.asarray(coords)
    boss_step = int(np.argmax(coords_arr))
    dropped = None
    if style == "forward":
        # Side-scroll: progress is continuous, so sample evenly along the
        # approach and drop the last anchor (it sits at the boss wall).
        idx = np.linspace(0, boss_step, n_anchors).astype(int)
        if drop_last:
            dropped = idx[-1]
            idx = idx[:-1]
    else:
        # Indoor/climb: progress is a small set of discrete rooms. Anchor the
        # entry of each room (first step at each screen number), including the
        # boss room — the boss scene is a meaningful start state here.
        first_step = {}
        for i in range(boss_step + 1):
            first_step.setdefault(int(coords_arr[i]), i)
        idx = np.array([first_step[c] for c in sorted(first_step)])

    saved = []
    used = set()
    for j in idx:
        name = f"level{level_1}_{ts}_{label}{int(coords[j]):04d}.state"
        if name in used:                # coord can repeat (e.g. one screen, many steps)
            name = f"level{level_1}_{ts}_{label}{int(coords[j]):04d}_t{int(j):04d}.state"
        used.add(name)
        path = os.path.join(out_dir, name)
        if not dry_run:
            with gzip.open(path, "wb") as f:
                f.write(states[j])
        saved.append((int(j), int(coords[j]), name))

    drop_info = (f"  (dropped boss step {dropped}, {label}={int(coords[dropped])})"
                 if dropped is not None else "")
    print(f"\n{os.path.basename(trace_path)}  L{level_1} {style}  "
          f"({len(states)} steps, boss@{boss_step}){drop_info}")
    for step, c, name in saved:
        print(f"  step {step:4d}  {label}{c:04d}  ->  {name}")
    return saved


def resolve_traces(args):
    """Return the list of trace paths to process."""
    if args.traces:
        traces = args.traces
    else:
        traces = DEFAULT_TRACES_BY_LEVEL.get(args.level)
        if traces is None:
            pool = sorted(glob.glob(os.path.join(TRACE_DIR, f"win_level{args.level}_*.npz")))
            if not pool:
                raise SystemExit(f"No traces found for level {args.level} in {TRACE_DIR}/")
            picks = sorted(set(np.linspace(0, len(pool) - 1, args.num_traces).astype(int).tolist()))
            traces = [pool[i] for i in picks]
    traces = [t if os.path.sep in t else os.path.join(TRACE_DIR, t) for t in traces]
    for t in traces:
        if not os.path.isfile(t):
            raise FileNotFoundError(t)
    return traces


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1,
                   help="Level to pick default traces for (ignored if --traces given)")
    p.add_argument("--num-traces", type=int, default=5,
                   help="How many traces to sample when no curated default exists")
    p.add_argument("--traces", nargs="+", default=None,
                   help="Trace npz paths or basenames (overrides --level selection)")
    p.add_argument("--n-anchors", type=int, default=10,
                   help="Anchors sampled per trace before dropping the last")
    p.add_argument("--keep-last", action="store_true",
                   help="Keep the final (boss/transition) anchor")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Print sampled positions without writing files")
    args = p.parse_args()

    traces = resolve_traces(args)

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for t in traces:
        total += len(gen_anchors(t, args.n_anchors, not args.keep_last,
                                 args.out_dir, args.dry_run))
    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {total} anchors from {len(traces)} traces to {args.out_dir}/")


if __name__ == "__main__":
    main()
