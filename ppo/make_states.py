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

Anchors are generated from a single win trace per level (the most recent one
under tmp/mc_trace/level<N>/, or an explicit --trace).

Usage:
    python ppo/make_states.py --dry-run            # level 1, newest trace, preview
    python ppo/make_states.py --level 3            # level-3 anchors from newest trace
    python ppo/make_states.py --trace a.npz        # explicit trace
"""
import argparse
import glob
import gzip
import os
import re

import numpy as np

import contra  # noqa: F401  registers the custom ROM integration
import stable_retro as retro
from contra.events import ADDR_LEVEL, level_advance_style
from contra.reward import progress_coord

GAME = "Contra-Nes"
SKIP = 3                       # traces were recorded at 60/3 = 20 fps
TRACE_DIR = "tmp/mc_trace"     # mc_search writes win_level<N>_*.npz under TRACE_DIR/level<N>/
OUT_DIR = "ppo/states"


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
        coords.append(progress_coord(env.unwrapped.get_ram()))
        a = np.asarray(act, dtype=np.uint8)
        for _ in range(SKIP):
            env.step(a.copy())
    states.append(env.em.get_state())                  # final (post-levelup) state
    coords.append(progress_coord(env.unwrapped.get_ram()))
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


def resolve_trace(args):
    """Return the single trace path to process.

    Uses --trace when given (a path, or a basename resolved under
    TRACE_DIR/level<N>/), otherwise auto-picks the most recent win trace for
    --level under TRACE_DIR/level<N>/.
    """
    if args.trace:
        path = args.trace
        if os.path.sep not in path:
            path = os.path.join(TRACE_DIR, f"level{args.level}", path)
    else:
        pool = sorted(glob.glob(
            os.path.join(TRACE_DIR, f"level{args.level}", f"win_level{args.level}_*.npz")))
        if not pool:
            raise SystemExit(
                f"No traces found for level {args.level} in {TRACE_DIR}/level{args.level}/")
        path = pool[-1]  # most recent timestamp
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1,
                   help="Level whose newest trace to use (ignored if --trace given)")
    p.add_argument("--trace", default=None,
                   help="Trace npz path or basename (overrides --level auto-pick)")
    p.add_argument("--n-anchors", type=int, default=10,
                   help="Anchors sampled per trace before dropping the last")
    p.add_argument("--keep-last", action="store_true",
                   help="Keep the final (boss/transition) anchor")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Print sampled positions without writing files")
    args = p.parse_args()

    trace = resolve_trace(args)

    os.makedirs(args.out_dir, exist_ok=True)
    total = len(gen_anchors(trace, args.n_anchors, not args.keep_last,
                            args.out_dir, args.dry_run))
    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {total} anchors from {os.path.basename(trace)} to {args.out_dir}/")


if __name__ == "__main__":
    main()
