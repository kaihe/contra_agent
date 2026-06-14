"""Generate multi-state training anchors from winning Contra traces.

For each winning trace, replay it through the emulator, sample N anchors evenly
along the trajectory (the first is always the x=0 start), drop the last one
(it lands in the boss / level-transition scene), and write each anchor as a
gzipped emulator savestate named:

    level1_<timestamp>_x<xscroll:04d>.state

Usage:
    python ppo/make_states.py --dry-run            # preview sampled x positions
    python ppo/make_states.py                      # write anchors for DEFAULT_TRACES
    python ppo/make_states.py --traces a.npz b.npz # custom trace list
"""
import argparse
import glob
import gzip
import os
import sys

import numpy as np

import contra  # noqa: F401  registers the custom ROM integration
import stable_retro as retro

sys.path.insert(0, os.path.dirname(__file__))
from contra_wrapper import xscroll  # noqa: E402

GAME = "Contra-Nes"
SKIP = 3                       # traces were recorded at 60/3 = 20 fps
TRACE_DIR = "synthetic/mc_trace"
OUT_DIR = "ppo/states"

# 5 winning traces spread across recording sessions, for varied start states.
DEFAULT_TRACES = [
    "win_level1_202603301145.npz",
    "win_level1_202603301703.npz",
    "win_level1_202604021858.npz",
    "win_level1_202604081140.npz",
    "win_level1_202604081539.npz",
]


def replay_snapshots(trace_path):
    """Replay a trace, returning (states, xs): the emulator savestate and the
    xscroll value at every step, plus the final post-trace state."""
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

    states, xs = [], []
    for act in actions:
        states.append(env.em.get_state())              # state *before* this action
        xs.append(xscroll(env.unwrapped.get_ram()))
        a = np.asarray(act, dtype=np.uint8)
        for _ in range(SKIP):
            env.step(a.copy())
    states.append(env.em.get_state())                  # final (post-levelup) state
    xs.append(xscroll(env.unwrapped.get_ram()))
    env.close()
    return states, xs


def trace_timestamp(trace_path):
    base = os.path.basename(trace_path)
    return base.replace("win_level1_", "").replace(".npz", "")


def gen_anchors(trace_path, n_anchors, drop_last, out_dir, dry_run):
    states, xs = replay_snapshots(trace_path)
    ts = trace_timestamp(trace_path)
    # The level-1 run ends when the player reaches the boss wall (xscroll peaks
    # at its max ~3072); after the levelup xscroll resets to 0. Sample over the
    # approach [0 .. boss-arrival] so anchors span the level, not the boss fight
    # or the post-transition garbage frames.
    boss_step = int(np.argmax(xs))
    idx = np.linspace(0, boss_step, n_anchors).astype(int)
    dropped = None
    if drop_last:                       # the last sample sits at the boss wall
        dropped = idx[-1]
        idx = idx[:-1]

    saved = []
    for j in idx:
        name = f"level1_{ts}_x{int(xs[j]):04d}.state"
        path = os.path.join(out_dir, name)
        if not dry_run:
            with gzip.open(path, "wb") as f:
                f.write(states[j])
        saved.append((int(j), int(xs[j]), name))

    drop_info = f"  (dropped boss step {dropped}, x={int(xs[dropped])})" if dropped is not None else ""
    print(f"\n{os.path.basename(trace_path)}  ({len(states)} steps, boss@{boss_step}){drop_info}")
    for step, x, name in saved:
        print(f"  step {step:4d}  x{x:04d}  ->  {name}")
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", nargs="+", default=None,
                   help="Trace npz paths or basenames (default: DEFAULT_TRACES)")
    p.add_argument("--n-anchors", type=int, default=10,
                   help="Anchors sampled per trace before dropping the last")
    p.add_argument("--keep-last", action="store_true",
                   help="Keep the final (boss/transition) anchor")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Print sampled x positions without writing files")
    args = p.parse_args()

    traces = args.traces or DEFAULT_TRACES
    traces = [t if os.path.sep in t else os.path.join(TRACE_DIR, t) for t in traces]
    for t in traces:
        if not os.path.isfile(t):
            raise FileNotFoundError(t)

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for t in traces:
        total += len(gen_anchors(t, args.n_anchors, not args.keep_last,
                                 args.out_dir, args.dry_run))
    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {total} anchors from {len(traces)} traces to {args.out_dir}/")


if __name__ == "__main__":
    main()
