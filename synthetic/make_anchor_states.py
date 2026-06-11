"""
make_anchor_states.py — turn a winning game trace into PPO anchor savestates.

Replays a recorded trace (npz with `actions` (N,9) + `initial_state`) through the
emulator and, at `num` uniformly-spaced points along the in-level portion of the
episode, writes a gzipped `.state` file compatible with RandomStateWrapper.

Usage:
    python synthetic/make_anchor_states.py \
        --npz synthetic/mc_trace/win_level1_202603301145.npz \
        --num 10 --out ppo/states
"""

import argparse
import gzip
import os
import warnings

warnings.filterwarnings("ignore", message=".*Gym.*")

import contra  # noqa: F401  registers custom ROM integration
import numpy as np
import stable_retro as retro

from contra.events import is_gameplay
from contra.replay import GAME, SKIP, rewind_state

ADDR_XSCROLL = 100
ADDR_LEVEL = 48


def _xscroll(ram: np.ndarray) -> int:
    return (int(ram[ADDR_XSCROLL]) << 8) | int(ram[ADDR_XSCROLL + 1])


def main():
    parser = argparse.ArgumentParser(description="Make anchor states from a game trace")
    parser.add_argument("--npz", required=True, help="Trace .npz (actions + initial_state)")
    parser.add_argument("--num", type=int, default=10, help="Number of anchor states")
    parser.add_argument("--out", type=str, default="ppo/states", help="Output directory")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filename tag (default: npz stem)")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    actions = data["actions"]
    initial_state = bytes(data["initial_state"])
    tag = args.tag or os.path.splitext(os.path.basename(args.npz))[0]
    os.makedirs(args.out, exist_ok=True)

    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE, render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial_state)

    # Single replay pass: snapshot (level, xscroll, raw emu state) at every
    # agent step so we can pick anchors after seeing where the level ends.
    start_level = int(env.unwrapped.get_ram()[ADDR_LEVEL])
    snapshots = []  # (step, xscroll, raw_state)
    for step, act in enumerate(actions):
        act_arr = np.asarray(act, dtype=np.uint8)
        for _ in range(SKIP):
            env.step(act_arr.copy())
        ram = env.unwrapped.get_ram()
        if int(ram[ADDR_LEVEL]) != start_level:
            break  # reached the next level — stop collecting in-level anchors
        if not is_gameplay(ram):
            continue  # skip title/post-boss transition frames (not active play)
        snapshots.append((step, _xscroll(ram), env.em.get_state()))
    env.close()

    if len(snapshots) < args.num:
        raise RuntimeError(f"Only {len(snapshots)} in-level steps; need >= {args.num}")

    # Uniformly-spaced indices across the in-level trajectory (start -> boss).
    picks = np.linspace(0, len(snapshots) - 1, args.num).round().astype(int)
    picks = sorted(set(picks.tolist()))

    print(f"Trace: {args.npz}")
    print(f"  in-level steps: {len(snapshots)} (level {start_level}, "
          f"x {snapshots[0][1]} -> {snapshots[-1][1]})")
    print(f"  writing {len(picks)} anchors to {args.out}/\n")
    print(f"{'#':>3}  {'step':>5}  {'xscroll':>7}  file")

    written = []
    for i, idx in enumerate(picks):
        step, x, raw = snapshots[idx]
        fname = f"{tag}_s{step:04d}_x{x:04d}.state"
        path = os.path.join(args.out, fname)
        with gzip.open(path, "wb") as f:
            f.write(raw)
        written.append(path)
        print(f"{i:>3}  {step:>5}  {x:>7}  {fname}")

    # Validate: reload each written state and confirm xscroll round-trips.
    verify = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE, render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    verify.reset()
    bad = []
    for path, idx in zip(written, picks):
        with gzip.open(path, "rb") as f:
            rewind_state(verify, f.read())
        if _xscroll(verify.unwrapped.get_ram()) != snapshots[idx][1]:
            bad.append(path)
    verify.close()

    if bad:
        print(f"\nWARNING: xscroll mismatch on reload for: {bad}")
    else:
        print(f"\nOK: all {len(written)} states reload with matching xscroll.")


if __name__ == "__main__":
    main()
