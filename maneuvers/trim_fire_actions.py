"""
trim_fire_actions.py — identify pruneable fire actions in a Contra trace.

For each fire action (B button pressed) at step t, look ahead up to WINDOW
steps. If no EV_ENEMY_HIT fires in [t, t+WINDOW), the fire action had no
observable effect on enemy HP and can safely be turned into a no-op.

Output
------
  - Total fire actions in the trace
  - How many could be trimmed (no enemy hit in forward window)
  - Optionally save a trimmed .npz with those fire buttons zeroed out

Usage
-----
    python -m maneuvers.trim_fire_actions path/to/trace.npz
    python -m maneuvers.trim_fire_actions path/to/trace.npz --window 128 --save
"""

import argparse
import warnings

import numpy as np
import stable_retro as retro

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.replay import rewind_state, step_env, GAME
from contra.events import (
    ADDR_ENEMY_TYPE, ADDR_ENEMY_HP, ADDR_ENEMY_HP_COUNT,
    ENEMY_TYPE_FALLING_ROCK,
)

WINDOW = 128   # look-ahead window in steps

_NOOP = np.zeros(9, dtype=np.uint8)


def _is_fire(act: np.ndarray) -> bool:
    """B button (index 0) is the fire button."""
    return bool(np.asarray(act, dtype=np.uint8)[0])


def _enemy_hit(pre_ram: np.ndarray, curr_ram: np.ndarray) -> bool:
    """Return True if any non-trivial enemy took an HP hit this step."""
    for slot in range(ADDR_ENEMY_HP_COUNT):
        etype   = int(pre_ram[ADDR_ENEMY_TYPE + slot])
        pre_hp  = int(pre_ram[ADDR_ENEMY_HP   + slot])
        curr_hp = int(curr_ram[ADDR_ENEMY_HP  + slot])
        if etype == ENEMY_TYPE_FALLING_ROCK:
            continue
        if pre_hp >= 0xF0 or curr_hp >= 0xF0:
            continue
        if pre_hp != curr_hp:
            return True
    return False


def collect_trace(env, actions: np.ndarray, initial_emu_state: bytes):
    """Full replay; returns per-step RAM snapshots."""
    rewind_state(env, initial_emu_state)
    pre_rams  = []
    curr_rams = []

    for act in actions:
        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, np.asarray(act, dtype=np.uint8))
        curr_ram = env.unwrapped.get_ram().copy()
        pre_rams.append(pre_ram)
        curr_rams.append(curr_ram)

    return pre_rams, curr_rams


def analyze_fire_trimmability(trace_path: str, window: int = WINDOW,
                               verbose: bool = True) -> dict:
    ckpt              = np.load(trace_path, allow_pickle=True)
    actions           = ckpt["actions"]
    initial_emu_state = bytes(ckpt["initial_state"])
    n                 = len(actions)

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()

    if verbose:
        print(f"Replaying {n} steps ...")
    pre_rams, curr_rams = collect_trace(env, actions, initial_emu_state)
    env.close()

    # Pre-compute which steps have an enemy hit.
    hit_mask = np.array(
        [_enemy_hit(pre_rams[t], curr_rams[t]) for t in range(n)],
        dtype=bool,
    )

    # For each fire action, check the [t, t+window) window.
    fire_steps      = []
    trimmable_steps = []

    for t in range(n):
        if not _is_fire(actions[t]):
            continue
        fire_steps.append(t)
        t_end = min(t + window, n)
        if not hit_mask[t:t_end].any():
            trimmable_steps.append(t)

    n_fire     = len(fire_steps)
    n_trim     = len(trimmable_steps)
    pct        = 100.0 * n_trim / n_fire if n_fire else 0.0

    if verbose:
        print(f"\nWindow = {window} steps")
        print(f"  Fire actions total : {n_fire}")
        print(f"  Trimmable (no hit) : {n_trim}  ({pct:.1f}%)")
        print(f"  Kept (caused a hit): {n_fire - n_trim}  ({100-pct:.1f}%)")

    return {
        "n_steps":          n,
        "n_fire":           n_fire,
        "n_trimmable":      n_trim,
        "n_kept":           n_fire - n_trim,
        "pct_trimmable":    pct,
        "trimmable_steps":  trimmable_steps,
        "fire_steps":       fire_steps,
        "actions":          actions,
        "initial_state":    initial_emu_state,
    }


def save_trimmed(result: dict, out_path: str):
    """Save a copy of the trace with trimmable fire buttons zeroed out."""
    trimmed = result["actions"].copy()
    for t in result["trimmable_steps"]:
        trimmed[t, 0] = 0   # zero B button; keep Jump etc.
    np.savez_compressed(
        out_path,
        actions=trimmed,
        initial_state=np.frombuffer(result["initial_state"], dtype=np.uint8),
    )
    print(f"Trimmed trace saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?",
                        default="synthetic/mc_trace/win_level1_202603301145.npz",
                        help="Path to .npz trace file")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Look-ahead window in steps (default {WINDOW})")
    parser.add_argument("--save", action="store_true",
                        help="Save trimmed trace alongside the input file")
    args = parser.parse_args()

    result = analyze_fire_trimmability(args.trace, window=args.window, verbose=True)

    if args.save:
        base = args.trace.replace(".npz", "")
        out  = f"{base}_trimmed.npz"
        save_trimmed(result, out)


if __name__ == "__main__":
    main()
