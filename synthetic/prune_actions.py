"""
prune_actions.py — Remove ineffective actions from a Contra trace.

Algorithm (from annotate/instance.py)
--------------------------------------
For each step i:
  1. Save the current emulator state.
  2. Apply the original action for SKIP frames → ram_orig, next_orig.
  3. Rewind and apply NOOP for SKIP frames → ram_noop.
  4. If ram_orig == ram_noop: action had zero effect → commit NOOP.
     Otherwise: commit original action, rewind to next_orig.

RAM equality is a perfect test for the deterministic NES: identical RAM
means identical future behaviour, so no lookahead window is needed.

Usage
-----
    python synthetic/prune_actions.py
"""

import argparse
import os
import warnings
from collections import Counter

import numpy as np
import stable_retro as retro

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.replay import rewind_state, step_env, replay_actions, GAME, SKIP
from contra.inputs import DPAD_TABLE, BUTTON_TABLE, DPAD_NAMES, BUTTON_NAMES

_NOOP = np.zeros(9, dtype=np.uint8)

# RAM bytes that mirror the controller input directly (not gameplay state).
# Comparing these would always show a difference when fire/jump bits differ,
# even when the gameplay outcome is identical.
#   $f1/$f2 = CONTROLLER_STATE (P1/P2 currently-pressed buttons)
#   $f5/$f6 = CONTROLLER_STATE_DIFF (delta between reads)
#   $f9/$fa = CTRL_KNOWN_GOOD (last valid read)
_INPUT_RAM_INDICES = np.array([0xf1, 0xf2, 0xf5, 0xf6, 0xf9, 0xfa], dtype=np.intp)


def _ram_eq(a, b):
    """Compare two RAM snapshots, ignoring controller-input bytes."""
    if np.array_equal(a, b):
        return True
    diff = np.where(a != b)[0]
    return np.all(np.isin(diff, _INPUT_RAM_INDICES))


# ── Action decoder ─────────────────────────────────────────────────────────────

def _action_str(act):
    """Decode a 9-element action array into a 'DPAD+BUTTON' label."""
    a = [int(v) for v in act]

    dpad_name = "_"
    for i, row in enumerate(DPAD_TABLE):
        if a[4] == row[4] and a[5] == row[5] and a[6] == row[6] and a[7] == row[7]:
            dpad_name = DPAD_NAMES[i]
            break

    button_name = "_"
    for i, row in enumerate(BUTTON_TABLE):
        if a[0] == row[0] and a[8] == row[8]:
            button_name = BUTTON_NAMES[i]
            break

    parts = [p for p in (dpad_name, button_name) if p != "_"]
    return "+".join(parts) if parts else "NOOP"


# ── Pruning ────────────────────────────────────────────────────────────────────

def prune_actions(actions, initial_emu_state, verbose=True):
    """
    Prune ineffective fire/jump bits from each action using RAM comparison.

    For each action that has fire (bit 0) or jump (bit 8) set, test removing
    each independently: if the RAM outcome is unchanged, that bit is pruned.
    D-pad bits are never touched.

    Returns
    -------
    pruned : np.ndarray (N, 9) uint8  — actions with ineffective fire/jump bits cleared
    """
    n = len(actions)
    pruned = np.array(actions, dtype=np.uint8).copy()

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial_emu_state)

    pruned_fire = 0
    pruned_jump = 0

    for i in range(n):
        act_arr   = np.asarray(actions[i], dtype=np.uint8)
        has_fire  = bool(act_arr[0])
        has_jump  = bool(act_arr[8])

        if not has_fire and not has_jump:
            step_env(env, act_arr)
            continue

        cur_state = env.em.get_state()

        # Apply original action → reference RAM
        step_env(env, act_arr)
        ram_orig  = env.unwrapped.get_ram().copy()
        next_orig = env.em.get_state()

        candidate = act_arr.copy()

        # Try removing fire
        if has_fire:
            no_fire = act_arr.copy()
            no_fire[0] = 0
            rewind_state(env, cur_state)
            step_env(env, no_fire)
            if _ram_eq(ram_orig, env.unwrapped.get_ram()):
                candidate[0] = 0
                pruned_fire += 1

        # Try removing jump
        if has_jump:
            no_jump = act_arr.copy()
            no_jump[8] = 0
            rewind_state(env, cur_state)
            step_env(env, no_jump)
            if _ram_eq(ram_orig, env.unwrapped.get_ram()):
                candidate[8] = 0
                pruned_jump += 1

        pruned[i] = candidate
        rewind_state(env, next_orig)

    env.close()

    if verbose:
        fire_total = sum(1 for a in actions if a[0])
        jump_total = sum(1 for a in actions if a[8])
        print(f"  Total steps       : {n}")
        print(f"  Steps with fire   : {fire_total}  →  pruned {pruned_fire}")
        print(f"  Steps with jump   : {jump_total}  →  pruned {pruned_jump}")

    return pruned


# ── Histogram ──────────────────────────────────────────────────────────────────

def show_action_histogram(actions, pruned, verbose=True):
    """Print and save a bar chart of which action labels had fire/jump pruned."""
    import matplotlib.pyplot as plt

    counts = Counter(
        _action_str(actions[i])
        for i in range(len(actions))
        if not np.array_equal(pruned[i], actions[i])
    )
    if not counts:
        print("  No actions pruned.")
        return

    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))

    if verbose:
        print("\n  Pruned action breakdown:")
        for label, cnt in zip(labels, values):
            print(f"    {label:<12}  {cnt}")

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
    ax.bar(labels, values)
    ax.set_xlabel("Action")
    ax.set_ylabel("Count pruned")
    ax.set_title("Pruned actions histogram")
    plt.tight_layout()
    os.makedirs("tmp", exist_ok=True)
    out = "tmp/pruned_histogram.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved → {out}")


# ── Verification ───────────────────────────────────────────────────────────────

def verify_level_up(actions, initial_emu_state, label="pruned", verbose=True):
    """Replay *actions* and check whether they produce a level-up or game-clear.

    Returns True if the sequence clears the level, False otherwise.
    """
    result = replay_actions(
        actions,
        initial_state=initial_emu_state,
        want_video=False,
        verbose=verbose,
    )
    outcome = result["result"]
    passed = outcome in ("level_up", "game_clear")
    print(f"\n  [{label}] outcome: {outcome}  →  {'PASS' if passed else 'FAIL'}")
    return passed


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prune ineffective actions from a Contra trace")
    parser.add_argument("--output", default=None, help="Output NPZ path")
    args = parser.parse_args()

    file_path = 'synthetic/mc_trace/win_level8_202604171330.npz'
    ckpt = np.load(file_path, allow_pickle=True)
    actions = ckpt["actions"]
    initial_emu_state = bytes(ckpt["initial_state"])

    print(f"Loaded {len(actions)} actions from {file_path}\n")

    pruned = prune_actions(actions, initial_emu_state, verbose=True)
    show_action_histogram(actions, pruned, verbose=True)

    print("\nVerifying pruned sequence…")
    verify_level_up(pruned, initial_emu_state, label="pruned")




if __name__ == "__main__":
    main()
