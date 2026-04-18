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
import warnings

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

def _dpad_name(act):
    """Return the d-pad label for an action (e.g. 'R', 'UL', or '_')."""
    a = [int(v) for v in act]
    for i, row in enumerate(DPAD_TABLE):
        if a[4] == row[4] and a[5] == row[5] and a[6] == row[6] and a[7] == row[7]:
            return DPAD_NAMES[i]
    return "_"


# ── Pruning ────────────────────────────────────────────────────────────────────

# The 6 independently-testable input bits and their human-readable names.
# NES layout: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
_CRITICAL_BITS = [
    (0, "fire"),
    (8, "jump"),
    (4, "up"),
    (5, "down"),
    (6, "left"),
    (7, "right"),
]

def prune_actions(actions: np.ndarray, initial_emu_state: bytes, verbose: bool = True) -> np.ndarray:
    """Zero out each of the 6 critical input bits when they leave RAM unchanged.

    Each bit (fire, jump, up, down, left, right) is tested independently
    against the original action's RAM result.  Controller-mirror bytes are
    excluded from the comparison so input-register noise never blocks pruning.

    To avoid creating fake 'Just Pressed' events (0->1 transitions) that diverge
    the PRNG and game state later, we process the sequence backwards and only
    allow pruning a bit if the NEXT frame also has that bit as 0.

    Returns
    -------
    pruned : np.ndarray (N, 9) uint8
    """
    import stable_retro as retro

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

    # 1. Forward pass to save all true states and true RAMs
    true_states = []
    true_rams = []
    for i in range(n):
        true_states.append(env.em.get_state())
        step_env(env, actions[i])
        true_rams.append(env.unwrapped.get_ram().copy())

    # 2. Backward pass to prune safely
    arrays_pruned = 0

    for i in range(n - 1, -1, -1):
        act_orig = actions[i]
        
        # Only consider bits that are 1 and where the NEXT frame (if any) is 0
        active_bits = []
        for idx, name in _CRITICAL_BITS:
            if act_orig[idx] == 1:
                # Safe to prune if it's the last frame OR the next frame also has this bit as 0
                if i == n - 1 or pruned[i+1][idx] == 0:
                    active_bits.append((idx, name))

        if not active_bits:
            continue

        candidate = act_orig.copy()
        cur_state = true_states[i]
        true_ram = true_rams[i]
        
        for idx, name in active_bits:
            probe = candidate.copy()
            probe[idx] = 0
            rewind_state(env, cur_state)
            step_env(env, probe)
            if _ram_eq(true_ram, env.unwrapped.get_ram()):
                candidate = probe
            
        pruned[i] = candidate
        if not np.array_equal(candidate, act_orig):
            arrays_pruned += 1

    env.close()

    if verbose:
        print(f"    prune: {arrays_pruned}/{n} action arrays modified")

    return pruned


# ── Summary table ──────────────────────────────────────────────────────────────

def show_action_histogram(actions, pruned, verbose=True):
    """Print a before/after table covering fire, jump, and all d-pad directions."""
    dpad_labels = [name for name in DPAD_NAMES if name != "_"]

    before = {"fire": 0, "jump": 0, **{d: 0 for d in dpad_labels}}
    after  = {"fire": 0, "jump": 0, **{d: 0 for d in dpad_labels}}

    for orig, prun in zip(actions, pruned):
        if orig[0]: before["fire"] += 1
        if orig[8]: before["jump"] += 1
        d = _dpad_name(orig)
        if d != "_":
            before[d] += 1

        if prun[0]: after["fire"] += 1
        if prun[8]: after["jump"] += 1
        d_after = _dpad_name(prun)
        if d_after != "_":
            after[d_after] += 1

    labels = ["fire", "jump"] + [d for d in dpad_labels if before[d] > 0]

    print(f"\n  {'input':<6}  {'before':>7}  {'after':>7}  {'pruned':>7}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
    for label in labels:
        print(f"  {label:<6}  {before[label]:>7}  {after[label]:>7}  {before[label] - after[label]:>7}")


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

    # file_path = 'synthetic/mc_trace/win_level1_202604091009.npz'
    # file_path = 'synthetic/mc_trace/win_level8_202604171419.npz'
    # file_path = 'synthetic/mc_trace/win_level1_202603301145.npz'
    file_path = 'synthetic/mc_trace/win_level1_202604101354.npz'
    ckpt = np.load(file_path, allow_pickle=True)
    actions = ckpt["actions"]
    initial_emu_state = bytes(ckpt["initial_state"])

    print(f"Loaded {len(actions)} actions from {file_path}\n")

    pruned = prune_actions(actions, initial_emu_state, verbose=True)
    show_action_histogram(actions, pruned, verbose=True)

    verify_level_up(pruned, initial_emu_state, label="pruned", verbose=True)

    



if __name__ == "__main__":
    main()
