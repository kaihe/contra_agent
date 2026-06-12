"""
prune_actions.py — Remove ineffective button presses from a Contra trace.

Algorithm
---------
1. Forward pass: replay the trace once, saving for every step the emulator
   state *before* the action and the RAM snapshot *after* it.
2. Backward pass: for each step, test each pressed bit (fire, jump, up,
   down, left, right) independently — rewind, re-step with that single bit
   zeroed, and drop the bit if RAM comes out identical to the original.

RAM equality is a perfect test on the deterministic NES: identical RAM
means identical future behaviour, so no lookahead window is needed.
RAM bytes that merely mirror the controller input are excluded from the
comparison (see _INPUT_RAM_INDICES).

Why backwards: zeroing a bit at step i while step i+1 still holds it
would turn a held button into a fresh 0->1 "just pressed" event at i+1
(e.g. an extra shot or jump), diverging the game state downstream.  A bit
is therefore only prunable if the already-pruned next step has it at 0.
Processing back-to-front guarantees pruned[i+1] is final when step i is
examined.

Usage
-----
    python synthetic/prune_actions.py [--input TRACE.npz] [--output PRUNED.npz]
"""

import argparse
import warnings

import numpy as np
import stable_retro as retro

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.replay import rewind_state, step_env, replay_actions, GAME, SKIP
from contra.inputs import DPAD_TABLE, DPAD_NAMES

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
    """Zero out each critical input bit wherever it leaves RAM unchanged.

    Parameters
    ----------
    actions : (N, 9) uint8 array of controller states, one per step.
    initial_emu_state : emulator savestate the trace starts from.

    Returns
    -------
    pruned : np.ndarray (N, 9) uint8
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

    # Forward pass: record the ground-truth state before and RAM after each step.
    true_states = []
    true_rams = []
    for i in range(n):
        true_states.append(env.em.get_state())
        step_env(env, actions[i])
        true_rams.append(env.unwrapped.get_ram().copy())

    # Backward pass: try dropping each pressed bit, keep the drop if RAM matches.
    steps_modified = 0

    for i in range(n - 1, -1, -1):
        act_orig = actions[i]

        # A bit is a pruning candidate only if pressed here AND already 0 on
        # the (pruned) next step, so removing it cannot create a fake
        # 0->1 "just pressed" transition there.
        candidate_bits = [
            (idx, name)
            for idx, name in _CRITICAL_BITS
            if act_orig[idx] == 1 and (i == n - 1 or pruned[i + 1][idx] == 0)
        ]
        if not candidate_bits:
            continue

        # Test bits one at a time; successful drops accumulate in `candidate`.
        candidate = act_orig.copy()
        for idx, name in candidate_bits:
            probe = candidate.copy()
            probe[idx] = 0
            rewind_state(env, true_states[i])
            step_env(env, probe)
            if _ram_eq(true_rams[i], env.unwrapped.get_ram()):
                candidate = probe

        pruned[i] = candidate
        if not np.array_equal(candidate, act_orig):
            steps_modified += 1

    env.close()

    if verbose:
        print(f"    prune: {steps_modified}/{n} action arrays modified")

    return pruned


# ── Summary table ──────────────────────────────────────────────────────────────

def show_action_histogram(actions, pruned):
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
    parser.add_argument(
        "--input",
        default="synthetic/mc_trace/win_level1_202603301145.npz",
        help="Input trace NPZ (needs 'actions' and 'initial_state' keys)",
    )
    parser.add_argument("--output", default=None, help="Output NPZ path for the pruned trace")
    args = parser.parse_args()

    ckpt = np.load(args.input, allow_pickle=True)
    actions = ckpt["actions"]
    initial_emu_state = bytes(ckpt["initial_state"])

    print(f"Loaded {len(actions)} actions from {args.input}\n")

    pruned = prune_actions(actions, initial_emu_state, verbose=True)
    passed = verify_level_up(pruned, initial_emu_state, label="pruned", verbose=True)

    if args.output:
        if not passed:
            print(f"\n  verification FAILED — not saving {args.output}")
            return
        data = {key: ckpt[key] for key in ckpt.files}
        data["actions"] = pruned
        np.savez(args.output, **data)
        print(f"\n  saved pruned trace to {args.output}")


if __name__ == "__main__":
    main()
