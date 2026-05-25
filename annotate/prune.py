"""Action pruning: remove redundant button presses that do not affect game RAM."""

import numpy as np
import stable_retro as retro

from contra.replay import rewind_state, step_env, GAME

# RAM bytes that mirror the controller input directly (not gameplay state).
#   $f1/$f2 = CONTROLLER_STATE (P1/P2 currently-pressed buttons)
#   $f5/$f6 = CONTROLLER_STATE_DIFF (delta between reads)
#   $f9/$fa = CTRL_KNOWN_GOOD (last valid read)
_INPUT_RAM_INDICES = np.array([0xf1, 0xf2, 0xf5, 0xf6, 0xf9, 0xfa], dtype=np.intp)

# The 6 independently-testable input bits.
_CRITICAL_BITS = [(0, "fire"), (8, "jump"), (4, "up"), (5, "down"), (6, "left"), (7, "right")]

_NOOP = np.zeros(9, dtype=np.uint8)


def _ram_eq(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a and b differ only in controller-mirror bytes."""
    if np.array_equal(a, b):
        return True
    diff = np.where(a != b)[0]
    return bool(np.all(np.isin(diff, _INPUT_RAM_INDICES)))


def prune_actions(
    actions: np.ndarray,
    initial_emu_state: bytes,
    verbose: bool = True,
    env=None,
) -> np.ndarray:
    """Zero out each of the 6 critical input bits when they leave RAM unchanged.

    Each bit (fire, jump, up, down, left, right) is tested independently
    against the original action's RAM result.  Controller-mirror bytes are
    excluded from the comparison so input-register noise never blocks pruning.

    To avoid creating fake 'Just Pressed' events (0->1 transitions) that diverge
    the PRNG and game state later, we process the sequence backwards and only
    allow pruning a bit if the NEXT frame also has that bit as 0.

    Parameters
    ----------
    actions : np.ndarray (N, 9) uint8
        NES MultiBinary action arrays.
    initial_emu_state : bytes
        Emulator save-state to rewind to before replaying.
    verbose : bool
        Print pruning statistics.
    env : retro.Env or None
        Optional pre-made environment.  If None, a new one is created and closed.

    Returns
    -------
    pruned : np.ndarray (N, 9) uint8
    """
    n = len(actions)
    pruned = np.array(actions, dtype=np.uint8).copy()

    _own_env = env is None
    if _own_env:
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
                if i == n - 1 or pruned[i + 1][idx] == 0:
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

    if _own_env:
        env.close()

    if verbose:
        print(f"    prune: {arrays_pruned}/{n} action arrays modified")

    return pruned
