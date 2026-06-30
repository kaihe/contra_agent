"""Stateful action mask — forbid structurally meaningless presses at sample time.

The old pipeline made traces clean *after the fact* (``prune_actions.py`` replays
and drops any button press that leaves RAM unchanged) or *softly* (a fire/jump
press penalty in ``search_reward.py``). This module instead reads the player's
RAM state plus the previously-committed action and returns, for each action in a
table, whether pressing it could change the game at all. The Monte-Carlo sampler
(``mc_search.rollout``) restricts itself to that legal set, so a meaningless
press is never generated in the first place.

What the disassembly says (reference/nes-contra-us):

* Jump (A button) — ``bank7.asm`` ``@player_jump_check`` → ``set_jump_status_and_y_velocity``.
  A only starts a jump when the player is grounded and idle:
    - ``PLAYER_JUMP_STATUS ($A0) == 0``   (not already jumping)
    - ``EDGE_FALL_CODE     ($A4) == 0``   (not falling off a ledge)
    - ``ELECTROCUTED_TIMER ($C8) == 0``   (input frozen while electrocuted)
    - ``PLAYER_WATER_STATE ($B2) == 0``   (no jumping in water)
  and it is read from CONTROLLER_STATE_DIFF, i.e. only on the 0->1 press edge,
  so a held A (``prev_action[A] == 1``) does nothing.

* Fire (B button) — ``check_player_fire`` (``bank6.asm:290``). Standard / F / S
  (spread) weapons read B from CONTROLLER_STATE_DIFF (``bank6.asm:319``): a held
  B fires nothing, you must release and re-press. M ($01) and L ($04) fire while
  held (``CONTROLLER_STATE``, line 314). Firing is also blocked while hidden or
  electrocuted, and you cannot shoot downward in water.

Because each level's action table always carries the bit-cleared twin of every
fire/jump action (``R``↔``RJ``/``RF``, ``_``↔``J``/``F`` …) and the no-op ``_``
is always legal, masking the inert-bit actions never empties the legal set — the
searcher just uses the canonical twin.

Bit order of an action vector (stable_retro ALL layout, matches prune_actions):
``[B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]`` — B=fire(0), A=jump(8).
"""

import numpy as np

from contra.events import ADDR_WEAPON  # $AA: low nibble = weapon type, bit4 = rapid fire

# ── Action-vector bit indices ────────────────────────────────────────────────
FIRE_BIT = 0   # B
DOWN_BIT = 5
JUMP_BIT = 8   # A

# ── Player-1 RAM addresses (index x=0; ram.asm zero-page) ────────────────────
ADDR_JUMP_STATUS = 0xA0   # PLAYER_JUMP_STATUS: nonzero while jumping
ADDR_EDGE_FALL   = 0xA4   # EDGE_FALL_CODE: nonzero while falling off a ledge
ADDR_WATER_STATE = 0xB2   # PLAYER_WATER_STATE: nonzero while in/exiting water
ADDR_ELECTRO     = 0xC8   # ELECTROCUTED_TIMER: nonzero freezes player input
ADDR_HIDDEN      = 0xBA   # PLAYER_HIDDEN: nonzero when player not visible

# Weapon types that fire on every held frame rather than only on the press edge.
_HELD_FIRE_WEAPONS = frozenset({0x01, 0x04})  # M (machine gun), L (laser)


def jump_is_live(ram: np.ndarray, prev_action: np.ndarray) -> bool:
    """True if pressing A this step could start a jump (else A is inert)."""
    return (
        prev_action[JUMP_BIT] == 0          # fresh press (CONTROLLER_STATE_DIFF)
        and ram[ADDR_JUMP_STATUS] == 0      # not already airborne
        and ram[ADDR_EDGE_FALL] == 0        # not falling off a ledge
        and ram[ADDR_ELECTRO] == 0          # input not frozen
        and ram[ADDR_WATER_STATE] == 0      # cannot jump in water
    )


def fire_gate_open(ram: np.ndarray, prev_action: np.ndarray) -> bool:
    """True if a B-press could fire at all this step, ignoring the d-pad.

    Shut while hidden/electrocuted, or when an edge-triggered weapon already had
    B held last step (held B does not re-fire). Does not model bullet-slot
    exhaustion (``create_bullet_max_04``); a valid edge-press into a full slot
    set still counts as open and is left to the reward press-penalty.
    """
    if ram[ADDR_ELECTRO] != 0 or ram[ADDR_HIDDEN] != 0:
        return False
    weapon = int(ram[ADDR_WEAPON]) & 0x0F
    if weapon not in _HELD_FIRE_WEAPONS and prev_action[FIRE_BIT] == 1:
        return False                        # held B does not re-fire edge weapons
    return True


def legal_mask(actions: np.ndarray, ram: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    """Boolean mask over `actions`: drop any action that presses an inert A/B bit.

    Parameters
    ----------
    actions     : (N, 9) uint8 action table.
    ram         : current RAM snapshot (before this step's action).
    prev_action : (9,) the previously committed/sampled action (for edge detection).

    Returns
    -------
    (N,) bool — always has at least the no-op action legal.
    """
    fire_bits = actions[:, FIRE_BIT] == 1
    mask = np.ones(len(actions), dtype=bool)
    if not jump_is_live(ram, prev_action):
        mask &= actions[:, JUMP_BIT] == 0
    if not fire_gate_open(ram, prev_action):
        mask &= ~fire_bits
    elif ram[ADDR_WATER_STATE] != 0:
        # In water B fires except while aiming down — mask only the down+fire rows.
        mask &= ~(fire_bits & (actions[:, DOWN_BIT] == 1))
    return mask
