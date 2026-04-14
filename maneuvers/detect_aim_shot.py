"""
detect_aim_shot.py — detect AIM_SHOT maneuvers in a winning Contra trace.

Algorithm
---------
For each per-slot enemy HP decrement (EV_ENEMY_HIT) at step t_hit:
  1. Look back up to FIRE_LOOKBACK steps for the most recent fire action at t_fire.
  2. Counterfactual: rewind to emu state before t_fire, apply no-op instead,
     replay steps t_fire+1..t_hit with original actions.
  3. If the HP delta for that slot drops to 0 in the counterfactual → causal
     fire confirmed → record an AIM_SHOT maneuver.

Output
------
JSON list of maneuver dicts, each with:
  tag, t_fire, t_hit, slot, enemy_type, hp_delta,
  enemy_x, enemy_y, player_x, player_y, weapon, desc

Usage
-----
    python -m maneuvers.detect_aim_shot path/to/trace.npz
    python -m maneuvers.detect_aim_shot path/to/trace.npz --out results.json
"""

import argparse
import json
import os
import warnings

import numpy as np
import stable_retro as retro

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.replay import rewind_state, step_env, GAME, replay_actions
from contra.events import (
    ADDR_ENEMY_TYPE, ADDR_ENEMY_HP, ADDR_ENEMY_HP_COUNT,
    ENEMY_TYPE_FALLING_ROCK, ADDR_WEAPON, ADDR_LEVEL,
    WEAPON_NAMES, enemy_type_name,
)

# ── Additional RAM addresses (from reference/nes-contra-us/src/ram.asm) ──────
ADDR_ENEMY_X_POS = 0x033e   # ENEMY_X_POS: 16-slot enemy screen X positions
ADDR_ENEMY_Y_POS = 0x0324   # ENEMY_Y_POS: 16-slot enemy screen Y positions
ADDR_PLAYER_X    = 0x0334   # SPRITE_X_POS[0]: player 1 screen X
ADDR_PLAYER_Y    = 0x031a   # SPRITE_Y_POS[0]: player 1 screen Y

# Lookback window per weapon type (logical steps at SKIP=3 NES frames/step).
#
# Source: bank6.asm bullet_velocity_* tables (all speeds in px/NES frame).
#   Regular (no rapid fire) : bullet_velocity_normal  → 3 px/f
#     worst case: 256 px / 3 px/f / 3 f/step = 28 steps → use 30
#   MachineGun (rapid fire) : bullet_velocity_rapid   → 4 px/f
#     256 / 4 / 3 = 21 steps → use 22
#   Flamethrower (rapid)    : bullet_velocity_f_rapid → 2 px/f
#     256 / 2 / 3 = 42 steps; but flame has short range → cap at 20
#   Spread                  : s_bullet_*_vel tables   → ~3 px/f (same as Regular)
#     256 / 3 / 3 = 28 steps → use 30
#   Laser                   : bullet_velocity_rapid   → 4 px/f (same table as M)
#     256 / 4 / 3 = 21 steps → use 22
FIRE_LOOKBACK_BY_WEAPON = {
    0: 30,   # Regular      — 3 px/NES frame, up to 28 logical steps
    1: 22,   # MachineGun   — 4 px/NES frame (rapid fire table), up to 21 steps
    2: 20,   # Flamethrower — 2 px/NES frame but short range
    3: 30,   # Spread       — ~3 px/NES frame, same speed as Regular
    4: 22,   # Laser        — 4 px/NES frame (uses bullet_velocity_rapid table)
}
FIRE_LOOKBACK_DEFAULT = 30


_TAN22 = 0.41   # tan(22.5°) — octant boundary


def _octant(dx: int, dy: int, labels_h: tuple, labels_v: tuple, labels_d: tuple) -> str:
    """Shared 8-direction classifier.  dy is already flipped so positive = up."""
    if dx == 0 and dy == 0:
        return "nearby"
    adx, ady = abs(dx), abs(dy)
    if ady < adx * _TAN22:
        return labels_h[0] if dx > 0 else labels_h[1]
    elif adx < ady * _TAN22:
        return labels_v[0] if dy > 0 else labels_v[1]
    else:
        vi = 0 if dy > 0 else 1
        hi = 0 if dx > 0 else 1
        return labels_d[vi * 2 + hi]


# Natural-language phrase for where the enemy is relative to the player.
_ENEMY_POSITION_PHRASE = {
    "right":       "to the right",
    "left":        "to the left",
    "up":          "above",
    "down":        "below",
    "up-right":    "to the upper right",
    "up-left":     "to the upper left",
    "down-right":  "to the lower right",
    "down-left":   "to the lower left",
    "nearby":      "nearby",
}

# Natural-language weapon label.
_WEAPON_PHRASE = {
    "Regular":      "regular gun",
    "MachineGun":   "machine gun",
    "Flamethrower": "flamethrower",
    "Spread":       "spread gun",
    "Laser":        "laser",
}


def _enemy_position(player_x: int, player_y: int, enemy_x: int, enemy_y: int) -> str:
    """Where the enemy is relative to the player, as a natural phrase."""
    dx =  enemy_x - player_x
    dy = -(enemy_y - player_y)      # flip: positive = up
    key = _octant(dx, dy,
                  ("right", "left"),
                  ("up", "down"),
                  ("up-right", "up-left", "down-right", "down-left"))
    return _ENEMY_POSITION_PHRASE[key]


# Max gap (in steps) between successive hits on the same slot before the run
# is considered a new event. MachineGun hits every ~2 steps, so 5 is generous.
HIT_GAP_TOLERANCE = 5   # steps; max silent gap within a same-slot run

_NOOP = np.zeros(9, dtype=np.uint8)


def _is_fire(act: np.ndarray) -> bool:
    """B button (index 0) is the fire button."""
    return bool(np.asarray(act, dtype=np.uint8)[0])


def _weapon_type(ram: np.ndarray) -> int:
    return int(ram[ADDR_WEAPON]) & 0x0F


# ── Pass 1: full replay, collecting per-step RAM snapshots + emu states ───────

def collect_trace(env, actions: np.ndarray, initial_emu_state: bytes):
    """
    Step through the entire action sequence once.

    Returns three parallel lists (one entry per step):
      pre_rams   : RAM snapshot before the step
      curr_rams  : RAM snapshot after the step
      emu_states : emulator state bytes BEFORE the step (for counterfactual rewinds)
    """
    rewind_state(env, initial_emu_state)
    pre_rams   = []
    curr_rams  = []
    emu_states = []

    for act in actions:
        emu_states.append(env.em.get_state())
        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, np.asarray(act, dtype=np.uint8))
        curr_ram = env.unwrapped.get_ram().copy()
        pre_rams.append(pre_ram)
        curr_rams.append(curr_ram)

    return pre_rams, curr_rams, emu_states


# ── Per-slot HP delta ─────────────────────────────────────────────────────────

ENEMY_TYPE_BULLET = 0x01  # projectile spawned by other enemies; no real HP

def _slot_hp_delta(pre_ram: np.ndarray, curr_ram: np.ndarray, slot: int) -> int:
    """
    Return HP decrease for one enemy slot in one step.
    Returns 0 if:
      - slot is inactive (HP sentinel >= 0xf0)
      - enemy is a Bullet (type 0x01) — projectile object, HP field is meaningless
      - enemy is a Falling Rock (type 0x13 on L3) — respawns endlessly, no HP pool
    """
    etype   = int(pre_ram[ADDR_ENEMY_TYPE + slot])
    pre_hp  = int(pre_ram[ADDR_ENEMY_HP   + slot])
    curr_hp = int(curr_ram[ADDR_ENEMY_HP  + slot])

    if etype == ENEMY_TYPE_BULLET:
        return 0
    if etype == ENEMY_TYPE_FALLING_ROCK and int(pre_ram[ADDR_LEVEL]) == 2:
        return 0
    if pre_hp >= 0xf0 or curr_hp >= 0xf0:
        return 0
    return max(0, pre_hp - curr_hp)




# ── Main detector ─────────────────────────────────────────────────────────────

def collect_hp_loss_events(trace_path: str) -> list[dict]:
    """Pass 1 only: replay the trace and return every per-slot enemy HP-loss event."""
    ckpt = np.load(trace_path, allow_pickle=True)
    actions           = ckpt["actions"]
    initial_emu_state = bytes(ckpt["initial_state"])

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()

    print(f"Collecting {len(actions)}-step trace from {trace_path} …")
    pre_rams, curr_rams, _ = collect_trace(env, actions, initial_emu_state)
    env.close()

    # open_run[slot] = in-progress merged event; "last_hit" tracks gap size
    open_run: dict[int, dict] = {}
    events: list[dict] = []

    for t_hit in range(len(actions)):
        pre_ram  = pre_rams[t_hit]
        curr_ram = curr_rams[t_hit]
        level_0  = int(pre_ram[ADDR_LEVEL])
        weapon   = _weapon_type(pre_ram)

        # Close runs whose last hit is older than the gap tolerance
        for slot in list(open_run):
            if t_hit - open_run[slot]["last_hit"] > HIT_GAP_TOLERANCE:
                run = open_run.pop(slot)
                del run["last_hit"]
                events.append(run)

        for slot in range(ADDR_ENEMY_HP_COUNT):
            hp_delta = _slot_hp_delta(pre_ram, curr_ram, slot)
            if hp_delta == 0:
                continue

            etype    = int(pre_ram[ADDR_ENEMY_TYPE + slot])
            enemy_x  = int(pre_ram[ADDR_ENEMY_X_POS + slot])
            enemy_y  = int(pre_ram[ADDR_ENEMY_Y_POS + slot])
            player_x = int(pre_ram[ADDR_PLAYER_X])
            player_y = int(pre_ram[ADDR_PLAYER_Y])

            hp_after = int(curr_ram[ADDR_ENEMY_HP + slot])

            if slot in open_run:
                open_run[slot]["t_end"]    = t_hit
                open_run[slot]["last_hit"] = t_hit
                open_run[slot]["hp_delta"] += hp_delta
                open_run[slot]["hp_after"] = hp_after
            else:
                open_run[slot] = dict(
                    t_start    = t_hit,
                    t_end      = t_hit,
                    last_hit   = t_hit,
                    slot       = slot,
                    level      = level_0 + 1,
                    enemy_type = f"0x{etype:02x}",
                    enemy_name = enemy_type_name(etype, level_0),
                    hp_delta   = hp_delta,
                    hp_after   = hp_after,
                    enemy_x    = enemy_x,
                    enemy_y    = enemy_y,
                    player_x   = player_x,
                    player_y   = player_y,
                    weapon     = WEAPON_NAMES.get(weapon, f"unknown({weapon})"),
                )

    # flush remaining open runs
    for run in open_run.values():
        del run["last_hit"]
        events.append(run)
    events.sort(key=lambda e: e["t_start"])

    # ── Annotate each event with fire actions and wrap into maneuver dicts ───
    maneuvers = []
    for e in events:
        weapon_id = next(
            (k for k, v in WEAPON_NAMES.items() if v == e["weapon"]), None
        )
        lookback      = FIRE_LOOKBACK_BY_WEAPON.get(weapon_id, FIRE_LOOKBACK_DEFAULT)
        t_search_from = max(0, e["t_start"] - lookback)

        fire_steps = [
            t for t in range(t_search_from, e["t_start"])
            if _is_fire(actions[t])
        ]
        t_aim      = fire_steps[0] if fire_steps else None  # earliest fire in window
        destroyed  = e["hp_after"] == 0 or e["hp_after"] >= 0xf0
        name       = e["enemy_name"].lower()
        pos_phrase = _enemy_position(e["player_x"], e["player_y"], e["enemy_x"], e["enemy_y"])
        weapon     = _WEAPON_PHRASE.get(e["weapon"], e["weapon"].lower())

        # Starting HP = damage dealt + HP remaining after last hit.
        # > 1 means the enemy required multiple shots to engage, regardless of
        # how many fire button presses were counted in the lookback window.
        starting_hp  = e["hp_delta"] + e["hp_after"]
        multi_shot   = starting_hp > 1

        if destroyed:
            verb = "keep firing at" if multi_shot else "fire at"
            tag  = f"{verb} the {name} {pos_phrase} with {weapon} to destroy it"
        else:
            verb = "keep firing at" if multi_shot else "fire at"
            tag  = f"{verb} the {name} {pos_phrase} with {weapon}"

        maneuvers.append(dict(
            tag        = tag,
            destroyed  = destroyed,
            # span: fire preparation → last hp-loss step
            t_aim      = t_aim,
            t_start    = e["t_start"],   # first hp-loss step
            t_end      = e["t_end"],     # last hp-loss step
            fire_steps = fire_steps,
            # enemy info
            slot       = e["slot"],
            level      = e["level"],
            enemy_type = e["enemy_type"],
            enemy_name = e["enemy_name"],
            hp_delta   = e["hp_delta"],
            enemy_x    = e["enemy_x"],
            enemy_y    = e["enemy_y"],
            player_x   = e["player_x"],
            player_y   = e["player_y"],
            weapon     = e["weapon"],
        ))

    return maneuvers


def detect_aim_shots():
    pass



def main():
    action_path = 'synthetic/mc_trace/win_level1_202603301145.npz'

    maneuvers = collect_hp_loss_events(action_path)
    print(f"\n{'aim':>5}  {'hit steps':>11}  {'#fire':>5}  {'slot':>4}  {'lvl':>3}  "
          f"{'type':>6}  {'hp':>4}  {'weapon':<12}  tag")
    print("-" * 100)
    for m in maneuvers:
        hit_range = (f"{m['t_start']}-{m['t_end']}" if m['t_start'] != m['t_end']
                     else str(m['t_start']))
        aim   = str(m["t_aim"]) if m["t_aim"] is not None else "?"
        nfire = len(m["fire_steps"])
        print(
            f"  {aim:>5}  {hit_range:>9}  {nfire:>5}  {m['slot']:>4}  {m['level']:>3}  "
            f"{m['enemy_type']:>6}  {m['hp_delta']:>4}  {m['weapon']:<12}  {m['tag']}"
        )
    n_destroy  = sum(1 for m in maneuvers if m["destroyed"])
    n_shoot_at = len(maneuvers) - n_destroy
    print(f"\n{len(maneuvers)} maneuver(s): {n_destroy} DESTROY, {n_shoot_at} SHOOT_AT.")



if __name__ == "__main__":
    main()
