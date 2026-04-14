import numpy as np


class ContraEvent:
    """Describes a single reward-shaping or narrative event in Contra.

    Attributes
    ----------
    tag         : identifier and log label (UPPER_CASE)
    important   : if True, the event is included in the narrative event log
    weight      : reward multiplier; 0.0 for pure-narrative events
    """

    def __init__(self, tag: str, desc: str, trigger_fn, weight: float,
                 important: bool = False, detail_fn=None):
        self.tag         = tag
        self._desc       = desc
        self._trigger_fn = trigger_fn
        self.weight      = weight
        self.important   = important
        self._detail_fn  = detail_fn

    def desc(self, pre_ram: np.ndarray = None, curr_ram: np.ndarray = None) -> str:
        if self._detail_fn is not None and pre_ram is not None and curr_ram is not None:
            detail = self._detail_fn(pre_ram, curr_ram)
            if detail:
                return f"{self._desc} ({detail})"
        return self._desc

    def trigger(self, pre_ram: np.ndarray, curr_ram: np.ndarray) -> float:
        return self._trigger_fn(pre_ram, curr_ram)

    def get_reward(self, pre_ram: np.ndarray, curr_ram: np.ndarray) -> float:
        return self.trigger(pre_ram, curr_ram) * self.weight


# ── RAM addresses (from Contra-Nes/data.json) ────────────────────────────────
ADDR_LIVES          = 50     # |i1   player lives
ADDR_LEVEL          = 48     # |u1   current level (0-indexed)
ADDR_WEAPON         = 0xAA   # |u1   bit4 = rapid fire flag; low nibble = weapon type
ADDR_GUN_ANI        = 0x010A # |u1   0x1F on the frame a gun powerup is collected
ADDR_XSCROLL        = 100    # >u2   horizontal scroll position (big-endian, 2 bytes)
ADDR_ENEMY_TYPE     = 0x528  # |u1   start of 16-slot enemy type array  ($0528)
ADDR_ENEMY_HP       = 1400   # |u1   start of 16-slot enemy HP array    ($0578)
ADDR_ENEMY_HP_COUNT = 16     #       number of enemy HP slots
ENEMY_TYPE_FALLING_ROCK    = 0x13  # spawns endlessly from rock cave on L3; same value is a real enemy on all other levels
ADDR_INDOOR_CLEARED = 0x37   # |u1   INDOOR_SCREEN_CLEARED
ADDR_WALL_CORE_LEFT = 0x86   # |u1   WALL_CORE_REMAINING
ADDR_XSCROLL_HI     = 0x64   # |u1   LEVEL_SCREEN_NUMBER (indoor/vertical screen index)
ADDR_SCROLL_OFF     = 0x65   # |u1   LEVEL_SCREEN_SCROLL_OFFSET: pixels scrolled within current screen (0→0xEF); wraps at 0xF0=240
ADDR_PLAYER_ADV     = 0xd0   # |u1   INDOOR_PLAYER_ADV_FLAG: set when player starts walking through door
ADDR_LEVEL_ROUTINE  = 0x2c   # |u1   LEVEL_ROUTINE_INDEX (from ram.asm $2c)
                              #       0x04 = active gameplay
                              #       0x08-0x09 = post-boss transition (end-of-level tune/sequence)
                              #       0x00-0x03 = title/intro screens (loading, lives display, score flash, nametable draw)
                              #       0x05-0x07 = game-over / continue screens

WEAPON_NAMES = {0: "Regular", 1: "MachineGun", 2: "Flamethrower",
                3: "Spread",  4: "Laser"}

# Enemy type → human-readable name, sourced from docs/Enemy Glossary.md.
# Types 0x00-0x0F are common across all levels.
# Types 0x10+ are level-specific: keyed by (level_0indexed, type).
ENEMY_TYPE_NAMES_COMMON = {
    0x00: "Weapon Item",
    0x01: "Bullet",
    0x02: "Pill Box Sensor",
    0x03: "Flying Capsule",
    0x04: "Rotating Gun",
    0x05: "Soldier",
    0x06: "Sniper",
    0x07: "Red Turret",
    0x08: "Wall Cannon",
    0x09: "Unused",
    0x0A: "Wall Plating",
    0x0B: "Mortar Shot",
    0x0C: "Scuba Diver",
    0x0D: "Unused",
    0x0E: "Basquez",
    0x0F: "Basquez Bullet",
}

ENEMY_TYPE_NAMES_BY_LEVEL = {
    0: {  # L1 — Jungle
        0x10: "Bomb Turret", 0x11: "Plated Door", 0x12: "Exploding Bridge",
    },
    1: {  # L2 — Indoor Base
        0x10: "Boss Eye", 0x11: "Roller", 0x12: "Grenade", 0x13: "Wall Turret",
        0x14: "Core", 0x15: "Indoor Soldier", 0x16: "Jumping Soldier",
        0x17: "Grenade Launcher", 0x18: "Group of Four Soldiers",
        0x19: "Indoor Soldier Generator", 0x1A: "Indoor Roller Generator",
        0x1B: "Boss Eye Fire Ring Projectile", 0x1C: "Godomuga",
        0x1D: "Gardegura", 0x1E: "Garth", 0x1F: "Rangel",
        0x20: "Garth and Rangel Generator",
    },
    2: {  # L3 — Waterfall
        0x10: "Floating Rock Platform", 0x11: "Moving Flame", 0x12: "Rock Cave",
        0x13: "Falling Rock", 0x14: "Dragon", 0x15: "Dragon Tentacle Orb",
    },
    3: {  # L4 — Indoor Base (same enemy set as L2)
        0x10: "Boss Eye", 0x11: "Roller", 0x12: "Grenade", 0x13: "Wall Turret",
        0x14: "Core", 0x15: "Indoor Soldier", 0x16: "Jumping Soldier",
        0x17: "Grenade Launcher", 0x18: "Group of Four Soldiers",
        0x19: "Indoor Soldier Generator", 0x1A: "Indoor Roller Generator",
        0x1B: "Boss Eye Fire Ring Projectile", 0x1C: "Godomuga",
        0x1D: "Gardegura", 0x1E: "Garth", 0x1F: "Rangel",
        0x20: "Garth and Rangel Generator",
    },
    4: {  # L5 — Snow Field
        0x10: "Ice Grenade Generator", 0x11: "Ice Grenade", 0x12: "Tank",
        0x13: "Ice Separator", 0x14: "Alien Carrier", 0x15: "Flying Saucer",
        0x16: "Drop Bomb",
    },
    5: {  # L6 — Energy Zone
        0x10: "Fire Beam", 0x11: "Fire Beam", 0x12: "Fire Beam",
        0x13: "Giant Boss Soldier", 0x14: "Spiked Projectile",
    },
    6: {  # L7 — Hangar
        0x10: "Claw", 0x11: "Rising Spiked Wall", 0x12: "Tall Spiked Wall",
        0x13: "Mining Cart Generator", 0x14: "Mining Cart",
        0x15: "Stationary Mining Cart", 0x16: "Armored Door",
        0x17: "Mortar Launcher", 0x18: "Soldier Generator",
    },
    7: {  # L8 — Alien's Lair
        0x10: "Emperor Demon Dragon God Java", 0x11: "Bundle",
        0x12: "Wadder", 0x13: "Poisonous Insect Gel",
        0x14: "Bugger", 0x15: "Eggron", 0x16: "Gomeramos King",
    },
}


def enemy_type_name(etype: int, level_0indexed: int) -> str:
    """Return a readable name for an enemy type given the current level (0-indexed)."""
    if etype <= 0x0F:
        return ENEMY_TYPE_NAMES_COMMON.get(etype, f"unknown(0x{etype:02x})")
    return ENEMY_TYPE_NAMES_BY_LEVEL.get(level_0indexed, {}).get(
        etype, f"unknown(0x{etype:02x})"
    )


def _xscroll(ram: np.ndarray) -> int:
    return (int(ram[ADDR_XSCROLL]) << 8) | int(ram[ADDR_XSCROLL + 1])

def _yscroll_delta(pre: np.ndarray, cur: np.ndarray) -> float:
    """Pixel progress in vertical scroll offset, handling wrap-around at 0xF0=240."""
    delta = int(cur[ADDR_SCROLL_OFF]) - int(pre[ADDR_SCROLL_OFF])
    if delta < -100:   # wrapped: e.g. 238 → 2 means +4, not -236
        delta += 240
    return max(0.0, float(delta))

def _gun_ani_rising(pre: np.ndarray, cur: np.ndarray) -> bool:
    return int(cur[ADDR_GUN_ANI]) == 0x1F and int(pre[ADDR_GUN_ANI]) != 0x1F

def _weapon_type(ram: np.ndarray) -> int:
    return int(ram[ADDR_WEAPON]) & 0x0F


# ── Individual event instances ────────────────────────────────────────────────

def _enemy_hit_trigger(pre: np.ndarray, cur: np.ndarray) -> float:
    """Sum HP decrements across all 16 slots, with level-aware handling of type 0x13.

    Type 0x13 meaning by level (0-indexed):
      L1 (0): no type-0x13 defined — skip
      L2 (1): Wall Turret         — include
      L3 (2): Falling Rock        — EXCLUDE (respawns endlessly from rock caves)
      L4 (3): Wall Turret         — include
      L5 (4): Ice Separator       — include
      L6 (5): Giant Boss Soldier  — include (was formerly EV_L6_BOSS_HIT)
      L7 (6): Mining Cart Gen.    — include
      L8 (7): Poisonous Insect Gel — include
    """
    level = int(pre[ADDR_LEVEL])
    total = 0.0
    for slot in range(ADDR_ENEMY_HP_COUNT):
        etype  = int(pre[ADDR_ENEMY_TYPE + slot])
        pre_hp = int(pre[ADDR_ENEMY_HP   + slot])
        cur_hp = int(cur[ADDR_ENEMY_HP   + slot])
        if pre_hp >= 0xf0 or cur_hp >= 0xf0:
            continue
        if etype == 0x01:             # Bullet — projectile object, no real HP
            continue
        if etype == ENEMY_TYPE_FALLING_ROCK:
            if level == 0:    # L1 — no type-0x13 enemies defined
                continue
            elif level == 1:  # L2 — Wall Turret: include
                pass
            elif level == 2:  # L3 — Falling Rock: exclude (endless respawn)
                continue
            elif level == 3:  # L4 — Wall Turret: include
                pass
            elif level == 4:  # L5 — Ice Separator: include
                pass
            elif level == 5:  # L6 — Giant Boss Soldier (Gordea): include
                pass
            elif level == 6:  # L7 — Mining Cart Generator: include
                pass
            elif level == 7:  # L8 — Poisonous Insect Gel: include
                pass
            else:
                continue
        delta = pre_hp - cur_hp
        if delta > 0:
            total += float(delta)
    return total


def _enemy_hit_detail(pre: np.ndarray, cur: np.ndarray) -> str:
    level = int(pre[ADDR_LEVEL])
    parts = []
    for slot in range(ADDR_ENEMY_HP_COUNT):
        etype  = int(pre[ADDR_ENEMY_TYPE + slot])
        pre_hp = int(pre[ADDR_ENEMY_HP   + slot])
        cur_hp = int(cur[ADDR_ENEMY_HP   + slot])
        if pre_hp >= 0xf0 or cur_hp >= 0xf0:
            continue
        if etype == 0x01:
            continue
        if etype == ENEMY_TYPE_FALLING_ROCK and level == 2:
            continue
        delta = pre_hp - cur_hp
        if delta > 0:
            parts.append(f"{enemy_type_name(etype, level)} -{delta}HP")
    return ", ".join(parts)


EV_ENEMY_HIT = ContraEvent(
    tag="ENEMY_HIT",
    desc="Sum of HP decrements across all active enemy slots. "
         "Type 0x13 is excluded only on L3 (Falling Rock); on all other levels "
         "it is a real enemy (Wall Turret, Giant Boss Soldier, etc.).",
    trigger_fn=_enemy_hit_trigger,
    detail_fn=_enemy_hit_detail,
    weight=1.0,
)


EV_PUSH_FORWARD = ContraEvent(
    tag="PUSH_FORWARD",
    desc="Reward forward scroll progress; normalised by pixels per screen width.",
    trigger_fn=lambda pre, cur: (_xscroll(cur) - _xscroll(pre)) ,
    weight=1.0/30.0,
)

EV_SPREAD_PICK = ContraEvent(
    tag="SPREAD_PICK",
    desc="Reward picking up the spread gun; penalise losing it.",
    trigger_fn=lambda pre, cur: (
        1.0  if (_gun_ani_rising(pre, cur) and _weapon_type(cur) == 3 and _weapon_type(pre) != 3)
        else -1.0 if (_weapon_type(pre) == 3 and _weapon_type(cur) != 3)
        else 0.0
    ),
    weight=10.0,
)

EV_SPREAD_LOST = ContraEvent(
    tag="SPREAD_LOST",
    desc="Lost the spread gun.",
    trigger_fn=lambda pre, cur: (
        1.0 if (_weapon_type(pre) == 3 and _weapon_type(cur) != 3) else 0.0
    ),
    weight=-200.0,
    important=True,
    detail_fn=lambda _, cur: f"Spread → {WEAPON_NAMES.get(_weapon_type(cur), '?')}",
)

# RAM[0xB4]=180 bit 0: PLAYER_DEATH_FLAG — set on enemy hit, reliable, multi-frame.
# RAM[0x90]=144: PLAYER_STATE — 0x02 when dead from any cause including pit falls.
ADDR_PLAYER_STATE = 0x90   # |u1  0x02 = dead (any cause)

EV_PLAYER_DIE = ContraEvent(
    tag="PLAYER_DIE",
    desc="Player died (enemy hit via RAM[180] bit0, or pit fall via PLAYER_STATE==0x02).",
    trigger_fn=lambda pre, cur: (
        1.0 if (int(pre[ADDR_PLAYER_STATE]) != 0x02) and
               ((int(cur[180]) & 0x01) or (int(cur[ADDR_PLAYER_STATE]) == 0x02))
        else 0.0
    ),
    weight=-5000.0,
    important=True,
    detail_fn=lambda pre, cur: f"lives {int(pre[50])} → {int(cur[50])}",
)


EV_EXTRA_LIFE = ContraEvent(
    tag="EXTRA_LIFE",
    desc="Player gained a life.",
    trigger_fn=lambda pre, cur: 1.0 if int(cur[ADDR_LIVES]) > int(pre[ADDR_LIVES]) else 0.0,
    weight=0.0,
    important=True,
    detail_fn=lambda pre, cur: f"lives {int(pre[ADDR_LIVES])} → {int(cur[ADDR_LIVES])}",
)

EV_GUN_PICKUP = ContraEvent(
    tag="GUN_PICKUP",
    desc="Weapon type changed via pickup animation.",
    trigger_fn=lambda pre, cur: (
        1.0 if (_gun_ani_rising(pre, cur) and _weapon_type(cur) != _weapon_type(pre))
        else 0.0
    ),
    weight=10.0,
    important=True,
    detail_fn=lambda pre, cur: (
        f"{WEAPON_NAMES.get(_weapon_type(pre), '?')} → {WEAPON_NAMES.get(_weapon_type(cur), '?')}"
        if (_gun_ani_rising(pre, cur) and _weapon_type(cur) != _weapon_type(pre)) else ""
    ),
)

EV_GUN_POWERUP = ContraEvent(
    tag="GUN_POWERUP",
    desc="Rapid fire flag gained via pickup animation.",
    trigger_fn=lambda pre, cur: (
        1.0 if (_gun_ani_rising(pre, cur) and
                _weapon_type(cur) == _weapon_type(pre) and
                bool(cur[ADDR_WEAPON] & 0x10) and not bool(pre[ADDR_WEAPON] & 0x10))
        else 0.0
    ),
    weight=10.0,
    important=True,
    detail_fn=lambda pre, cur: (
        f"{WEAPON_NAMES.get(_weapon_type(cur), '?')} rapid fire"
        if (_gun_ani_rising(pre, cur) and
            _weapon_type(cur) == _weapon_type(pre) and
            bool(cur[ADDR_WEAPON] & 0x10) and not bool(pre[ADDR_WEAPON] & 0x10)) else ""
    ),
)

EV_LEVELUP = ContraEvent(
    tag="LEVELUP",
    desc="Completed the level.",
    trigger_fn=lambda pre, cur: 1.0 if int(cur[ADDR_LEVEL]) > int(pre[ADDR_LEVEL]) else 0.0,
    weight=100.0,
    important=True,
    detail_fn=lambda pre, cur: (
        f"level {int(pre[ADDR_LEVEL]) + 1} → {int(cur[ADDR_LEVEL]) + 1}"
        if int(cur[ADDR_LEVEL]) > int(pre[ADDR_LEVEL]) else ""
    ),
)

EV_GAME_CLEAR = ContraEvent(
    tag="GAME_CLEAR",
    desc="Beat the final boss and cleared the game (Level 8 complete).",
    trigger_fn=lambda pre, cur: (
        1.0 if int(pre[ADDR_LEVEL]) == 7 and int(cur[ADDR_LEVEL]) > 7 else 0.0
    ),
    weight=1000.0,
    important=True,
)

EV_CORE_BROKEN = ContraEvent(
    tag="CORE_BROKEN",
    desc="Indoor screen cleared (wall core destroyed).",
    trigger_fn=lambda pre, cur: (
        1.0 if (int(pre[ADDR_INDOOR_CLEARED]) == 0 and int(cur[ADDR_INDOOR_CLEARED]) != 0)
        else 0.0
    ),
    weight=10.0,
    important=True,
)

# Fires every step while the player is walking through the indoor door (~44 steps per transition).
# Bridges the ~52-step gap between CORE_BROKEN and ROOM_ENTER that exceeds rollout length.
EV_PUSH_INSIDE = ContraEvent(
    tag="PUSH_INSIDE",
    desc="Player is walking through the indoor door (0xd0 non-zero).",
    trigger_fn=lambda _, cur: 1.0 if int(cur[ADDR_PLAYER_ADV]) != 0 else 0.0,
    weight=0.5,
    important=False,
)

EV_PUSH_UP = ContraEvent(
    tag="PUSH_UP",
    desc="Reward vertical scroll progress on climbing levels; normalised like PUSH_FORWARD.",
    trigger_fn=lambda pre, cur: _yscroll_delta(pre, cur),
    weight=1.0 / 2.0,
    important=False,
)

EV_ROOM_ENTER = ContraEvent(
    tag="ROOM_ENTER",
    desc="Entered the next indoor screen.",
    trigger_fn=lambda pre, cur: (
        1.0 if int(cur[ADDR_XSCROLL_HI]) > int(pre[ADDR_XSCROLL_HI]) else 0.0
    ),
    weight=0.0,
    important=False,
    detail_fn=lambda pre, cur: (
        f"screen {int(pre[ADDR_XSCROLL_HI])} → {int(cur[ADDR_XSCROLL_HI])}"
    ),
)


# ── Game-flow phase events (based on LEVEL_ROUTINE_INDEX at $2c) ──────────────
#
# These are not reward events — they let the caller know what phase the game is
# in so it can decide whether to send real actions or no-ops.
#
#   LEVEL_BEGIN       : pre in {0x00..0x03}, cur == 0x04  (fires once when gameplay starts)
#   LEVEL_TRANSITION  : LEVEL_ROUTINE_INDEX in {0x08, 0x09}  (post-boss sequence, end-of-level tune)
#   TITLE_SCREEN      : LEVEL_ROUTINE_INDEX in {0x00..0x03}  (loading, "REST xx", score flash, nametable draw)

def _routine(ram: np.ndarray) -> int:
    return int(ram[ADDR_LEVEL_ROUTINE])


EV_LEVEL_BEGIN = ContraEvent(
    tag="LEVEL_BEGIN",
    desc="Level started: title screen ended and gameplay began (routine 0x00-0x03 → 0x04).",
    trigger_fn=lambda pre, cur: (
        1.0 if _routine(pre) in (0x00, 0x01, 0x02, 0x03) and _routine(cur) == 0x04 else 0.0
    ),
    weight=0.0,
    important=True,
    detail_fn=lambda _, cur: f"level={int(cur[ADDR_LEVEL]) + 1}",
)

EV_LEVEL_TRANSITION = ContraEvent(
    tag="LEVEL_TRANSITION",
    desc="Post-boss end-of-level sequence started (rising edge into routine 0x08-0x09).",
    trigger_fn=lambda pre, cur: (
        1.0 if _routine(pre) not in (0x08, 0x09) and _routine(cur) in (0x08, 0x09) else 0.0
    ),
    weight=0.0,
    important=True,
    detail_fn=lambda _, cur: f"routine=0x{_routine(cur):02x}",
)

EV_TITLE_SCREEN = ContraEvent(
    tag="TITLE_SCREEN",
    desc="Between-level title screens — small per-step reward to motivate waiting.",
    trigger_fn=lambda _, cur: 1.0 if _routine(cur) in (0x00, 0x01, 0x02, 0x03) else 0.0,
    weight=0.0,
    important=False,
    detail_fn=lambda _, cur: f"routine=0x{_routine(cur):02x}",
)


def is_gameplay(ram: np.ndarray) -> bool:
    """Return True when the game is in active play (inputs have effect)."""
    return _routine(ram) == 0x04


# ── Level event lists ─────────────────────────────────────────────────────────

LEVELS_BASE = [
    EV_TITLE_SCREEN,
    EV_LEVEL_BEGIN,
    EV_LEVEL_TRANSITION,
    EV_ENEMY_HIT,
    EV_SPREAD_LOST,
    EV_PLAYER_DIE,
    EV_GUN_PICKUP,
    EV_GUN_POWERUP,
    EV_LEVELUP,
]

LEVELS_PUSH_RIGHT  = [*LEVELS_BASE, EV_PUSH_FORWARD]
LEVELS_PUSH_INSIDE = [*LEVELS_BASE, EV_CORE_BROKEN, EV_PUSH_INSIDE, EV_ROOM_ENTER]
LEVELS_PUSH_UP     = [*LEVELS_BASE, EV_PUSH_UP]

# 0-indexed level → event list (levels not listed fall back to LEVELS_PUSH_RIGHT)
# Contra NES structure:
#   L1 (0): side-scroll jungle      L2 (1): indoor base
#   L3 (2): waterfall climb         L4 (3): indoor base
#   L5 (4): side-scroll snow        L6 (5): indoor base
#   L7 (6): side-scroll             L8 (7): final boss (side-scroll)
EVENTS_BY_LEVEL = {
    0: LEVELS_PUSH_RIGHT,   # L1  jungle
    1: LEVELS_PUSH_INSIDE,       # L2  indoor
    2: LEVELS_PUSH_UP,       # L3  waterfall
    3: LEVELS_PUSH_INSIDE,  # L4  indoor
    4: LEVELS_PUSH_RIGHT,   # L5  snow
    5: LEVELS_PUSH_RIGHT,                      # L6  factory (Giant Boss Soldier type $13 now in EV_ENEMY_HIT)
    6: LEVELS_PUSH_RIGHT,   # L7  side-scroll
    7: [*LEVELS_PUSH_RIGHT, EV_GAME_CLEAR],   # L8  final boss
}


def _level_events(pre_ram: np.ndarray) -> list:
    """Return the event list for the level the player was in at the start of a step."""
    return EVENTS_BY_LEVEL.get(int(pre_ram[ADDR_LEVEL]), LEVELS_PUSH_RIGHT)

def get_level(ram: np.ndarray) -> int:
    """Return the current 1-indexed level from RAM."""
    return int(ram[ADDR_LEVEL]) + 1

def scan_events(prev_ram: np.ndarray, curr_ram: np.ndarray,
                step: int) -> list[dict]:
    """Return narrative event dicts for one step, driven by the current level in RAM."""
    events = []
    for ev in _level_events(prev_ram):
        if not ev.important:
            continue
        if ev.trigger(prev_ram, curr_ram):
            detail = ev._detail_fn(prev_ram, curr_ram) if ev._detail_fn else ""
            events.append({"step": step, "tag": ev.tag, "detail": detail})
    return events


def compute_reward(pre_ram: np.ndarray, curr_ram: np.ndarray) -> float:
    """Level-aware reward: sum of all event rewards for the level at step start."""
    return sum(ev.get_reward(pre_ram, curr_ram) for ev in _level_events(pre_ram))
