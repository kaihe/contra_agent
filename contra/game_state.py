"""Structured game-state extraction from NES Contra RAM.

State layout — four logical sections
======================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. NUMERIC PLAYER INFO                                       8 values  │
  │  ─────────────────────────────────────────────────────────────────────  │
  │   [0]  scroll_x_total   total horizontal scroll progress (px)           │
  │   [1]  scroll_y_offset  intra-screen vertical pixel offset              │
  │   [2]  lives            remaining lives                                  │
  │   [3]  player_x         screen X  (0 = left edge)                       │
  │   [4]  player_y         screen Y  (0 = top; larger = lower on screen)   │
  │   [5]  player_x_vel     signed; positive = right                        │
  │   [6]  player_y_vel     signed; negative = rising, positive = falling   │
  │   [7]  in_air           1 if airborne, else 0                           │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  2. ONE-HOT PLAYER INFO                                      18 values  │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  aim_dir    (11)  right_up / right_up_diag / right / right_down_diag   │
  │                   / crouch_right / crouch_left / left_down_diag         │
  │                   / left / left_up_diag / left_up / unknown             │
  │  weapon      (5)  Regular / Machine / Flame / Spray / Laser             │
  │  rapid_fire  (2)  off / on                                              │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  3. NUMERIC ENEMY POS AND TYPE                               64 values  │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  16 slots × (type, screen_x, screen_y, hp)                              │
  │  Inactive slots (type = 0 or hp ≥ 0xf0) are zeroed out.               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  4. ONE-HOT SCENE TYPE                                       28 values  │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  level          (8)   L1 .. L8                                          │
  │  level_routine (11)   routine index 0x00 .. 0x0a                        │
  │  location_type  (3)   outdoor / indoor / indoor_boss                    │
  │  scrolling_type (2)   horizontal / vertical                             │
  │  indoor_cleared (2)   no / yes                                          │
  │  boss_defeated  (2)   no / yes                                          │
  └─────────────────────────────────────────────────────────────────────────┘

  Total: 8 + 18 + 64 + 28 = 118 float32 values  (STATE_DIM)
"""

import numpy as np

# ── RAM addresses ──────────────────────────────────────────────────────────────
# All addresses are for player 1; the byte at offset +1 is player 2 (unused).
_ADDR_LIVES      = 0x32
_ADDR_SCREEN_NUM = 0x64    # high byte of 16-bit scroll progress (screen index)
_ADDR_SCROLL_OFF = 0x65    # low byte: intra-screen pixel offset
_ADDR_PLAYER_X   = 0x0334  # SPRITE_X_POS[0] — player 1 screen X
_ADDR_PLAYER_Y   = 0x031A  # SPRITE_Y_POS[0] — player 1 screen Y
_ADDR_X_VEL      = 0x98    # PLAYER_X_VELOCITY player 1
_ADDR_Y_VEL      = 0xC6    # PLAYER_Y_FAST_VELOCITY player 1 (positive = down)
_ADDR_JUMP       = 0xA0    # PLAYER_JUMP_STATUS player 1
_ADDR_EDGE_FALL  = 0xA4    # EDGE_FALL_CODE player 1
_ADDR_AIM_DIR    = 0xC2    # PLAYER_AIM_DIR player 1
_ADDR_WEAPON     = 0xAA    # P1_CURRENT_WEAPON: low nibble = type, bit 4 = rapid fire
_ADDR_ENEMY_TYPE = 0x0528  # ENEMY_TYPE[0..15]
_ADDR_ENEMY_X    = 0x033E  # ENEMY_X_POS[0..15]
_ADDR_ENEMY_Y    = 0x0324  # ENEMY_Y_POS[0..15]
_ADDR_ENEMY_HP   = 0x0578  # ENEMY_HP[0..15]
ENEMY_SLOTS      = 16      # slots 0..15 ($00..$0f), confirmed by remove_all_enemies loop
_ADDR_LEVEL           = 0x30   # CURRENT_LEVEL: #$00-#$07 = L1-L8, #$09 = game over
_ADDR_LEVEL_ROUTINE   = 0x2C   # LEVEL_ROUTINE_INDEX: index into level_routine_ptr_tbl
_ADDR_LOCATION_TYPE   = 0x40   # LEVEL_LOCATION_TYPE: $00=outdoor, $01=indoor, $80=boss
_ADDR_SCROLLING_TYPE  = 0x41   # LEVEL_SCROLLING_TYPE: $00=horizontal, $01=vertical
_ADDR_INDOOR_CLEARED  = 0x37   # INDOOR_SCREEN_CLEARED: 0=no, 1=cleared, $80=cleared+fence
_ADDR_BOSS_DEFEATED   = 0x3B   # BOSS_DEFEATED_FLAG: 0=no, non-zero=yes

# ── Aim direction ──────────────────────────────────────────────────────────────
# Source: player_aim_dir_ptr_tbl (bank6.asm), PLAYER_AIM_DIR comment (ram.asm).
#
# 10 named directions (#$00-#$09) encoding facing side × shooting angle.
# There is no plain "down" — only diagonal-down (#$03/#$06) or crouch (#$04/#$05).
# #$0a exists in the RAM range but is undefined in the sprite table ("??"); it
# falls through to the "unknown" catch-all at index 10.
#
#   value  name              description
#   #$00   right_up          facing right, aiming straight up
#   #$01   right_up_diag     facing right, aiming diagonally up-right
#   #$02   right             facing right, aiming flat
#   #$03   right_down_diag   facing right, aiming diagonally down-right
#   #$04   crouch_right      crouching, facing right
#   #$05   crouch_left       crouching, facing left
#   #$06   left_down_diag    facing left, aiming diagonally down-left
#   #$07   left              facing left, aiming flat
#   #$08   left_up_diag      facing left, aiming diagonally up-left
#   #$09   left_up           facing left, aiming straight up
#   #$0a   unknown           undefined in sprite table (catch-all)
AIM_DIR_NAMES: tuple[str, ...] = (
    "right_up",
    "right_up_diag",
    "right",
    "right_down_diag",
    "crouch_right",
    "crouch_left",
    "left_down_diag",
    "left",
    "left_up_diag",
    "left_up",
    "unknown",
)
N_AIM_DIR = len(AIM_DIR_NAMES)  # 11


NUMERIC_PLAYER_DIM = 8


def _xscroll_total(ram: np.ndarray) -> int:
    """16-bit scroll progress: (screen_number << 8) | intra-screen offset."""
    return (int(ram[_ADDR_SCREEN_NUM]) << 8) | int(ram[_ADDR_SCROLL_OFF])


def _signed8(byte: int) -> int:
    """Reinterpret a uint8 RAM byte as a signed int8."""
    return byte - 256 if byte > 127 else byte


def _in_air(ram: np.ndarray) -> int:
    """Return 1 if player 1 is airborne, else 0.

    Three independent conditions all indicate the player is off the ground:
      - PLAYER_JUMP_STATUS ($a0) non-zero  → actively mid-jump
      - PLAYER_Y_FAST_VELOCITY ($c6) non-zero → vertical momentum (rising or falling)
      - EDGE_FALL_CODE ($a4) non-zero  → walked off a ledge without jumping
    """
    return int(
        ram[_ADDR_JUMP] != 0
        or ram[_ADDR_Y_VEL] != 0
        or ram[_ADDR_EDGE_FALL] != 0
    )


def _get_numeric_player_info(ram: np.ndarray) -> np.ndarray:
    """Extract section 1: numeric player info (8 dims).

    Index  Name             Source address  Notes
    -----  ---------------  --------------  ----------------------------------
      0    scroll_x_total   $64:$65         (screen_num << 8) | scroll_offset
      1    scroll_y_offset  $65             intra-screen pixel offset
      2    lives            $32
      3    player_x         $0334           SPRITE_X_POS[0], screen X (0 = left)
      4    player_y         $031a           SPRITE_Y_POS[0], screen Y (0 = top, increases downward)
                                            Higher value → lower on screen; decreases while jumping
      5    player_x_vel     $98             signed; positive = right
      6    player_y_vel     $c6             signed; negative = rising (-5 at jump start per ASM),
                                            positive = falling; 0 on ground
      7    in_air           derived         1 if airborne, else 0

    Returns
    -------
    state : np.ndarray, shape (8,), dtype float32
    """
    state = np.empty(NUMERIC_PLAYER_DIM, dtype=np.float32)
    state[0] = _xscroll_total(ram)
    state[1] = int(ram[_ADDR_SCROLL_OFF])
    state[2] = int(ram[_ADDR_LIVES])
    state[3] = int(ram[_ADDR_PLAYER_X])
    state[4] = int(ram[_ADDR_PLAYER_Y])
    state[5] = _signed8(int(ram[_ADDR_X_VEL]))
    state[6] = _signed8(int(ram[_ADDR_Y_VEL]))
    state[7] = _in_air(ram)
    return state


# ── Weapon ────────────────────────────────────────────────────────────────────
# Source: P1_CURRENT_WEAPON comment (ram.asm $aa).
# The byte is bit-packed: low nibble = weapon index, bit 4 = rapid-fire flag.
#
#   value  name
#   #$00   Regular
#   #$01   Machine Gun
#   #$02   Flame Thrower
#   #$03   Spray  (the "S" spread weapon)
#   #$04   Laser
WEAPON_NAMES: tuple[str, ...] = ("Regular", "Machine", "Flame", "Spray", "Laser")
N_WEAPON     = len(WEAPON_NAMES)   # 5
N_RAPID_FIRE = 2                   # off / on
WEAPON_INFO_DIM = N_WEAPON + N_RAPID_FIRE  # 7


def _get_onehot_weapon_info(ram: np.ndarray) -> np.ndarray:
    """One-hot encode player 1's weapon type and rapid-fire flag (7 dims).

    Both values are packed in P1_CURRENT_WEAPON ($aa):
      - bits 0-3  weapon index  → one-hot over 5 weapon types
      - bit 4     rapid-fire    → one-hot [off, on]

    Layout of the returned vector:
      [0..4]  weapon    Regular / Machine / Flame / Spray / Laser
      [5..6]  rapid_fire  off / on

    Returns
    -------
    vec : np.ndarray, shape (7,), dtype float32
    """
    byte = int(ram[_ADDR_WEAPON])
    weapon_idx = byte & 0x0F
    rapid_fire = (byte >> 4) & 0x01

    vec = np.zeros(WEAPON_INFO_DIM, dtype=np.float32)
    if weapon_idx < N_WEAPON:
        vec[weapon_idx] = 1.0
    vec[N_WEAPON + rapid_fire] = 1.0
    return vec


def _onehot_aim_dir(ram: np.ndarray) -> np.ndarray:
    """One-hot encode player 1's aim direction (11 dims).

    Reads PLAYER_AIM_DIR ($c2).  Values #$00-#$09 each map to a named slot;
    #$0a and any out-of-range byte map to the "unknown" slot at index 10.

    Returns
    -------
    vec : np.ndarray, shape (11,), dtype float32
        Exactly one element is 1.0.  Index correspondence: see AIM_DIR_NAMES.
    """
    vec = np.zeros(N_AIM_DIR, dtype=np.float32)
    d = int(ram[_ADDR_AIM_DIR])
    idx = d if d < N_AIM_DIR else N_AIM_DIR - 1
    vec[idx] = 1.0
    return vec


# ── Numeric enemy pos and type ─────────────────────────────────────────────────
# Slots with ENEMY_TYPE == 0 or ENEMY_HP >= $f0 are inactive; the game uses
# HP >= $f0 as a sentinel for spawned-but-not-yet-visible enemies.
# Inactive slots are zeroed in the output so the model sees a clean zero vector.
ENEMY_INFO_DIM = ENEMY_SLOTS * 4  # 64: 16 slots × (type, x, y, hp)


def _get_numeric_enemy_info(ram: np.ndarray) -> np.ndarray:
    """Extract section 3: numeric enemy positions and types (64 dims).

    Reads all 16 enemy slots from RAM and packs them into a flat float32 array.
    Each slot occupies 4 consecutive elements:

      slot_offset + 0  enemy_type   ENEMY_TYPE[$0528 + slot]
      slot_offset + 1  enemy_x      ENEMY_X_POS[$033e + slot]  screen X
      slot_offset + 2  enemy_y      ENEMY_Y_POS[$0324 + slot]  screen Y
      slot_offset + 3  enemy_hp     ENEMY_HP[$0578 + slot]

    Inactive slots (type == 0 or hp >= $f0) are written as four zeros.

    Returns
    -------
    state : np.ndarray, shape (64,), dtype float32
    """
    state = np.zeros(ENEMY_INFO_DIM, dtype=np.float32)
    for slot in range(ENEMY_SLOTS):
        etype = int(ram[_ADDR_ENEMY_TYPE + slot])
        ehp   = int(ram[_ADDR_ENEMY_HP   + slot])
        if etype == 0 or ehp >= 0xF0:
            continue
        base = slot * 4
        state[base]     = etype
        state[base + 1] = int(ram[_ADDR_ENEMY_X + slot])
        state[base + 2] = int(ram[_ADDR_ENEMY_Y + slot])
        state[base + 3] = ehp
    return state


# ── One-hot scene type ─────────────────────────────────────────────────────────
N_LEVEL         = 8   # L1..L8 (CURRENT_LEVEL $00-$07; $09 = game over, clamped out)
N_LEVEL_ROUTINE = 11  # LEVEL_ROUTINE_INDEX $00-$0a
N_LOCATION      = 3   # outdoor($00) / indoor($01) / boss($80)
N_SCROLLING     = 2   # horizontal($00) / vertical($01)
N_INDOOR_CLR    = 2   # not cleared / cleared (1 or $80 both mean cleared)
N_BOSS_DEF      = 2   # not defeated / defeated (any non-zero value)
SCENE_TYPE_DIM  = N_LEVEL + N_LEVEL_ROUTINE + N_LOCATION + N_SCROLLING + N_INDOOR_CLR + N_BOSS_DEF  # 28


def _get_onehot_scene_type(ram: np.ndarray) -> np.ndarray:
    """One-hot encode section 4: scene type (28 dims).

    Fields and their RAM sources:

      dims   field           address  encoding
      -----  --------------  -------  ----------------------------------------
      0..7   level           $30      #$00-#$07 → L1-L8; out-of-range → all zero
      8..18  level_routine   $2c      #$00-#$0a; out-of-range → all zero
      19..21 location_type   $40      $00→outdoor, $01→indoor, $80→indoor_boss
      22..23 scrolling_type  $41      $00→horizontal, $01→vertical
      24..25 indoor_cleared  $37      0→no; 1 or $80→yes (fence also removed)
      26..27 boss_defeated   $3b      0→no; any non-zero→yes

    Returns
    -------
    vec : np.ndarray, shape (28,), dtype float32
    """
    vec = np.zeros(SCENE_TYPE_DIM, dtype=np.float32)
    off = 0

    # level (8)
    lvl = int(ram[_ADDR_LEVEL])
    if lvl < N_LEVEL:
        vec[off + lvl] = 1.0
    off += N_LEVEL

    # level_routine (11)
    rtn = int(ram[_ADDR_LEVEL_ROUTINE])
    if rtn < N_LEVEL_ROUTINE:
        vec[off + rtn] = 1.0
    off += N_LEVEL_ROUTINE

    # location_type (3): three non-contiguous raw values
    loc = int(ram[_ADDR_LOCATION_TYPE])
    loc_idx = {0x00: 0, 0x01: 1, 0x80: 2}.get(loc)
    if loc_idx is not None:
        vec[off + loc_idx] = 1.0
    off += N_LOCATION

    # scrolling_type (2)
    scr = int(ram[_ADDR_SCROLLING_TYPE])
    if scr < N_SCROLLING:
        vec[off + scr] = 1.0
    off += N_SCROLLING

    # indoor_cleared (2)
    vec[off + (1 if ram[_ADDR_INDOOR_CLEARED] != 0 else 0)] = 1.0
    off += N_INDOOR_CLR

    # boss_defeated (2)
    vec[off + (1 if ram[_ADDR_BOSS_DEFEATED] != 0 else 0)] = 1.0

    return vec


# ── Total state dimension ──────────────────────────────────────────────────────
STATE_DIM = NUMERIC_PLAYER_DIM + WEAPON_INFO_DIM + N_AIM_DIR + ENEMY_INFO_DIM + SCENE_TYPE_DIM
# = 8 + 7 + 11 + 64 + 28 = 118

_LOCATION_NAMES = {0x00: "outdoor", 0x01: "indoor", 0x80: "indoor_boss"}
_SCROLLING_NAMES = ("horizontal", "vertical")


def _decode_ram(ram: np.ndarray) -> dict:
    """Decode all game-state fields from RAM into Python native types.

    Returns a plain dict — no numpy types — suitable for logging, serialisation,
    or feeding into :func:`describe_game_state`.

    Keys
    ----
    scroll_x_total, scroll_y_offset, lives,
    player_x, player_y, player_x_vel, player_y_vel, in_air,
    aim_dir, weapon, rapid_fire,
    enemies  (list of dicts: slot, type, x, y, hp — inactive slots omitted),
    level (0-indexed int), level_routine, location, scrolling,
    indoor_cleared, boss_defeated
    """
    # weapon byte
    wbyte = int(ram[_ADDR_WEAPON])
    weapon_idx = wbyte & 0x0F
    rapid_fire = bool((wbyte >> 4) & 0x01)

    # aim direction
    aim_val = int(ram[_ADDR_AIM_DIR])
    aim_idx = aim_val if aim_val < N_AIM_DIR else N_AIM_DIR - 1

    # active enemies
    enemies = []
    for slot in range(ENEMY_SLOTS):
        etype = int(ram[_ADDR_ENEMY_TYPE + slot])
        ehp   = int(ram[_ADDR_ENEMY_HP   + slot])
        if etype == 0 or ehp >= 0xF0:
            continue
        enemies.append({
            "slot": slot,
            "type": etype,
            "x":    int(ram[_ADDR_ENEMY_X + slot]),
            "y":    int(ram[_ADDR_ENEMY_Y + slot]),
            "hp":   ehp,
        })

    # scene
    lvl     = int(ram[_ADDR_LEVEL])
    loc_raw = int(ram[_ADDR_LOCATION_TYPE])
    scr_raw = int(ram[_ADDR_SCROLLING_TYPE])

    return {
        "scroll_x_total":  _xscroll_total(ram),
        "scroll_y_offset": int(ram[_ADDR_SCROLL_OFF]),
        "lives":           int(ram[_ADDR_LIVES]),
        "player_x":        int(ram[_ADDR_PLAYER_X]),
        "player_y":        int(ram[_ADDR_PLAYER_Y]),
        "player_x_vel":    _signed8(int(ram[_ADDR_X_VEL])),
        "player_y_vel":    _signed8(int(ram[_ADDR_Y_VEL])),
        "in_air":          bool(_in_air(ram)),
        "aim_dir":         AIM_DIR_NAMES[aim_idx],
        "weapon":          WEAPON_NAMES[weapon_idx] if weapon_idx < N_WEAPON else "Unknown",
        "rapid_fire":      rapid_fire,
        "enemies":         enemies,
        "level":           lvl,
        "level_routine":   int(ram[_ADDR_LEVEL_ROUTINE]),
        "location":        _LOCATION_NAMES.get(loc_raw, "unknown"),
        "scrolling":       _SCROLLING_NAMES[scr_raw] if scr_raw < 2 else "unknown",
        "indoor_cleared":  bool(ram[_ADDR_INDOOR_CLEARED] != 0),
        "boss_defeated":   bool(ram[_ADDR_BOSS_DEFEATED] != 0),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def decode_game_state(env) -> dict:
    """Return a decoded game-state dict from a live stable-retro environment.

    All values are Python built-ins (int / bool / str / list); no numpy types.
    Useful for logging, JSON serialisation, or feeding into :func:`describe_game_state`.

    See :func:`_decode_ram` for the full list of keys.
    """
    return _decode_ram(env.unwrapped.get_ram())


def decode_ram(ram: np.ndarray) -> dict:
    """Return a decoded game-state dict from a raw 2048-byte RAM snapshot.

    All values are Python built-ins (int / bool / str / list); no numpy types.
    Useful for GUI overlays, logging, or JSON serialisation without a live env.

    See :func:`_decode_ram` for the full list of keys.
    """
    return _decode_ram(ram)


def _format_state(s: dict) -> str:
    """Format a decoded state dict (from _decode_ram) as a human-readable string."""
    lines: list[str] = []

    # Player
    lines.append("Player:")
    air_str  = "in the air" if s["in_air"] else "on ground"
    lines.append(
        f"  Position: ({s['player_x']}, {s['player_y']})"
        f"  vel: ({s['player_x_vel']:+d}, {s['player_y_vel']:+d})  {air_str}"
    )
    lines.append(f"  Lives: {s['lives']}  scroll: {s['scroll_x_total']:#06x}")
    fire_str = "rapid fire" if s["rapid_fire"] else "normal fire"
    lines.append(f"  Weapon: {s['weapon']} ({fire_str})  aim: {s['aim_dir']}")

    lines.append("")

    # Enemies
    active = s["enemies"]
    if active:
        lines.append(f"Enemies ({len(active)} active):")
        lines.append(f"  {'Slot':>4}  {'Type':>4}  {'X':>4}  {'Y':>4}  {'HP':>3}")
        lines.append(f"  {'----':>4}  {'----':>4}  {'----':>4}  {'----':>4}  {'---':>3}")
        for e in active:
            lines.append(
                f"  {e['slot']:>4}  {e['type']:>4}  {e['x']:>4}  {e['y']:>4}  {e['hp']:>3}"
            )
    else:
        lines.append("Enemies: none active")

    lines.append("")

    # Scene
    lvl_str = f"L{s['level'] + 1}" if s["level"] < N_LEVEL else f"game over ({s['level']:#x})"
    lines.append("Scene:")
    lines.append(f"  Level: {lvl_str}  routine: {s['level_routine']:#04x}")
    lines.append(f"  Location: {s['location']}  scrolling: {s['scrolling']}")
    if s["location"] in ("indoor", "indoor_boss"):
        lines.append(f"  Indoor: {'cleared' if s['indoor_cleared'] else 'not cleared'}")
    lines.append(f"  Boss: {'defeated' if s['boss_defeated'] else 'not defeated'}")

    return "\n".join(lines)


def describe_game_state(env) -> str:
    """Return a human-readable description of the current game state.

    Reads RAM from a live stable-retro environment.  For replaying saved
    snapshots without an emulator use :func:`describe_ram` instead.

    Example output::

        Player:
          Position: (120, 180)  vel: (+2, 0)  on ground
          Lives: 3  scroll: 0x0265
          Weapon: Laser (rapid fire)  aim: right_up_diag

        Enemies (2 active):
          Slot  Type     X     Y   HP
          ----  ----  ----  ----  ---
             2    12   150    80    4
             5     7    60   100    2

        Scene:
          Level: L3  routine: 0x02
          Location: indoor  scrolling: horizontal
          Indoor: not cleared
          Boss: not defeated

    Parameters
    ----------
    env : retro.Env
        A stable-retro environment running Contra (US).
    """
    return _format_state(_decode_ram(env.unwrapped.get_ram()))


def describe_ram(ram: np.ndarray) -> str:
    """Format a raw 2048-byte RAM snapshot as a human-readable game state string.

    Equivalent to :func:`describe_game_state` but accepts the RAM array directly,
    which is useful when replaying saved per-frame snapshots (e.g. ``ram.npy``)
    without a live emulator.

    Parameters
    ----------
    ram : np.ndarray
        Shape (2048,), dtype uint8 — as returned by ``env.unwrapped.get_ram()``
        or loaded from a saved ``ram.npy`` file.
    """
    return _format_state(_decode_ram(ram))


def get_game_state(env) -> np.ndarray:
    """Extract the full structured state vector from a stable-retro environment.

    Concatenates all four sections in order:

      section                  dims  indices
      -----------------------  ----  -----------
      1. numeric player info      8  [0:8)
      2. one-hot player info  11+7=18  [8:26)
           aim_dir              11
           weapon + rapid_fire   7
      3. numeric enemy info      64  [26:90)
      4. one-hot scene type      28  [90:118)

    Parameters
    ----------
    env : retro.Env
        A stable-retro environment running Contra (US) with RAM accessible
        via env.unwrapped.get_ram().

    Returns
    -------
    state : np.ndarray, shape (118,), dtype float32
    """
    ram = env.unwrapped.get_ram()
    return np.concatenate([
        _get_numeric_player_info(ram),   # 8
        _onehot_aim_dir(ram),            # 11
        _get_onehot_weapon_info(ram),    # 7
        _get_numeric_enemy_info(ram),    # 64
        _get_onehot_scene_type(ram),     # 28
    ])


def state_from_ram(ram: np.ndarray) -> np.ndarray:
    """Extract 118-dim state vector from a raw 2048-byte RAM snapshot."""
    return np.concatenate([
        _get_numeric_player_info(ram),   # 8
        _onehot_aim_dir(ram),            # 11
        _get_onehot_weapon_info(ram),    # 7
        _get_numeric_enemy_info(ram),    # 64
        _get_onehot_scene_type(ram),     # 28
    ])
