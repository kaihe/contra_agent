# Reading entity positions from Contra RAM (`contra/entities.py`)

How the agent can know, from a single RAM snapshot, where the player, every
enemy, and every bullet (enemy **and** player) are on screen — for reward /
observation features or for drawing debug overlays.

## Why it works: sprite coords *are* pixel coords

The NES PPU renders a 256×240 frame. Each on-screen actor stores its position in
work RAM as a **sprite coordinate** in that same space:

- `x ∈ [0, 255]`, `y ∈ [0, 239]`
- **y grows downward** (0 = top of screen)

So a value read from RAM is directly the actor's pixel position on the frame — no
transform needed to locate it, and only a small overscan shift to *draw* on a
cropped frame (see below).

## Address map

Player 1:

| what | address | notes |
|---|---|---|
| `SPRITE_X_POS` | `$0334` | player on-screen x |
| `SPRITE_Y_POS` | `$031a` | player on-screen y |

Enemies **and enemy bullets** share one 16-slot array (parallel arrays, index `i` = `0..15`):

| what | base address | notes |
|---|---|---|
| `ENEMY_ROUTINE` | `$04b8` | **slot is live iff `!= 0`** |
| `ENEMY_TYPE` | `$0528` | `0x01` = enemy **bullet**; anything else = an enemy |
| `ENEMY_X_POS` | `$033e` | on-screen x |
| `ENEMY_Y_POS` | `$0324` | on-screen y |

Player bullets — their own 16-slot array:

| what | base address | notes |
|---|---|---|
| `PLAYER_BULLET_SLOT` | `$0388` | **slot is active iff `!= 0`** (holds weapon-type+1) |
| `PLAYER_BULLET_X_POS` | `$03c8` | on-screen x |
| `PLAYER_BULLET_Y_POS` | `$03b8` | on-screen y |
| `PLAYER_BULLET_OWNER` | `$0448` | `0` = P1, `1` = P2 |

## The "is this slot live?" test

The arrays always have 16 slots; most are empty at any moment. You must gate on
the liveness byte, **not** on position (a stale slot keeps its last position):

- **enemy / enemy-bullet slot** is live when `ENEMY_ROUTINE[$04b8+i] != 0`
  (this is exactly what the reference Lua *Show Enemy Positions* checks).
- **player-bullet slot** is active when `PLAYER_BULLET_SLOT[$0388+i] != 0`
  (the fire code treats `0` as an empty slot — `bank6.asm` `@find_bullet_slot`).

Within the live enemy slots, `ENEMY_TYPE == 0x01` separates gunfire from enemies.
This is because enemy bullets are *created as enemies of type 1*
(`create_enemy_bullet`, `bank7.asm:9849`):

```asm
create_enemy_bullet:
    jsr find_next_enemy_slot   ; grab a free enemy slot (x = index)
    lda #$01
    sta ENEMY_TYPE,x           ; type 1 = bullet
    ...
    sta ENEMY_Y_POS,x          ; goes in the enemy arrays
    sta ENEMY_X_POS,x
```

## Overscan offset (only for drawing)

RAM coords are full-PPU (256×240). `stable_retro` returns a cropped **240×224**
frame (an 8px overscan crop on every side). To place a marker at a RAM `(x, y)`
on that frame, subtract the crop:

```python
xoff = (256 - frame.shape[1]) // 2   # 8 for 240-wide
yoff = (240 - frame.shape[0]) // 2   # 8 for 224-tall
```

`annotate()` does this automatically (and is a no-op offset for a full 256×240
frame). The offset matters **only for pixel-accurate overlays** — for reward math
you compare RAM coords to RAM coords, so no offset is needed.

## Using it (reward / observation)

The core (`scan` + accessors) is **numpy-only**; imaging deps load lazily inside
`annotate`, so training/reward code never pulls in a rendering stack.

```python
from contra import entities

e = entities.scan(ram)          # one pass over RAM
e.player          # (2,)   int16 (x, y)
e.enemies         # (N, 2) live non-bullet enemies
e.enemy_bullets   # (M, 2) incoming gunfire
e.player_bullets  # (K, 2) the player's own bullets (P1)

# ready-made scalar features for reward / observation:
entities.nearest_enemy_bullet_dist(ram)   # px to closest incoming bullet (inf if none)
entities.nearest_enemy_dist(ram)          # px to closest enemy          (inf if none)
```

Reward-design ideas these enable: penalty that rises as an enemy bullet closes on
the player (dodging pressure), reward for a player bullet being near an enemy
(aiming), threat-count in the observation, etc. All are pure functions of `ram`,
so they work identically in search rollouts, replay, and live rollouts.

## Caveats

- The 16 enemy slots are **shared** with real enemies — always split on
  `ENEMY_TYPE == 0x01`; other indices are soldiers/turrets/capsules/etc.
- A few **boss projectiles** use their own enemy types (eye fire-ring, dragon
  fireball) rather than `0x01`, so they won't appear as `enemy_bullets`.
  `ENEMY_VAR_1 $05b8` holds the bullet sub-type (regular / cannonball / indoor).
- Positions are **on-screen** (per-frame). For a level-absolute x (like the trace
  map) add the horizontal scroll (`contra.reward.xscroll`).
- `PLAYER_BULLET_X_POS` is the integer screen x; some weapons keep sub-pixel
  fraction in a separate byte (`PLAYER_BULLET_FS_X $0478`) — negligible for
  locating a bullet.

## Sources

- Disassembly: `reference/nes-contra-us/src/bank7.asm` (`create_enemy_bullet`),
  `src/bank6.asm` (`@find_bullet_slot`), `src/ram.asm` (array addresses).
- `reference/nes-contra-us/docs/lua_scripts/mesen/Show Enemy Positions.lua`
  (the `ENEMY_ROUTINE != 0` liveness test and the address map).
