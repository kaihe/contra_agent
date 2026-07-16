# Discovered events in Contra (`contra/events.py`)

What the agent can detect happening *between two RAM snapshots* — a hit landed, a
gun picked up, the player died, a level cleared, a boss fight began. Each event is
a pure function of `(pre_ram, curr_ram)`, so it works identically in search
rollouts, replay, and live play.

Every event is a `ContraEvent` with:

- **tag** — UPPER_CASE identifier and log label
- **trigger** — `f(pre, cur) → float` (usually 0/1, sometimes a magnitude)
- **weight** — reward multiplier (`0.0` for pure-narrative events)
- **important** — if True, emitted into the narrative event log (`scan_events`)
- **detail** — optional human-readable annotation (e.g. `"Spread → Regular"`)

Reward for a step is `Σ weight·trigger` over the events active for the level the
player was in *at the start* of the step (`compute_reward`).

## RAM addresses these events read

Sourced from `Contra-Nes/data.json` and `reference/nes-contra-us/src/ram.asm`.

| name | addr | meaning |
|---|---|---|
| `ADDR_LIVES` | `$32` (50) | player lives |
| `ADDR_LEVEL` | `$30` (48) | current level, 0-indexed |
| `ADDR_WEAPON` | `$AA` | low nibble = weapon type; bit4 = rapid-fire flag |
| `ADDR_GUN_ANI` | `$010A` | `0x1F` on the exact frame a gun powerup is collected |
| `ADDR_XSCROLL` | `$64` (100) | horizontal scroll position, big-endian 2 bytes |
| `ADDR_ENEMY_TYPE` | `$0528` | 16-slot enemy type array |
| `ADDR_ENEMY_HP` | `$0578` (1400) | 16-slot enemy HP array |
| `ADDR_INDOOR_CLEARED` | `$37` | INDOOR_SCREEN_CLEARED |
| `ADDR_WALL_CORE_LEFT` | `$86` | WALL_CORE_REMAINING |
| `ADDR_XSCROLL_HI` | `$64` | LEVEL_SCREEN_NUMBER (indoor/vertical screen index) |
| `ADDR_SCROLL_OFF` | `$65` | pixels scrolled within current screen (0→0xEF, wraps at 0xF0=240) |
| `ADDR_PLAYER_ADV` | `$D0` | set while player walks through an indoor door |
| `ADDR_LEVEL_STOP_SCROLL` | `$58` | level length during play; `0xff` once boss-reveal auto-scroll starts |
| `ADDR_LEVEL_ROUTINE` | `$2C` | LEVEL_ROUTINE_INDEX (game-flow phase, see below) |
| `ADDR_PLAYER_STATE` | `$90` | `0x02` = dead (any cause, incl. pit falls) |
| — | `$B4` (180) | PLAYER_DEATH_FLAG; bit0 set on enemy hit |

### `LEVEL_ROUTINE_INDEX` (`$2c`) phases

| value | phase |
|---|---|
| `0x00–0x03` | title / intro screens (loading, lives, score flash, nametable draw) |
| `0x04` | **active gameplay** (inputs have effect — `is_gameplay`) |
| `0x05–0x07` | game-over / continue screens |
| `0x08–0x09` | post-boss transition (end-of-level tune/sequence) |

## The event catalog

### Combat

| tag | fires when | weight | important |
|---|---|---|---|
| `ENEMY_HIT` | sum of HP decrements across *all* live enemy slots | 1.0 | — |
| `REGULAR_ENEMY_HIT` | HP decrements on non-boss enemies only | 1.0 | — |
| `BOSS_HIT` | HP decrements on bosses / minibosses / finite boss-objective parts | 1.0 | — |
| `ENEMY_KILL` | one or more regular enemies destroyed (HP crossed to 0); detail names the types | 0.0 | ✓ |
| `BOSS_STAGE` | player just engaged the end-of-level boss fight | 0.0 | ✓ |

`ENEMY_HIT` is the sum of per-slot `pre_hp − cur_hp` over the 16-slot HP array,
gated on liveness (`hp < 0xf0`). Two objects are always excluded:

- **type `0x01` (Bullet)** — a projectile, has no real HP.
- **type `0x13` (Falling Rock) on L3 only** — respawns endlessly from the
  waterfall's rock caves, so it would be an infinite reward farm. On *every other*
  level `0x13` is a genuine enemy (Wall Turret, Ice Separator, Giant Boss Soldier,
  Mining Cart Generator, Poisonous Insect Gel) and is counted.

`ENEMY_KILL` is the *kill* subset of those decrements: the damage routine runs
`add_enemy_score_set_enemy_routine` (score + destroyed routine) exactly when
`ENEMY_HP` reaches 0 after a hit ([bank7.asm `bullet_collision_logic`](../reference/nes-contra-us/src/bank7.asm#L7013)),
so the kill frame is `0 < pre_hp < 0xf0` and `cur_hp == 0`. It's narrative-only
(weight 0); its detail names the types killed this step with counts, e.g.
`"Soldier ×2, Sniper"`. Query the raw list with `events.enemy_kills(pre, cur)` →
`[(level, enemy_type), …]`. **Bosses and armored cores are excluded**: types like
Wall Plating (`0x0A`) and Core keep `ENEMY_HP = 1` with real HP in `ENEMY_VAR_1`
and reset HP to 1 after each "ting" hit ([bank0.asm](../reference/nes-contra-us/src/bank0.asm#L2776)),
so `cur_hp==0` there is a hit, not a kill — they stay on `BOSS_HIT`.

Boss-vs-regular split is by enemy type per level (`BOSS_ENEMY_TYPES_*`).
`BOSS_STAGE` uses `_boss_scene`: on scrolling levels the game sets
`LEVEL_STOP_SCROLL=$58` to `0xff` when the boss-reveal auto-scroll begins (this
cleanly excludes mid-level minibosses like the L5 snow Tank); indoor levels
(L2/L4) don't auto-scroll, so it falls back to boss-enemy presence, since
Boss Eye / Core only spawn in the final room.

### Weapons

| tag | fires when | weight | important |
|---|---|---|---|
| `GUN_PICKUP` | weapon type changed via pickup animation | 10.0 | ✓ |
| `GUN_POWERUP` | rapid-fire flag gained via pickup (same weapon) | 10.0 | ✓ |
| `SPREAD_PICK` | picked up spread gun (+1) / lost it (−1) | 10.0 | — |
| `SPREAD_LOST` | lost the spread gun | −200.0 | ✓ |

A pickup is detected by `ADDR_GUN_ANI` rising to `0x1F` (`_gun_ani_rising`)
combined with the weapon nibble changing. Spread (type 3) is singled out because
it's the strongest weapon and losing it on death is a large regression.

### Life & progress

| tag | fires when | weight | important |
|---|---|---|---|
| `PLAYER_DIE` | `RAM[180] bit0` set (enemy hit) or `PLAYER_STATE==0x02` (pit) | −5000.0 | ✓ |
| `EXTRA_LIFE` | lives count increased | 0.0 | ✓ |
| `LEVELUP` | level index increased | 100.0 | ✓ |
| `GAME_CLEAR` | level went past 7 (beat final boss) | 1000.0 | ✓ |

`PLAYER_DIE` covers *both* death causes: the death flag at `$B4` bit0 catches
enemy hits reliably over multiple frames, while `PLAYER_STATE==0x02` at `$90`
catches pit falls the flag misses. It's gated so it doesn't re-fire while already
dead.

### Directional progress (level-shape specific)

Different level shapes reward progress differently — only one of these is attached
per level (see level map below).

| tag | fires when | weight | important |
|---|---|---|---|
| `PUSH_FORWARD` | Δ horizontal scroll (side-scroll levels) | 1/30 | — |
| `PUSH_UP` | Δ vertical scroll offset (climbing levels) | 1/2 | — |
| `CORE_BROKEN` | indoor screen cleared (wall core destroyed) | 10.0 | ✓ |
| `PUSH_INSIDE` | player walking through an indoor door (`$d0 ≠ 0`) | 0.5 | — |
| `ROOM_ENTER` | entered next indoor screen (`$64` incremented) | 0.0 | — |

`PUSH_UP` handles the wrap at `0xF0=240` in the scroll offset. `PUSH_INSIDE` is a
per-step trickle that bridges the ~52-step gap between `CORE_BROKEN` and
`ROOM_ENTER` (longer than a rollout), so credit isn't lost across the door walk.

### Game-flow phase (not rewards)

These read `LEVEL_ROUTINE_INDEX` so the caller knows whether to send real actions
or no-ops during transitions.

| tag | fires when | weight | important |
|---|---|---|---|
| `LEVEL_BEGIN` | routine `0x00–0x03` → `0x04` (gameplay starts) | 0.0 | ✓ |
| `LEVEL_TRANSITION` | rising edge into routine `0x08–0x09` (post-boss) | 0.0 | ✓ |
| `TITLE_SCREEN` | routine in `0x00–0x03` (between-level screens) | 0.0 | — |

## Which events apply per level

Progress style is derived from `EVENTS_BY_LEVEL`, the single source of truth
(`level_advance_style`):

| level (1-idx) | shape | advance style | progress events |
|---|---|---|---|
| L1 | jungle | forward | `PUSH_FORWARD` |
| L2 | indoor base | inside | `CORE_BROKEN`, `PUSH_INSIDE`, `ROOM_ENTER` |
| L3 | waterfall climb | up | `PUSH_UP` |
| L4 | indoor base | inside | `CORE_BROKEN`, `PUSH_INSIDE`, `ROOM_ENTER` |
| L5 | snow field | forward | `PUSH_FORWARD` |
| L6 | energy zone | forward | `PUSH_FORWARD` |
| L7 | hangar | forward | `PUSH_FORWARD` |
| L8 | final boss | forward | `PUSH_FORWARD` + `GAME_CLEAR` |

Every level also gets `LEVELS_BASE`: the combat, weapon, life/progress, and
game-flow events above.

## Using it

```python
from contra import events

# reward for one step (level-aware; picks the event list from pre_ram's level)
r = events.compute_reward(pre_ram, curr_ram)

# narrative log entries for one step (only `important` events that fired)
for e in events.scan_events(pre_ram, curr_ram, step):
    print(e["step"], e["tag"], e["detail"])   # e.g. 412 GUN_PICKUP "Regular → Spread"

# helpers
events.is_gameplay(ram)          # True only during active play (routine 0x04)
events.get_level(ram)            # 1-indexed level
events.enemy_type_name(0x14, 2)  # "Dragon" (type 0x14 on L3)
events.enemy_kills(pre, curr)    # [(level, enemy_type), …] killed this step
```

## Sources

- Reward/event definitions: `contra/events.py`.
- RAM addresses: `Contra-Nes/data.json`, `reference/nes-contra-us/src/ram.asm`.
- Enemy type names & boss roles: `reference/nes-contra-us/docs/Enemy Glossary.md`.
- Boss auto-scroll flag: `set_boss_auto_scroll` in `src/bank7.asm`.
