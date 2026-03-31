---
layout: default
title: "9. Fixed Anchor for Save and Load"
parent: Content
nav_order: 9
---

# 9. Fixed Anchor for Save and Load
{: .no_toc }

**Date:** 2026-03-31 · **Type:** Infrastructure / Data Collection
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## The Human-Play Shortcut

Collecting the behavior cloning dataset described in Chapter 8 requires playing through Contra from start to finish without dying. A single death sends the player back to the level start, which in practice makes recording a clean run from scratch exhausting and time-consuming.

The emulator provides a natural solution: **save states**. At every moment of meaningful progress — picking up the Spread gun, killing a miniboss, crossing a difficult screen — the current game state can be serialised to disk. If the player then dies, the emulator rewinds to the most recent save rather than restarting the level from scratch.

This is the **fixed anchor** technique. Each save is a checkpoint; the player is always at most one mistake away from the most recent anchor. The result is a data collection session that essentially guarantees completion. Every recording in the dataset was produced this way: a clean, uninterrupted run that beat the boss without losing a single life, at a fraction of the time a cold-start approach would have required.

## The Insight: Anchors for Monte Carlo Search

Monte Carlo search is not new to this project. Chapters 3 and 4 used it as a validation tool: a way to stress-test the action space before committing to costly RL training, and to confirm that the Level 1 boss was mathematically beatable given the right weapon. It was never expected to beat a full level — with a sparse reward signal, the search is hopelessly inefficient. It's the infinite monkey theorem: given enough time, a monkey randomly pressing keys could type out all of Shakespeare, but we can't afford to wait that long.

Anchor points change that attitude entirely. Each anchor permanently locks in progress up to that point — bit by bit, the search marches toward beating the boss.

The question then becomes: **what counts as a milestone worth anchoring?** For human play the answer was intuitive — the player recognised a good moment and pressed save. For an automated search, we need a formal criterion.

## The RAM Reference

After consulting both Grok and Gemini, a high-quality reference emerged: the [nes-contra-us](https://github.com/vermiceli/nes-contra-us/tree/main) repository, a community reverse-engineering project that documents the Contra NES ROM in annotated 6502 assembly. It names and explains hundreds of RAM addresses — `PLAYER_DEATH_FLAG`, `PLAYER_STATE`, `LEVEL_SCREEN_NUMBER`, `LEVEL_ROUTINE_INDEX`, `WALL_CORE_REMAINING` — and describes exactly how they change with game state.

Honestly, the repo is full of 6502 assembly that I have absolutely no idea how to read. But thanks to knowledgeable LLMs, we can design reward events that are semantically meaningful rather than purely heuristic — turning raw RAM addresses into little named achievements and milestones.

## The Event System

The reward logic and anchor criteria are factored into a dedicated **event system**. Each event is an instance of `ContraEvent`, a small data class that bundles three things:

- **`tag`**: an identifier used in logs and trace files.
- **`trigger_fn`**: a function `(pre_ram, curr_ram) → float` that fires on the transition between two consecutive RAM snapshots and returns the magnitude of the event (zero if it did not occur).
- **`weight`**: a scalar multiplier that converts the trigger magnitude into a reward contribution.

A representative event is `EV_ENEMY_HIT`:

```python
EV_ENEMY_HIT = ContraEvent(
    tag="ENEMY_HIT",
    desc="Sum of HP decrements for non-trivial enemies (pre-HP < 0xf0), excluding "
         "falling rocks (type $13) which respawn endlessly from rock caves on L3.",
    trigger_fn=lambda pre, cur: float(np.sum(
        np.where(
            (pre[ADDR_ENEMY_TYPE:ADDR_ENEMY_TYPE + ADDR_ENEMY_HP_COUNT] != ENEMY_TYPE_FALLING_ROCK) &
            (pre[ADDR_ENEMY_HP:ADDR_ENEMY_HP + ADDR_ENEMY_HP_COUNT] < 0xf0) &
            (cur[ADDR_ENEMY_HP:ADDR_ENEMY_HP + ADDR_ENEMY_HP_COUNT] < 0xf0),
            (pre[ADDR_ENEMY_HP:ADDR_ENEMY_HP + ADDR_ENEMY_HP_COUNT].astype(int) -
             cur[ADDR_ENEMY_HP:ADDR_ENEMY_HP + ADDR_ENEMY_HP_COUNT].astype(int)).clip(min=0),
            0,
        )
    )),
    weight=1.0,
)
```

The trigger examines the 16-slot enemy HP array in one vectorised pass. It masks out two categories of false positives: slots with an initial value of `0xf0` or higher (the sentinel for an empty slot — no enemy loaded here) and falling rocks on Level 3 (enemy type `0x13`), which respawn indefinitely from cave openings and would otherwise be an infinite reward source. For every remaining slot, it accumulates how much HP dropped since the last frame.

The full event catalogue covers the entire game lifecycle:

| Event | Trigger | Weight |
|---|---|---|
| `EV_PUSH_FORWARD` | Horizontal scroll progress | 1/30 per pixel |
| `EV_PUSH_UP` | Vertical scroll progress (L3 waterfall) | 0.5 per pixel |
| `EV_ENEMY_HIT` | Enemy HP decremented | 1.0 per HP point |
| `EV_SPREAD_PICK` | Spread gun collected | +10.0 |
| `EV_SPREAD_LOST` | Spread gun lost | −200.0 |
| `EV_GUN_PICKUP` | Any weapon pickup | +10.0 |
| `EV_CORE_BROKEN` | Indoor wall core destroyed | +10.0 |
| `EV_LEVELUP` | Level completed | +100.0 |
| `EV_GAME_CLEAR` | Final boss defeated | +1000.0 |
| `EV_PLAYER_DIE` | Player died | −5000.0 |

Because the game's level structure mixes side-scrolling, vertical-climbing, and indoor layouts, the active event set is selected per level. `EV_PUSH_FORWARD` fires on the jungle and snow levels but not on the indoor base levels; `EV_PUSH_UP` is exclusive to the Level 3 waterfall; `EV_CORE_BROKEN` applies only to the indoor stages. This is handled by `EVENTS_BY_LEVEL`, a dictionary that maps each 0-indexed level to the appropriate subset of events.

Honestly, I have no idea which RAM address combinations correspond to which game logic — the credit for those goes entirely to Claude. But the event system has been verified: a successful Level 2 run prints a structured event log like this:

```
action   event              desc
    98   GUN_PICKUP         Regular → MachineGun
   162   GUN_POWERUP        MachineGun rapid fire
   945   GUN_PICKUP         MachineGun → Laser
   996   GUN_POWERUP        Laser rapid fire
  1254   LEVEL_TRANSITION   routine=0x08
  1411   LEVELUP            level 1 → 2
```

No need to watch the replay video to know what happened — the log tells the whole story.