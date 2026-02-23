---
layout: default
title: "5. Pick Better Gun"
parent: Experiments
nav_order: 5
---

# 5. Pick Better Gun
{: .no_toc }

**Date:** 2026-02-23 · **Type:** Action Space Investigation
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Motivation

The [baseline experiment](baseline) revealed a fundamental flaw in our action space design: **fire and movement were mutually exclusive**. The agent had to choose between shooting and moving on each step, forcing it to arrange fire and movement in a sequence. In contrast, a human player uses both hands simultaneously — pressing the D-pad and the B button at the same time — making it possible to combine movement and fire at any moment. This severely limited the agent's DPS and made boss fights nearly unwinnable.

To fix this, we introduced an "always-fire" action space (`B=1` on all actions) so the agent could run-and-gun like a human player.

However, our first attempt ([Boss Fight Mix](boss_fight_mix)) was a catastrophic failure: the agent scored **0** — it didn't kill a single enemy. We discovered that holding the `B` button down continuously without ever releasing it caused Contra's semi-automatic default rifle to fire exactly **one bullet** at the start of the game and never again. The agent was effectively unarmed for the entire episode.

We fixed this by adding a **B-release frame** on the last frame of each 4-frame skip, ensuring the weapon can re-trigger on the next action. Before retraining, we needed to verify how each weapon behaves under this new rapid-fire mechanism — especially the Laser Rifle, which has unique firing mechanics.

---

## Weapons in Contra (Level 1)

| Weapon | Symbol | Fire Pattern | Pickup |
|--------|--------|-------------|--------|
| **Default Rifle** | — | Single bullet per press, semi-automatic | Start weapon |
| **Spread Gun** | S | 5 bullets in a fan pattern | Red capsule |
| **Laser Rifle** | L | Single piercing beam, cancels on re-fire | Red capsule |

---

## Experiment: Constant Fire with Each Weapon

We used `run_sequence.py` to execute a repeating `[JF, F]` (Jump+Fire, Fire) pattern for 60 cycles from three different save states, each with a different weapon equipped.

### Default Rifle (Normal)

![Default Rifle]({{ site.baseurl }}/assets/recordings/seq_JF_F_60_normal.gif)

**Observation:** The default rifle fires rapidly. Each B-release triggers a new shot. The agent puts out a steady stream of bullets while alternating between jumping and standing. **Good DPS.**

### Spread Gun (S)

![Spread Gun]({{ site.baseurl }}/assets/recordings/seq_JF_F_60_spread.gif)

**Observation:** The Spread Gun fires 5 bullets in a fan pattern on each trigger. With rapid fire, the screen fills with projectiles. **Excellent DPS — best weapon for the boss fight.**

### Laser Rifle (L)

![Laser Rifle]({{ site.baseurl }}/assets/recordings/seq_JF_F_60_laser.gif)

{: .warning }
> **Critical finding:** The Laser Rifle fires a single beam that travels across the screen. However, **firing again cancels the previous beam** and replaces it with a new one starting from the player's position. With constant rapid fire, the laser beam is cancelled every 4 frames and never travels far enough to hit anything. The weapon becomes **nearly useless** under constant fire.

---

## Impact on Training

| Weapon | Without Constant Fire (Baseline) | With Constant Fire (New) |
|--------|----------------------------------|--------------------------|
| Default Rifle | Semi-auto, decent | **Rapid fire, great** ✅ |
| Spread Gun | Semi-auto, good | **Rapid fire, devastating** ✅ |
| Laser Rifle | Slow but reaches far | **Self-cancelling, useless** ❌ |

### Why This Is Actually Good

We already know from the reference implementation [vietnh1009/Contra-PPO-pytorch](https://github.com/vietnh1009/Contra-PPO-pytorch) that **avoiding the Laser Rifle leads to victory**. Their trained agent successfully dodges the laser capsule and keeps the Spread Gun through the boss fight:

![Reference agent avoiding laser and beating the boss]({{ site.baseurl }}/assets/video-1.gif)

The laser becoming even worse under our constant-fire mechanism **strengthens the avoidance signal**:

1. Agent picks up laser → fires rapidly → laser cancels itself → zero DPS → agent dies quickly
2. The death penalty (-20 per life) accumulates faster
3. Over training, the agent learns a stronger association: **"laser pickup = death"**

The agent should naturally learn to avoid the Laser Rifle capsule, since the reward consequence of picking it up is now even more negative than in the baseline.

---

## Training Plan: Run & Gun

Based on these findings, we will train with:

### Action Space — Discrete(8), Always-Fire

| ID | Action | Buttons |
|----|--------|---------|
| 0 | Fire | B |
| 1 | Left+Fire | B+LEFT |
| 2 | Right+Fire | B+RIGHT |
| 3 | Up+Fire | B+UP |
| 4 | Down+Fire | B+DOWN |
| 5 | Left+Jump+Fire | B+LEFT+A |
| 6 | Right+Jump+Fire | B+RIGHT+A |
| 7 | Jump+Fire | B+A |

### B-Release Fix

On the last frame of each 4-frame skip, the B button is released (`act[0] = 0`). This ensures the semi-automatic weapons fire on every action step instead of getting stuck on the first shot.

### Multi-State Training — 4 States

| State File | Description | Purpose |
|------------|-------------|---------|
| `Level1` (built-in) | Level 1 start | Full level traversal |
| `Level1_x2113_step595.state` | Mid-level with Spread Gun | Learn to play with the best weapon |
| `Level1_x2728_step779.state` | Pre-boss with Laser Rifle | Learn to survive/die quickly with trap weapon |
| `Level1_x3048_step921.state` | Boss fight entrance | Boss fight practice |

---

## Expected Results

With the always-fire action space, B-release fix, and multi-state training, we expect the agent to:

1. **Beat the Level 1 boss** — The combination of rapid fire and dedicated boss-area training data should give the agent enough DPS and experience to clear the boss fight.

2. **Intentionally avoid the Laser Rifle** — Since the laser is now even more punishing under constant fire (self-cancelling beam = zero DPS), the agent should learn to jump over or walk around the laser capsule to keep its current weapon.

3. **Prefer the Spread Gun** — Episodes starting from the Spread Gun state should show significantly higher rewards and survival rates, reinforcing the agent's preference for keeping this weapon through the boss fight.