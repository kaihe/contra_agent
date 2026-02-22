---
layout: default
title: "Baseline"
parent: Experiments
nav_order: 1
---

# Experiment: Baseline
{: .no_toc }

**Date:** 2026-02-22 · **Model:** `no_fire_final.zip`
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Training Configuration

| Parameter          | Value                  |
|--------------------|------------------------|
| Algorithm          | PPO (Stable-Baselines3)|
| Policy             | CnnPolicy             |
| Total Timesteps    | 32,000,000             |
| Parallel Envs      | 32                     |
| n_steps (per env)  | 2,048                  |
| Batch Size         | 2,048                  |
| n_epochs           | 10                     |
| Gamma              | 0.99                   |
| Learning Rate      | 1e-4 (constant)        |
| Clip Range         | 0.2 → 0.05 (linear)   |
| Entropy Coef       | 0.1 → 0.005 (linear)  |
| Frame Skip         | 4                      |
| Frame Stack        | 4                      |
| Observation        | Grayscale 84×84×4      |
| Action Space       | Discrete(7)            |
| Game / State       | Contra-Nes / Level1    |

## Action Space

The agent uses a reduced 7-action discrete space. **Fire and movement are mutually exclusive** — the agent can only shoot while standing still. It cannot fire while moving or jumping.

| ID | Action           | Buttons Pressed | Description          |
|----|------------------|-----------------|----------------------|
| 0  | Fire             | B               | Stand still & shoot  |
| 1  | Left             | LEFT            | Move left (no fire)  |
| 2  | Right            | RIGHT           | Move right (no fire) |
| 3  | Up               | UP              | Aim up (no fire)     |
| 4  | Down             | DOWN            | Crouch (no fire)     |
| 5  | Left + Jump      | LEFT + A        | Jump left (no fire)  |
| 6  | Right + Jump     | RIGHT + A       | Jump right (no fire) |

{: .warning }
> **Key limitation:** The agent must choose between firing and moving. This means it cannot run-and-gun — a fundamental tactic in Contra. This design choice is reflected in the model name **"no_fire"** (i.e., no fire while moving).

---

## Reward Structure

The reward is computed per agent step (every 4 emulator frames):

| Component              | Formula / Value                           | Notes |
|------------------------|-------------------------------------------|-------|
| **Distance reward**    | `clamp(x_delta, 0, 3) × 0.1`            | Encourages rightward progress; ~300 total over a full level |
| **Score reward**       | `max(score_delta, 0)`                    | Raw game score increase (killing enemies, picking up items) |
| **Idle penalty**       | `-0.05` per step after 20 idle steps     | Punishes the agent for not pushing max distance forward |
| **Death penalty**      | `-20` per life lost                      | Discourages dying |
| **Timeout penalty**    | `-50` if episode reaches 4,000 steps     | Prevents infinite stalling |
| **Win bonus**          | `+100` if game ends with lives > 0       | Rewards beating the level |

---

## Results

### Test Runs (1 Episode Each)

| Metric         | Run 1  | Run 2  |
|----------------|--------|--------|
| End Reason     | game_over | game_over |
| Score          | 98     | 95     |
| Total Reward   | 272.4  | 279.4  |
| Distance Reward| 216.9  | 234.9  |
| Score Reward   | 98.0   | 95.0   |
| Steps          | 915    | 1,130  |
| Max Distance   | 2,838  | 3,072  |

### Gameplay Recording

![Baseline gameplay]({{ site.baseurl }}/assets/recordings/no_fire_final.gif)

---

## Analysis

### Early Game — Agent Performs Well

The agent has learned solid navigation: it moves right, jumps over obstacles, and shoots enemies effectively. It accumulates score and distance reward at a healthy rate.

### Boss Fight — Agent Gets Stuck and Dies

{: .important }
> The agent consistently fails to defeat the Level 1 boss. The core issue is **weapon selection**: the agent blindly picks up the **Laser Rifle (L)** weapon capsule that appears before the boss fight.

The Laser Rifle is a "trap weapon" — it fires a single slow, piercing beam that is extremely ineffective against the boss's multiple weak points.

### Comparison with Reference Implementation

Compared with the PPO agent from [vietnh1009/Contra-PPO-pytorch](https://github.com/vietnh1009/Contra-PPO-pytorch):

| Aspect               | Reference Agent                  | Our Baseline Agent               |
|-----------------------|----------------------------------|----------------------------------|
| Boss fight            | ✅ Clears the boss               | ❌ Dies at the boss              |
| Weapon handling       | **Avoids the Laser Rifle** capsule; keeps the Spread Gun (S) | **Blindly picks up** the Laser Rifle |
| Why it matters        | Spread Gun fires 5 bullets in a fan — ideal for the boss | Laser Rifle fires 1 slow beam — near useless against the boss |

---

## Root Cause Analysis

The fundamental problem is that **the agent cannot plan for the future**.

1. **Short-term incentive vs. long-term consequence:** Picking up any weapon capsule gives a small **score reward** (positive reinforcement). The agent is rewarded for walking into the capsule. However, swapping from the powerful Spread Gun to the weak Laser Rifle makes the boss fight nearly unwinnable — a consequence the agent never learns to associate with the pickup action.

2. **No "don't pick up" action:** In Contra, weapon pickup is automatic on contact. The only way to *avoid* a weapon is to jump over the capsule or not move toward it. This requires the agent to understand that a specific spatial region should be avoided — a far more abstract concept than reacting to immediate threats.

3. **Temporal credit assignment:** The negative consequence (failing the boss) occurs hundreds of steps after the weapon pickup. Standard PPO with γ=0.99 struggles to propagate this long-horizon signal back to the critical pickup moment.

---

## TODO: Next Experiment

**Goal:** Beat the Level 1 boss.
{: .fs-5 }

Two changes combined in the next training run:

1. **Constant-fire action space** — Add `B=1` (fire) to all movement actions so the agent always fires while moving. This enables run-and-gun, which is essential for Contra gameplay and should dramatically increase DPS during the boss fight.

2. **Boss-area save state** — Create a new save state near the boss fight and train with `--state Level1 Level1_Boss` so the agent gets ~50% of episodes starting near the boss. This addresses the lack of boss-fight training data in the baseline.

