---
layout: default
title: "2. Enhance Boss Fight Training"
parent: Experiments
nav_order: 2
---

# 2. Enhance Boss Fight Training
{: .no_toc }

**Date:** 2026-02-23 · **Model:** `boss_fight_mix_final.zip`
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Changes from Baseline

To allow the agent to "run-and-gun" and increase its DPS during the boss fight, we made two key changes:

1. **Always-Fire Action Space**: Modified `ACTION_TABLE` so every movement action includes the `B` (Fire) button.
2. **Boss State Mixing**: Used a custom state file (`states/Level1_x3048_step921.state`) so that episodes randomly start from either the beginning of Level 1 or right before the Boss.

---

## Results

### Test Runs (Level 1 Start)

We tested the final model starting from the beginning of Level 1.

| Metric         | Run 1  | Run 2  |
|----------------|--------|--------|
| End Reason     | game_over | game_over |
| Score          | **0**  | **0**  |
| Total Reward   | 106.5  | 67.15  |
| Distance       | 1983   | 1455   |

### Test Run (Boss State Start)

We also tested the agent starting directly from the Boss state.

| Metric         | Run 1  |
|----------------|--------|
| End Reason     | game_over |
| Score          | 99     |
| Total Reward   | **-89.8** |
| Distance       | 3072   |

### Gameplay Recording

![Boss Fight Mix gameplay — agent fires once and never again]({{ site.baseurl }}/assets/recordings/boss_fight_mix_final_level1.gif)

---

## Analysis: The "Glued Trigger" Failure

The agent performed **significantly worse** than the baseline, registering a Score of **0** when starting from the beginning of the level. A score of 0 in Contra means the agent **killed zero enemies**.

### Why did this happen?

The issue stems from how the NES hardware registers button presses, combined with a bug in our `ContraWrapper.step` function.

**1. The Action Wrapper Bug**
In `contra_wrapper.py`, the `skip` loop executes the chosen action for 4 consecutive emulator frames. A comment suggested it would hold for 3 frames and release for 1, but the code actually passes the `nes_action` for *all* 4 frames:

**2. The Contra Weapon Mechanic**
With our new "always-fire" action space, the agent always chooses an action where `B=1`. Because `step()` perpetually holds the `B` button down without ever releasing it between actions, it mimics a player physically gluing down the `B` button. 

In Contra, the default rifle is **semi-automatic**. If you hold the `B` button down, the character fires exactly *one* bullet and never fires again until you release and press it again. 

### Conclusion

By attempting to give the agent constant "run-and-gun" abilities, we accidentally created an agent that **fired exactly once at the start of the level and never again**. Without the ability to shoot, the agent was quickly overwhelmed by enemies (Distance ~1500-2000) and completely annihilated during the boss fight (yielding negative rewards).

### Lesson Learned: Test Actions Before Training

This failure cost us an entire training run (~32M timesteps). The bug could have been caught in minutes with a simple action sequence test. Going forward, every time we modify the action table or wrapper behavior, we should **verify the actions actually work** before committing to a long training run.

One approach is a lightweight **beam search** over the action space — brute-force a short sequence of actions and check if the game state progresses as expected (e.g., score increases, enemies die, distance advances). This is reminiscent of Tom Murphy's [learnfun & playfun](https://tom7.org/mario/) project, where:

- **learnfun** identifies valuable RAM addresses correlated with "winning" (e.g., score, distance, lives)
- **playfun** uses beam search to brute-force the action space, optimizing those RAM addresses to play NES games without any neural network at all

While we use PPO instead of beam search for the actual agent, the *testing* phase could benefit from this philosophy: define success metrics from RAM, then verify that our action space can actually move those metrics in the right direction before investing GPU hours into training.
