---
layout: default
title: "Boss Fight Mix (Always Fire Bug)"
parent: Experiments
nav_order: 2
---

# Experiment: Boss Fight Mix
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

---

## Analysis: The "Glued Trigger" Failure

The agent performed **significantly worse** than the baseline, registering a Score of **0** when starting from the beginning of the level. A score of 0 in Contra means the agent **killed zero enemies**.

### Why did this happen?

The issue stems from how the NES hardware registers button presses, combined with a bug in our `ContraWrapper.step` function.

**1. The Action Wrapper Bug**
In `contra_wrapper.py`, the `skip` loop executes the chosen action for 4 consecutive emulator frames. A comment suggested it would hold for 3 frames and release for 1, but the code actually passes the `nes_action` for *all* 4 frames:
```python
# Hold buttons for (skip-1) frames, then 1 no-op release frame  <-- (Comment lied!)
for i in range(self.skip):
    act = nes_action 
    state, _, term, trunc, info = self.env.step(act)
```

**2. The Contra Weapon Mechanic**
With our new "always-fire" action space, the agent always chooses an action where `B=1`. Because `step()` perpetually holds the `B` button down without ever releasing it between actions, it mimics a player physically gluing down the `B` button. 

In Contra, the default rifle is **semi-automatic**. If you hold the `B` button down, the character fires exactly *one* bullet and never fires again until you release and press it again. 

### Conclusion

By attempting to give the agent constant "run-and-gun" abilities, we accidentally created an agent that **fired exactly once at the start of the level and never again**. Without the ability to shoot, the agent was quickly overwhelmed by enemies (Distance ~1500-2000) and completely annihilated during the boss fight (yielding negative rewards).

---

## Next Steps

We need to fix the action wrapper to pulse the `B` button (trigger a release frame) to enable true rapid fire. 

**Planned Fix for `ContraWrapper.step`:**
```python
for i in range(self.skip):
    # Enforce a release on the last frame of the skip so the weapon can fire again
    act = nes_action if i < self.skip - 1 else self._no_op
    state, _, term, trunc, info = self.env.step(act)
```
After fixing the rapid-fire bug, we will retrain the boss fight mix.
