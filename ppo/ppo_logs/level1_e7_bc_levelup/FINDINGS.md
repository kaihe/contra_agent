# e7 (level1_e7_bc_levelup) — why e8 adds game-state augmentation

**Date:** 2026-06-19
**Config:** BC warm-start → PPO, `reward_config: boss_more_reward` (levelup 30,
player_die −20, time_out −20), vanilla pixel-only `CnnPolicy` (shared NatureCNN
trunk, stack-3 grayscale 84×84).

## What e7 shows

- **Total loss is ~entirely the value loss.** Recent steps: `0.5·value_loss ≈
  total_loss` (≈10–11), while `policy_gradient_loss ≈ ±0.001` and the entropy term
  ≈ −0.0015. PPO normalizes advantages, so the policy term is tiny by construction;
  the optimized loss is the critic.
- **The critic is globally fine but wins don't stick.** `explained_variance` reaches
  **0.985**, yet `win_rate` peaks **0.16 @ ~12M** and decays to **~0.02** by 56M while
  `mean_reward` (→264) and `mean_progress` (→~1990) keep climbing.
- **Per-anchor eval (sibling run e6, 83M, 15 eps/anchor)** isolates the failure:
  every anchor reaches the boss at **x≈3072** and **dies there ~97%** — 131/135
  deaths, **0 timeouts**. It is *not* dawdling/farming (no timeouts) and *not*
  reachability (boss reached from all 9 anchors). The aggregate `win_rate` is really
  a **boss-survival rate**, and it regressed as PPO optimized the dominant dense
  reward at the expense of boss skill.

## Why pixels are the bottleneck → the e8 motivation

The Level-1 boss has **no on-screen HP bar**, so 3 grayscale frames give the network
the boss sprite and hit-flashes but **not its remaining HP**. The critic therefore
cannot anticipate *when* the terminal `levelup` (+boss-kill) payout fires — the
return near the boss has an **irreducibly unpredictable** component. This is the
representation-level version of the e1/e2 finding ("a big terminal spike breaks the
critic"): not just that the spike is big, but that the *state needed to predict it
is unobservable*. Because the NatureCNN trunk is **shared**, the dominant value
gradient also flows into the policy's visual features, so the unfittable boss
terminal can degrade the representation the actor relies on.

## e8 change

Augment the observation with the **118-dim structured RAM state**
(`contra/game_state.py` → `state_from_ram`): player x/y + velocity, weapon/aim,
**16 enemy slots × (type, x, y, hp)** (the boss occupies an enemy slot, so its HP is
here), and scene flags (`location_type=boss`, `boss_defeated`). Obs becomes
`Dict{image, priv}`; a shared trunk runs NatureCNN on `image` + a (BatchNorm→MLP)
encoder on `priv`, concatenated, feeding both actor and critic. The critic can now
**see boss HP** and predict the terminal payout (lower, learnable value loss), and
the actor gets the same signals to time the boss. One builder (`ppo/model.py`) is
shared by BC pretraining and PPO so the network is identical across both stages.

**Caveat:** the boss may also be a control-precision wall (skip=3 is coarse for
dodging); the state augmentation targets the value-prediction defect, not
necessarily the full win-rate gap.
