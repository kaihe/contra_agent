# Shrinking the win reward fixes the value function — e2_no_levelup vs e1_less_hp

**Date:** 2026-06-14
**Compared:** `level1_e1_less_hp` (levelup reward = 100) vs `level1_e2_no_levelup` (levelup reward = 1).
Both: same 9 anchors, 64M steps, identical PPO hyperparameters. The **only** config
differences are the two event-reward weights:

| reward weight | e1_less_hp | e2_no_levelup |
|---|---|---|
| `levelup` (win) | **100.0** | **1.0** |
| `spread_pick` | 20.0 | 2.0 |
| (all others) | identical | identical |

## TL;DR — the assumption holds

Cutting the win reward from +100 to +1 made training **smoother, faster past the mid-game, and reach a much higher ceiling**:

| metric (final) | e1 (levelup=100) | e2 (levelup=1) |
|---|---|---|
| **win_rate** | 0.53 (max 0.67) | **0.82 (max 0.90)** |
| **explained_variance** (value fit) | 0.84, *collapsed to ~0.55 mid-run* | **0.995, never collapses** |
| **value_loss** | 69 (*peaked ~160*) | **4.3** |
| value_loss jitter (step-to-step std) | 12.7 | **8.7** |
| explained_variance jitter | 0.041 | **0.023** |

See `compare_e1_e2.png` (win_rate / explained_variance / value_loss, both runs).

## The mechanism: the +100 spike breaks the critic exactly when wins begin

The three curves line up into one causal story:

1. **First ~20M steps:** the policy rarely wins, so the +100 reward almost never fires.
   Both runs are nearly identical — win_rate ~0, explained_variance ~0.9, value_loss ~35.

2. **~20–30M, e1 only:** as the policy starts reaching the boss, **+100 wins start appearing
   in the returns**. The critic now has to fit a return distribution with rare, huge,
   terminal spikes:
   - `explained_variance` **collapses 0.90 → ~0.55** (the value head can no longer predict returns),
   - `value_loss` **explodes ~35 → ~120–160**.

   This degradation is *synchronized with win onset* — the exact moment the +100 enters the
   data — not with the start of training.

3. **Result, e1:** with a confused critic, advantage estimates are noisy, so policy
   improvement stalls and **win_rate plateaus at ~0.53**.

4. **e2 (levelup=1):** the return distribution stays compact even once wins are frequent.
   The critic dips only slightly (~0.78), then climbs **smoothly to 0.995** while
   `value_loss` falls to ~5. Clean advantages → the policy keeps improving →
   **win_rate climbs to 0.82**.

Why the boss region specifically? With `gamma=0.99`, a +100 terminal reward forces the
value function to learn a steep ramp up to ~+100 near the boss (a high-curvature target),
while still predicting near-zero far away. That sharp, sparse, heavy-tailed target is what
PPO's MSE value loss struggles to fit. At `levelup=1` the same ramp is 100× smaller and
sits in the same scale as the dense `progress` (0.1/px) and `enemy_hp` rewards the critic
already fits well.

## On "faster"

It's less "faster everywhere" than "doesn't stall." Both cross win_rate 0.3 at ~31–35M.
But e2 crosses 0.5 a bit sooner (45M vs 48M) and, crucially, **keeps going** (0.8 at ~61M)
whereas e1 flattens. The win is the removed ceiling, driven by the healthy critic.

## Caveat (honest accounting)

e2 changed **two** weights: `levelup` 100→1 *and* `spread_pick` 20→2. So this isn't a
perfectly isolated ablation. However, the evidence points squarely at `levelup`:
- `spread_pick` fires once, early, in essentially every episode from the start — if its
  magnitude were the problem, the critic would have struggled from step 0, not at 20M.
- The critic collapse is time-locked to **win onset**, and `levelup` (100) is 5× the next
  largest reward and the only large *terminal* one.

A clean follow-up would be an ablation that drops only `levelup` (keeping `spread_pick=20`),
but the timing argument already isolates the win spike as the cause.

## Takeaways

- Keep terminal/event reward magnitudes in the **same scale** as the dense shaping rewards;
  a +100 sparse spike is a value-function hazard, not a stronger learning signal.
- Watch `explained_variance` and `value_loss` as early-warning signals: e1's mid-run EV
  collapse predicted the win_rate plateau before it happened.
- `levelup=1` (the new `ppo/level1_win.yaml` recipe) is the better default; the win signal
  still ranks winning episodes highest without destabilizing the critic.

## Reproduce

```
# curves + plot
python - <<'PY'  # (see the comparison snippet used for compare_e1_e2.png)
PY
```
Artifacts in this dir: `compare_e1_e2.png`, `level1_win.yaml` (the e2 config), `final.zip`.
Note: `contra/mean_delta_x` in these logs predates the high-water-mark fix, so its negative
late values are the levelup xscroll-reset artifact, not regressions — ignore it here.
