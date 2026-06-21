# e10 (level1_e10_192) — 192x192 + deeper backbone fails; resolution was not the bottleneck

**Date:** 2026-06-21
**Config:** PPO from scratch (no BC), state-augmented net, **resolution 192**,
**backbone `rescnn`** (4 stride-2 residual stages + global pool, ~1M image params),
`boss_more_reward`, n_steps 1024 / batch 512, stopped ~15.4M steps.

A direct A/B vs **e9** (identical except 84x84 + NatureCNN): only resolution +
backbone changed.

## Result — strictly worse than 84x84 at equal steps

| metric @ ~13–15M | e9 (84, nature) | **e10 (192, rescnn)** |
|---|---|---|
| win_rate | climbing (peaks 0.28 later) | **0.000 (never)** |
| mean_progress (max) | **1707** @13M | **814** @6.7M, last 546 |
| mean_reward (max) | ~200 | **76** @6.7M, last 46 |
| entropy_loss | → −2.2 (and lower later) | stuck at **−2.45** (uniform = −3.04) |
| fps | ~1440 | **394** (~3.7× slower) |
| value_loss / EV | healthy | 0.85 / 0.97 (fine) |

- **Peaks at 6.7M, then regresses** (reward 76→46, progress 814→546) — it went
  backwards, not plateaued.
- Reaches only ~the first third of the level; **never gets near the boss (~x3072)**,
  let alone wins.
- The critic is healthy (EV 0.97) — so this is not a value-fit problem.

## Interpretation

192 + a deeper backbone is **both slower and less sample-efficient**. From scratch,
the larger input and bigger conv stack need far more samples to learn useful features
from sparse reward, so at equal steps it is ~3× behind 84x84 on progress — and at
~3.7× lower fps each step costs more, compounding the loss. The near-flat entropy
shows the policy barely committed to actions in 15M steps.

**The "84x84 / model too weak" hypothesis is not supported.** Bigger input and more
capacity made things worse, not better — so input resolution and backbone size are
**not** what gates winning. Combined with e9 (state augmentation fixed the critic but
not the win), the persistent bottleneck lies elsewhere:

- **Optimization / prior:** RL-from-scratch on this task is the common weakness across
  e9/e10; the runs that progressed furthest used a **BC warm-start**. A bigger net only
  amplifies the cold-start cost.
- **Reward shaping:** farm-over-finish pressure (ep_len grows, wins don't) persists
  regardless of architecture.

## Takeaway

Drop the 192/deeper-backbone direction. Return to **84x84 + NatureCNN** and attack the
real levers: **BC warm-start** (don't train the boss policy from scratch) and the
**reward balance / boss-execution** problem — not perception capacity.
