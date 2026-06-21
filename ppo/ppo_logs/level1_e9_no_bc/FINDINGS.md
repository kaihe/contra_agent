# e9 (level1_e9_no_bc) — state augmentation fixed the critic, not the win

**Date:** 2026-06-20
**Config:** PPO from scratch (no BC warm-start), `reward_config: boss_more_reward`,
**state-augmented** network (Dict obs `{image 3×84×84, priv 118}`, `ImageStateExtractor`,
640-dim features, 1.8M params), stack-3, skip-3, max_episode_steps 5000, 128M budget.

## Result — still doesn't win

| metric | value |
|---|---|
| win_rate | peaks **0.28 @ 46M**, decays to **0.02** by 99M |
| mean_reward | max 206 @83M, last 147 |
| mean_progress | max 1707 **@13M (early)**, last 1048 |
| ep_len_mean | **196 → max 3587 @85M**, last ~2470 |
| value_loss | spike **131 @46M**, settles ~7.8 |
| explained_variance | recovers to **0.966** (max 1.000) |
| entropy_loss | −3.044 (random init) → −2.23 |

## Two facts that reframe the problem

1. **The critic is now healthy.** With `priv` (incl. boss HP) the value head recovers
   to EV ≈ 0.966 and value_loss settles to ~8 after the win-onset spike (131 @46M,
   time-locked to the win peak, exactly the e1/e2 mechanism — but this time it recovers
   *because* the state makes the terminal predictable). **So the value-prediction defect
   that motivated e8 is resolved — yet win_rate still collapses.** Value prediction was
   not the binding constraint on winning.

2. **The agent now survives long without finishing.** ep_len blows up from ~600 (e7,
   died at the boss) to **~2500–3500** here, while win_rate falls. It learned to use the
   state to *dodge and stay alive* near the boss, but not to *land the kill*. No-BC even
   explored a better win policy early (0.28 vs e7's 0.16) before regressing.

## Live hypotheses (next experiments)

- **Policy capacity / input too weak (primary).** NatureCNN (~1.8M) on 84×84 grayscale
  + skip-3 may lack the spatial precision and reaction granularity to execute the boss
  fight (read bullet patterns, aim the core, frame-tight dodges). Survives-but-can't-kill
  is consistent with a capability ceiling, not a value-fit problem. → try a larger
  backbone and higher input resolution (168/192), and/or smaller frame-skip for boss
  control.
- **Reward still rewards farming over finishing (secondary).** ep_len↑ while win↓ means
  continued play still out-earns the marginal win. → make finishing dominate (opportunity
  cost via potential-based shaping, or stronger terminal vs dense balance).

**Takeaway:** state augmentation did its job (critic), so it is *not* sufficient for the
win. The bottleneck has moved to the policy's ability/incentive to *execute* the boss
kill — capacity + resolution (+ skip) are the next levers to test.
