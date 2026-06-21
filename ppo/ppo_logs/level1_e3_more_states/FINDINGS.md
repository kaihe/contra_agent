# e3_more_states does not learn faster than e2_no_levelup

**Date:** 2026-06-15
**Compared:**
- `tmp/ppo/checkpoints/level1_e2_no_levelup`
- `tmp/ppo/checkpoints/level1_e3_more_states`

Both runs train for ~64M PPO environment steps with the same reward scale and PPO
hyperparameters. The intended ablation is the start-state distribution:

| run | anchors | source |
|---|---:|---|
| e2_no_levelup | 9 | one winning trace, `win_level1_202603301145_*` |
| e3_more_states | 45 | `ppo/states/level1_*.state`, 5 traces x 9 anchors |

The e3 config really does expand to 45 anchors. The copied config uses
`states: ppo/states/level1_*.state`, and the live glob currently matches 45 files.

## TL;DR

The "more anchors should learn faster" assumption is the wrong expectation for
this setup. e3 has the same 64M-step budget as e2 but spreads it over 45 anchors
instead of 9, including several hard states where e2 had near-zero win rate.
That means e3 spends compute learning hard situations, not simply accelerating
the original 9-anchor training curve.

On the aggregate training win-rate chart, e3 is smoother but not faster, and it
does not reach e2's final training win-rate ceiling:

| metric | e2_no_levelup | e3_more_states |
|---|---:|---:|
| final `contra/win_rate` | **0.82** | 0.68 |


Most important correction: the low-win e2 anchors from the 45-state set are no
longer dead under e3:

| anchor | e2 eval win rate | e3 eval win rate |
|---|---:|---:|
| `level1_202603301703_x2779.state` | 5% | **80%** |
| `level1_202604021858_x1824.state` | 0% | **65%** |
| `level1_202604081140_x2419.state` | 0% | **75%** |

That is where the extra training compute went.

## Why the plots look less variant

The lower visual variance is partly real, but it is not the same as better
learning.

Late-run window, last ~5M steps:

| metric | e2 mean | e2 std | e3 mean | e3 std |
|---|---:|---:|---:|---:|
| `contra/win_rate` | **0.787** | 0.0506 | 0.674 | **0.0422** |
| `contra/mean_reward` | **226.4** | 12.14 | 223.5 | **10.24** |
| `rollout/ep_rew_mean` | **226.4** | 11.78 | 223.9 | **10.37** |
| `rollout/ep_len_mean` | 276.2 | 14.24 | 269.3 | **12.58** |
| `contra/mean_delta_x` | -839.2 | 151.5 | **1658.0** | **90.5** |
| `contra/mean_enemy_hp_cost` | **15.79** | 0.923 | 15.63 | **0.766** |

That supports the eyeball read: e3's aggregate episode curves are smoother.
But the PPO training diagnostics do **not** get smoother or healthier:

| metric | e2 late mean | e2 late std | e3 late mean | e3 late std |
|---|---:|---:|---:|---:|
| `train/explained_variance` | **0.984** | **0.0072** | 0.979 | 0.0083 |
| `train/value_loss` | **12.33** | **5.51** | 15.08 | 5.97 |
| `train/approx_kl` | **0.0075** | **0.0023** | 0.0118 | 0.0043 |
| `train/clip_fraction` | **0.235** | **0.0458** | 0.297 | 0.0512 |

The important distinction: e3 averages across a broader start distribution, so
some per-anchor spikes get washed out in the displayed aggregate. But the update
problem is harder: the policy is asked to improve from 45 distinct snapshots
instead of 9, and the late optimizer metrics show more clipping/KL pressure and
a weaker critic.

## Full 45-anchor e3 evaluation

I had not tested all 45 anchors before the first writeup; only the three
previously dead e2 anchors had been spot-checked. I then evaluated
`level1_e3_more_states/final.zip` on all 45 `ppo/states/level1_*.state` anchors
with 20 stochastic episodes per anchor.

Overall: **605/900 wins = 67.2%**. There were 295 deaths, 0 game-overs, and 0
timeouts.

Per-series summary:

| series | wins | win rate |
|---|---:|---:|
| `202603301145` | 109/180 | 60.6% |
| `202603301703` | 130/180 | 72.2% |
| `202604021858` | 121/180 | 67.2% |
| `202604081140` | 131/180 | 72.8% |
| `202604081539` | 114/180 | 63.3% |

Per-anchor table:

| anchor | win/20 | win rate | death | dx mean | reward |
|---|---:|---:|---:|---:|---:|
| `202603301145_x0000` | 12/20 | 60% | 8 | 2963 | 362.5 |
| `202603301145_x0313` | 8/20 | 40% | 12 | 2550 | 302.9 |
| `202603301145_x0645` | 14/20 | 70% | 6 | 2397 | 303.4 |
| `202603301145_x1021` | 12/20 | 60% | 8 | 1978 | 259.4 |
| `202603301145_x1343` | 12/20 | 60% | 8 | 1728 | 236.6 |
| `202603301145_x1695` | 17/20 | 85% | 3 | 1377 | 203.4 |
| `202603301145_x2078` | 13/20 | 65% | 7 | 965 | 148.5 |
| `202603301145_x2411` | 10/20 | 50% | 10 | 660 | 96.1 |
| `202603301145_x2721` | 11/20 | 55% | 9 | 351 | 65.3 |
| `202603301703_x0000` | 13/20 | 65% | 7 | 3059 | 372.6 |
| `202603301703_x0310` | 13/20 | 65% | 7 | 2692 | 334.0 |
| `202603301703_x0705` | 16/20 | 80% | 4 | 2272 | 291.8 |
| `202603301703_x1068` | 16/20 | 80% | 4 | 2003 | 269.4 |
| `202603301703_x1450` | 18/20 | 90% | 2 | 1620 | 228.1 |
| `202603301703_x1766` | 15/20 | 75% | 5 | 1305 | 193.4 |
| `202603301703_x2116` | 13/20 | 65% | 7 | 926 | 145.1 |
| `202603301703_x2395` | 11/20 | 55% | 9 | 649 | 105.9 |
| `202603301703_x2779` | 15/20 | 75% | 5 | 293 | 81.6 |
| `202604021858_x0000` | 10/20 | 50% | 10 | 2923 | 347.8 |
| `202604021858_x0303` | 14/20 | 70% | 6 | 2702 | 336.8 |
| `202604021858_x0667` | 13/20 | 65% | 7 | 2404 | 308.4 |
| `202604021858_x1083` | 17/20 | 85% | 3 | 1989 | 266.9 |
| `202604021858_x1437` | 14/20 | 70% | 6 | 1635 | 227.1 |
| `202604021858_x1824` | 13/20 | 65% | 7 | 1245 | 179.0 |
| `202604021858_x2065` | 13/20 | 65% | 7 | 1007 | 154.9 |
| `202604021858_x2369` | 13/20 | 65% | 7 | 702 | 119.5 |
| `202604021858_x2704` | 14/20 | 70% | 6 | 367 | 85.7 |
| `202604081140_x0000` | 8/20 | 40% | 12 | 3019 | 365.0 |
| `202604081140_x0252` | 14/20 | 70% | 6 | 2820 | 354.4 |
| `202604081140_x0581` | 13/20 | 65% | 7 | 2491 | 315.8 |
| `202604081140_x0942` | 17/20 | 85% | 3 | 2130 | 284.3 |
| `202604081140_x1316` | 16/20 | 80% | 4 | 1756 | 242.1 |
| `202604081140_x1593` | 13/20 | 65% | 7 | 1420 | 193.9 |
| `202604081140_x2009` | 18/20 | 90% | 2 | 1063 | 168.6 |
| `202604081140_x2419` | 16/20 | 80% | 4 | 652 | 121.2 |
| `202604081140_x2756` | 16/20 | 80% | 4 | 316 | 85.0 |
| `202604081539_x0000` | 12/20 | 60% | 8 | 2922 | 354.5 |
| `202604081539_x0301` | 9/20 | 45% | 11 | 2375 | 289.8 |
| `202604081539_x0663` | 14/20 | 70% | 6 | 2341 | 300.5 |
| `202604081539_x1027` | 13/20 | 65% | 7 | 1696 | 221.7 |
| `202604081539_x1387` | 12/20 | 60% | 8 | 1610 | 215.3 |
| `202604081539_x1710` | 12/20 | 60% | 8 | 1279 | 178.5 |
| `202604081539_x2057` | 11/20 | 55% | 9 | 1014 | 153.7 |
| `202604081539_x2426` | 14/20 | 70% | 6 | 644 | 107.7 |
| `202604081539_x2732` | 17/20 | 85% | 3 | 339 | 90.0 |
