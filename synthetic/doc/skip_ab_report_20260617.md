# mc_search skip A/B test: skip 3 vs skip 4

Date: 2026-06-17

## Question

Which frame skip is more efficient for `synthetic/mc_search.py`: `skip = 3` or
`skip = 4`?

## Setup

The checked-out action config was not edited during the test. Instead, the test
harness patched both runtime skip globals per case:

- `synthetic.mc_search.SKIP`
- `contra.replay.SKIP`

This matters because `mc_search` calls `contra.replay.step_env()`, and that
function reads `contra.replay.SKIP`.

Common search budget:

- levels: 1, 2, 3, 4
- skips: 3 and 4
- rollouts per step: 64
- rollout length: 48 actions
- workers: 8
- max time: 45 seconds per case
- max actions: 1600
- goal: `level_up`
- reward configs: `level{n}_stable`

Raw output:

- `synthetic/doc/skip_ab_raw_20260617.jsonl`

## Results

No run reached a level-up within the short 45 second cap, so this is a
progress-efficiency test rather than a win-rate test.

| level | skip | win | wall_s | committed actions | sampled actions | reward | reward/s | reward/1k sampled |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | no | 45.2 | 1507 | 191508 | 414.2 | 9.17 | 2.16 |
| 1 | 4 | no | 45.7 | 1100 | 142014 | 378.2 | 8.28 | 2.66 |
| 2 | 3 | no | 46.1 | 941 | 181344 | 133.0 | 2.89 | 0.73 |
| 2 | 4 | no | 45.3 | 1294 | 136546 | 325.0 | 7.18 | 2.38 |
| 3 | 3 | no | 34.9 | 1601 | 133741 | 791.0 | 22.65 | 5.91 |
| 3 | 4 | no | 45.7 | 1291 | 131404 | 1204.0 | 26.35 | 9.16 |
| 4 | 3 | no | 45.4 | 970 | 174379 | 151.0 | 3.32 | 0.87 |
| 4 | 4 | no | 45.5 | 949 | 137528 | 171.0 | 3.76 | 1.24 |

## Readout

`skip = 4` looks more efficient overall in this short-budget test.

Evidence:

- `skip = 4` had better reward per 1k sampled actions on all four levels.
- `skip = 4` had higher raw reward on levels 2, 3, and 4.
- `skip = 4` sampled fewer rollout actions in every paired case, because each
  action advances more emulator frames.
- Level 1 is the one raw-reward exception: `skip = 3` reached reward `414.2`
  vs `378.2` for `skip = 4`. But `skip = 4` did that with about 26% fewer
  sampled actions, so it was still more sample-efficient by reward per sampled
  action.

## Interpretation

For `mc_search`, `skip = 4` appears to be the better default if the objective is
search efficiency per sampled rollout action.

The likely reason is simple: at `skip = 3`, the search spends more decisions on
small motion increments. That can give finer control, but it also increases the
number of committed decisions needed to cross the same real game distance. At
`skip = 4`, each sampled action covers more game time while still being short
enough to preserve useful control. In these tests that improved reward density,
especially on the indoor and vertical levels.

## Caveats

This is not a final win-rate benchmark:

- only one seed per level/skip pair
- short 45 second cap
- no successful level-up in any case
- multiprocessing workers add some stochasticity because worker seeds depend on
  process IDs
- Level 3 `skip = 3` hit the action cap before the time cap, so max-actions is
  part of that result

## Recommendation

Use `skip = 4` for the next `mc_search` iteration.

Before treating it as final, run a longer confirmation benchmark:

- 3 seeds per level
- levels 1-4
- `max_time = 180` or `300`
- same `rollouts`, `rollout_len`, `workers`, and `max_actions`
- compare win rate first, then median time-to-win, sampled actions, and trace
  length
