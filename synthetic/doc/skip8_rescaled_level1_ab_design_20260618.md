# Level 1 mc_search A/B: frame-matched skip comparison

Date: 2026-06-18

## Question

PPO suggests `skip = 8` is much more efficient than the older `skip = 3` run,
but `mc_search` has favored lower skip values in search-to-win tests. The old
search comparison kept `rollout_len` and `max_rewind` fixed in action units,
which changes the actual number of NES frames searched at each skip.

This experiment tests whether `skip = 8` still looks worse for `mc_search` after
the lookahead and rewind windows are normalized to the same game-frame horizon.

## Frame math

The baseline search window is the current `skip = 3` default:

| setting | action units | skip | game frames |
|---|---:|---:|---:|
| rollout lookahead | 48 | 3 | 144 |
| max rewind | 30 | 3 | 90 |
| max committed actions | 8000 | 3 | 24000 |

For a frame-matched condition, compute:

- `rollout_len = ceil(144 / skip)`
- `max_rewind = ceil(90 / skip)`
- `max_actions = ceil(24000 / skip)`

That gives:

| skip | rollout_len | rollout frames | max_rewind | rewind frames | max_actions | action-cap frames |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 48 | 144 | 30 | 90 | 8000 | 24000 |
| 4 | 36 | 144 | 23 | 92 | 6000 | 24000 |
| 8 | 18 | 144 | 12 | 96 | 3000 | 24000 |

The small upward rounding is intentional: the higher-skip variants should not
lose simply because their lookahead is slightly shorter.

## Experiment matrix

Run level 1 only, `goal = level_up`, `reward_config = level1_stable`, workers
fixed to the same machine count, and use at least 5 seeds per condition.

Primary A/B:

| condition | skip | rollout_len | max_rewind | max_actions | purpose |
|---|---:|---:|---:|---:|---|
| A: baseline skip 3 | 3 | 48 | 30 | 8000 | Current search reference |
| B: frame-matched skip 8 | 8 | 18 | 12 | 3000 | Tests PPO-favored skip with equal game-time horizon |

Diagnostic controls:

| condition | skip | rollout_len | max_rewind | max_actions | purpose |
|---|---:|---:|---:|---:|---|
| C: unscaled skip 8 | 8 | 48 | 30 | 3000 | Separates skip effect from too-long search horizon |
| D: frame-matched skip 4 | 4 | 36 | 23 | 6000 | Bridge to existing skip 3 vs 4 evidence |

Use the same `rollouts` for every condition. Run two budgets:

| phase | rollouts | max_time | seeds | purpose |
|---|---:|---:|---:|---|
| probe | 64 | 300s | 5 | Fast signal; catches obvious failures |
| confirmation | 512 | 900s | 5-10 | Win-rate and time-to-win comparison |

## Metrics

Primary metrics:

- win rate
- median wall-clock seconds to win
- median `sampled_actions`
- median sampled game frames: `sampled_actions * skip`
- median committed trace frames: `trace_steps * skip`

Secondary metrics:

- final reward for failed runs
- reward per wall-clock second
- reward per 1k sampled game frames
- prune shrinkage: `search_steps - trace_steps`
- number of time-cap failures vs action-cap failures

Do not rank skip values by `sampled_actions` alone. A sampled action at
`skip = 8` advances more emulator frames than one at `skip = 3`, and CPU cost is
closer to sampled game frames than sampled decision count.

## Readout rules

Prefer `skip = 8` for `mc_search` only if condition B beats or matches condition
A on win rate and median sampled game frames, without a major wall-clock
regression.

If condition C beats B, then `skip = 8` may need a longer lookahead than the
frame-matched `144` frames; the issue is horizon length, not skip itself.

If B loses to A but PPO still strongly prefers `skip = 8`, treat search and PPO
as different regimes: PPO benefits from temporally compressed credit assignment,
while random branch search may still need finer control for early level-1
hazards.

## Harness requirements

Patch both runtime skip globals per run, as the previous A/B did:

- `synthetic.mc_search.SKIP`
- `contra.replay.SKIP`

Future trace files now include `skip`, `sampled_actions`, `search_wall_s`,
`search_steps`, and `trace_steps`, so the raw JSON can be rebuilt from the saved
trace artifacts as long as the runner also records seed, win, and trace path.
