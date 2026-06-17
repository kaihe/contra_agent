# PPO Training Design

This directory contains the Stable-Baselines3 PPO baseline for Contra NES. The
current design is intentionally small: one training entrypoint, one environment
wrapper, and one YAML config file.

## Quick Start

From the repository root:

```bash
python ppo/train.py --config ppo/ppo.yaml
```

Short test run:

```bash
python ppo/train.py --config ppo/ppo.yaml --name smoke --timesteps 1000000
```

Resume:

```bash
python ppo/train.py --config ppo/ppo.yaml --resume tmp/ppo/checkpoints/baseline/baseline_final.zip
```

## Files

- `ppo.yaml` is the default training config.
- `train.py` builds vectorized environments, creates or resumes the PPO model,
  attaches callbacks, and runs `model.learn()`.
- `contra_wrapper.py` converts the raw stable-retro environment into the PPO
  interface: action space, frame history, reward shaping, episode metadata, and
  model config embedding.
- `test.py` loads a trained model and records a rollout.

Checkpoints are grouped by experiment name under
`tmp/ppo/checkpoints/<name>/`. TensorBoard similarly groups logs by name and
adds its own numeric run suffix.

## Config Flow

`train.py` loads `ppo.yaml` into `PPOConfig`. CLI flags override common run
fields:

- `--config`
- `--timesteps`
- `--resume`
- `--state`
- `--random-start`
- `--name`

The YAML `timesteps` field and CLI `--timesteps` flag both mean steps to run
in the current invocation. A resume run adds those steps to the checkpoint's
existing count.

Unknown YAML fields are rejected. Unknown reward keys are also rejected. This is
deliberate: PPO runs are expensive, so typo-driven config drift should fail
early.

Training starts from `state` by default. When the optional `states` list is set
(see `level1_win.yaml`), each episode instead loads a uniformly sampled `.state`
file from that list (multi-state training). Anchor savestates for Level 1 live
in `ppo/states/`; they were captured during human play (chapter 9 fixed-anchor
technique) and let episodes start mid-level and at the boss arena, removing the
exploration bottleneck of always replaying from x=0.

## Environment Stack

Each worker is created by `make_env()`:

```text
stable_retro.make(...)
  -> ContraWrapper(...)
  -> stable_baselines3 Monitor(...)
  -> SubprocVecEnv
```

`num_envs` controls how many subprocess environments run in parallel. With the
current config:

```text
num_envs = 16
n_steps  = 512
rollout  = 16 * 512 = 8192 environment steps per PPO update
```

`batch_size` is the optimizer minibatch size used by SB3. With
`batch_size = 2048`, each rollout is split into 4 minibatches per epoch.

## Actions

The wrapper exposes a flat discrete action space:

```text
Discrete(NUM_ACTIONS)
```

Each action is one named NES button vector. The set is defined in
`contra/action_configs/baseline.json` as a `name -> vector` map (e.g.
`"RF" -> Right+Fire`) and loaded via `contra/action_space.py`. The same config
is shared with the Monte-Carlo searcher (`synthetic/mc_search.py`) and the frame
`skip`, so a win path found by search is reproducible by the trained policy.
The agent picks an index; the wrapper holds that vector for `skip` frames.

## Observations

The policy receives temporal `84 x 84 x 3` observations, channels-last. The
wrapper keeps three recent RGB frames:

```python
HISTORY_OFFSETS = [0, 1, 3]
```

Each output channel is sliced from a different historical RGB frame:

```text
channel 0: R from frame_t
channel 1: G from frame_t-1
channel 2: B from frame_t-3
```

This keeps the normal three-channel image shape while smuggling in short-term
temporal context. It is not a normal RGB image: each color channel comes from a
different moment in time.

`stack` must match `len(HISTORY_OFFSETS)` and the three channels are tied to the
RGB channel order.

## Model Architecture

`train.py` currently creates PPO with:

```python
PPO("CnnPolicy", env, ...)
```

In Stable-Baselines3, `CnnPolicy` is not a custom Contra model. It is SB3's
standard actor-critic CNN policy for image observations. The default feature
extractor is `NatureCNN`, the Atari-style convolutional encoder from the DQN
Nature paper.

Our wrapper exposes observations as channels-last:

```text
84 x 84 x 3
```

SB3 detects image observations and internally transposes them to channels-first
for PyTorch:

```text
3 x 84 x 84
```

The default `NatureCNN` feature stack is:

```text
Conv2d(3, 32, kernel=8, stride=4)
ReLU
Conv2d(32, 64, kernel=4, stride=2)
ReLU
Conv2d(64, 64, kernel=3, stride=1)
ReLU
Flatten
Linear(flattened, 512)
ReLU
```

That produces a 512-dimensional feature vector. For SB3 CNN actor-critic
policies, the default `net_arch` is empty, so there are no extra hidden MLP
layers after the CNN feature extractor unless `policy_kwargs` is provided.

The actor and critic then branch from the shared 512-dimensional features:

```text
shared NatureCNN features
  -> policy/action distribution head
  -> value head
```

Because the action space is:

```text
Discrete(NUM_ACTIONS)
```

the policy distribution is a single categorical over the flat action list. The
sampled index is mapped by `ContraWrapper` to its NES button vector.

The value head outputs one scalar:

```text
V(s)
```

which PPO uses for advantage estimation and value loss.

Important implications:

- The model has no recurrence or memory beyond the three history channels.
- The D-pad and button actions are sampled as separate categorical decisions,
  but they are conditioned on the same CNN features.
- The architecture is generic Atari-style CNN PPO, not a Contra-specific model.
- To change the architecture, pass `policy_kwargs` in `train.py` or introduce a
  custom SB3 policy/features extractor.

## Parameter Estimate

For the current observation and action spaces, the model has about **1.69M
trainable parameters**.

Input:

```text
3 x 84 x 84
```

Feature extractor:

```text
Conv1: 3 -> 32, kernel 8, stride 4
  output: 32 x 20 x 20
  params: 32 * 3 * 8 * 8 + 32 = 6,176

Conv2: 32 -> 64, kernel 4, stride 2
  output: 64 x 9 x 9
  params: 64 * 32 * 4 * 4 + 64 = 32,832

Conv3: 64 -> 64, kernel 3, stride 1
  output: 64 x 7 x 7
  params: 64 * 64 * 3 * 3 + 64 = 36,928

Linear: 64 * 7 * 7 -> 512
  params: 3,136 * 512 + 512 = 1,606,144
```

NatureCNN subtotal:

```text
1,682,080 parameters
```

Heads:

```text
Policy head: 512 -> 11 logits
  11 = 7 D-pad logits + 4 button logits
  params: 512 * 11 + 11 = 5,643

Value head: 512 -> 1
  params: 512 * 1 + 1 = 513
```

Total:

```text
1,682,080 + 5,643 + 513 = 1,688,236 parameters
```

Adam optimizer state is not counted here. During training, optimizer state and
rollout buffers use much more memory than the raw parameter tensor itself.

## Reset And Step Semantics

On reset, the wrapper:

1. Resets the retro env (with `states` set, a random anchor savestate is loaded).
2. Runs no-op frames until the game enters active play (`is_gameplay`), capped
   at `warmup_frames`. Anchor states are already mid-gameplay, so they skip
   warmup entirely.
3. Optionally runs up to `random_start_frames` extra no-op frames.
4. Snapshots RAM and fills the frame ring buffer with the current processed
   frame.

On each agent step, the selected NES action is repeated for `skip` emulator
frames, with two exceptions borrowed from the chapter-5 redesign:

- The B (fire) button is released on the last skip frame so auto-fire
  retriggers on the next step (otherwise holding B fires exactly once — the
  "B-stuck" bug).
- The observation frame is the max-pool of the last two raw frames, which
  defeats NES sprite flicker.

`ppo.yaml` (baseline) uses `skip: 3`; `level1_win.yaml` uses `skip: 8`, which
matches the bullet cooldown and human play rhythm.

Player death is **not** terminal: the per-death penalty applies and the player
respawns with remaining lives. Episodes end on level completion (`win`), true
game over (lives exhausted, signaled by the retro scenario), or `time_out`
after `max_episode_steps` agent steps.

## Reward Design

Rewards are computed from RAM event detectors in `contra.events`:

- `enemy_hp`
- `boss_hp`
- `progress`
- `spread_pick`
- `levelup`
- `player_die`
- `time_out`

`spread_pick` uses `EV_SPREAD_PICK`: it is positive when the player picks up the
Spread Gun and negative when the player loses it. Generic weapon pickups and
generic rapid-fire pickups are intentionally not rewarded in PPO, because the
Spread Gun is the weapon we most want the policy to preserve.

`DEFAULT_REWARD_WEIGHTS` provides defaults, and `ppo.yaml` can override them.
`enemy_hp` covers regular enemies and `boss_hp` covers bosses, minibosses, and
finite boss-objective components. `enemy_hp_cap_per_region` limits rewarded
regular-enemy damage in each 256-pixel scroll region per episode. Boss damage is
not capped. This bounds rewards from endlessly respawning enemies while
preserving dense feedback during boss fights.
The wrapper emits both event counts and weighted rewards into the episode `info`
dict:

```text
episode_enemy_hp_event
episode_enemy_hp_reward
episode_boss_hp_event
episode_boss_hp_reward
episode_progress_event
episode_progress_reward
...
```

Important: reward logging is diagnostic. The TensorBoard reward metrics do not
change the reward function.

## TensorBoard Logging

`TensorboardCallback` aggregates every 100 completed episodes and logs:

```text
contra/mean_max_x
contra/mean_reward
contra/end_time_out
contra/end_game_over
contra/end_win
```

It also logs reward diagnostics:

```text
reward/return_mean
reward/return_std
reward/normalized_return_mean
reward/normalized_return_abs_mean
reward/<component>_mean
reward/<component>_std
reward/<component>_abs_pct
reward/<component>_event_freq
reward/<component>_event_mean
reward/terminal_player_die_freq
reward/terminal_levelup_freq
```

`reward/<component>_abs_pct` is the component's share of absolute reward
magnitude over the logging window. It helps answer questions like "is PPO mostly
learning progress, enemy damage, or terminal events?"

## Schedules

The learning rate is constant. Clip range and entropy coefficient use linear
schedules:

```yaml
clip_range_initial: 0.2
clip_range_final: 0.05
ent_coef_initial: 0.1
ent_coef_final: 0.005
```

SB3 passes `progress_remaining` from `1.0` at the start of training to `0.0` at
the end. `EntropyScheduleCallback` mutates `model.ent_coef` during training and
logs `train/ent_coef`.
