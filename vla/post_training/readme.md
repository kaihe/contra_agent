# ContraVLA GRPO Post-Training Design

This document describes how to post-train a behavior-cloned ContraVLA policy with
GRPO so it moves beyond the common BC prior of "hold right". The target skills
are:

- shoot enemies and fixed hazards before collision;
- jump over pits and terrain hazards;
- dodge bullets, grenades, soldiers, and other moving threats;
- preserve the useful BC behavior of progressing through the level.

The design assumes the current BC model contract:

- observation: one image frame plus 118-dimensional RAM-derived state;
- action space: 36 discrete controller actions from `vla.datasets.preprocess`;
- model output: one action token, decoded to NES `MultiBinary(9)`;
- control rate: one model action every 3 emulator frames, matching the BC data.

## Starting Point

The BC policy is a good initialization because it already knows the observation
format, level text prompt, and the dominant direction of travel. It is also
biased because the dataset is imbalanced: most labels are rightward D-pad
actions, while shooting, jumping, crouching, and diagonal aim are rarer.

GRPO should not replace BC immediately. The policy should start from a trained
BC checkpoint and optimize short on-policy rollouts with a KL penalty against a
frozen reference copy of that same checkpoint. The reference policy keeps the
agent near the BC distribution while rewards push it toward combat and survival.

## Environment Setup

Use a single-process `stable_retro` wrapper for Contra:

- game: `Contra-Nes`;
- integration: `CUSTOM_ONLY`;
- state: `retro.State.NONE`, then restore `contra/integration/Contra-Nes/LevelN.state`;
- restricted actions: `retro.Actions.ALL`;
- observation mode: image, plus RAM readout from `env.unwrapped.get_ram()`;
- emulator skip: 3 frames per model action.

The VLA environment wrapper should expose:

```python
obs = {
    "images": torch.Tensor,   # [1, 3, 192, 192], ImageNet-normalized
    "proprio": torch.Tensor,  # [118], from contra.game_state.state_from_ram
}
```

Use one current frame because the BC checkpoint was trained with
`ContraVLADataset`, which returns `images` as `[1, 3, 192, 192]` and `actions`
as `[1]`. The batched model input is therefore `[B, 1, 3, 192, 192]`, and the
batched action target/output is `[B, 1]`. Frame stacking can be a future upgrade,
but it changes the model input contract and should not be introduced during
GRPO on an existing one-frame BC checkpoint.

The image preprocessing must match `vla.datasets.dataset`: resize to 192x192 RGB
and apply ImageNet mean/std.

The wrapper should provide:

- `reset(level="Level1") -> obs`;
- `step(action: int) -> (obs, reward, status, info)`;
- `snapshot() -> bytes`;
- `restore(state: bytes) -> obs`;
- `close()`.

`status` should be one of:

- `RUNNING`;
- `DEAD`, triggered by `EV_PLAYER_DIE`;
- `DONE`, triggered by `EV_LEVELUP` for the Level 1 task.

`info` should include enough data for debugging and reward logging:

- current x-scroll and y-scroll;
- current level;
- player lives;
- event list from `contra.events.scan_events`;
- per-event reward components.

Important emulator detail: start with one `stable_retro` emulator and one model
instance. For grouped GRPO rollouts from the same prompt/state, snapshot once
and maintain `G` branch snapshots. At each rollout step, restore each branch
sequentially to collect its current observation, batch the `G` observations into
one policy forward pass, then restore and step each branch with its sampled
action. This keeps model memory low while still using batched prediction.

## Action Decoding

The policy samples one action token in `[0, 35]`. Decode it exactly like the BC
dataset:

```text
action_id = dpad_id * 4 + button_id
```

D-pad ids:

```text
0 neutral, 1 left, 2 right, 3 up, 4 down,
5 up-left, 6 up-right, 7 down-left, 8 down-right
```

Button ids:

```text
0 none, 1 A/jump, 2 B/fire, 3 A+B
```

Each decoded NES action is repeated for 3 emulator frames. Check death and level
completion on every emulator frame, not only after the 3-frame block, because
some RAM events are short-lived.

## Reward Signal

For the first GRPO target, train only on Level 1 with the objective:

```text
clear Level 1 without losing a single life
```

Losing one life is terminal failure. This matches the BC data collection
assumption and should be enforced by the environment: as soon as `EV_PLAYER_DIE`
fires, end the rollout.

Use two kinds of reward:

- Dense reward: given during the rollout to teach local skills such as shooting,
  moving through the stage, picking up weapons, and avoiding stalls.
- Terminal reward: given only when the rollout ends, to define success or
  failure. For Level 1, terminal success is `EV_LEVELUP`; terminal failure is
  `EV_PLAYER_DIE`.

Dense reward is the coaching signal. Terminal reward is the mission score. GRPO
needs both: terminal-only learning is too sparse, while dense-only learning can
produce local behavior that never clears the level.

Reuse the event detectors in `contra.events`, but use a Level-1-specific reward
profile rather than blindly using every current event weight. Log every
component separately so reward hacking is visible.

Existing event rewards include:

| Event | Purpose | Current weight |
| --- | --- | ---: |
| `EV_ENEMY_HIT` | reward enemy HP reduction, including bosses and turrets | `+1.0 * HP delta` |
| `EV_PUSH_FORWARD` | reward horizontal progress | `+1 / 30 px` |
| `EV_PLAYER_DIE` | penalize enemy hit death and pit death | `-5000` |
| `EV_LEVELUP` | reward level completion | `+100` |
| `EV_GUN_PICKUP` | reward weapon pickup | `+10` |
| `EV_GUN_POWERUP` | reward rapid-fire pickup | `+10` |
| `EV_SPREAD_LOST` | penalize losing spread weapon | `-200` |

For Level 1 GRPO, rescale these into:

```text
R = dense_combat_reward
  + dense_progress_reward
  + dense_pickup_reward
  + terminal_success_or_failure
```

Recommended starting profile:

| Component | Suggested weight | Notes |
| --- | ---: | --- |
| enemy HP damage | `+2.0 * HP delta` | Main dense signal for learning to shoot. |
| forward progress | `+1 / 60 px`, capped per action | Small guardrail against camping; not the main objective. |
| weapon pickup | `+20` | Useful but secondary. |
| rapid-fire pickup | `+20` | Useful but secondary. |
| spread lost | `-100` | Meaningful but not catastrophic. |
| levelup | `+2000` | Terminal success: cleared Level 1. |
| player death | `-1000` | Terminal failure: lost one life. |

The current `EV_PLAYER_DIE = -5000` is probably too large for GRPO. Death is
terminal failure, so it must be strongly negative, but if it dwarfs all dense
signals then failed rollouts become hard to rank. We still want GRPO to learn
that one failed rollout was better than another because it shot enemies, jumped
a pit, survived longer, or reached a harder part of the level. `-1000` is a
better first scale when paired with `+2000` for Level 1 clear.


## Rollout Strategy

Use a policy-guided version of `synthetic/mc_search.py` as the rollout driver.
The MC search loop is already a stable way to discover winning traces: from the
current committed emulator state it samples many futures, chooses a good future,
commits a safe prefix, snapshots the new state, and rewinds when all sampled
futures fail. GRPO can keep this control structure and replace the random or
bigram rollout sampler with the VLA policy.

The important distinction is that search and training use the same sampled
group differently:

- search commits only the best non-death prefix, to move the real emulator
  forward and create new reachable states;
- GRPO trains on the whole group of sampled rollouts from the same starting
  state, using group-relative returns as advantages.

This gives the trainer both properties we want: MC-style stability for data
generation, and policy-gradient learning from all sampled attempts rather than
only cloning the selected best path.

Policy-guided MC-GRPO loop:

1. Restore the current committed emulator state `s_commit`.
2. Sample `G` policy rollouts from exactly `s_commit`.
3. For every rollout, store observations, action ids, old logprobs, rewards,
   terminal status, and event counters.
4. Score each rollout with the shaped dense plus terminal reward.
5. Normalize returns across the `G` rollouts to produce GRPO advantages.
6. Run one or two GRPO epochs on this group, with KL to the frozen BC reference.
7. Choose the highest-return rollout that does not die during its committed
   prefix.
8. Commit a random prefix of that rollout into the emulator, matching the MC
   search behavior: if the selected rollout has `n` actions, sample
   `commit_n` uniformly from `[n // 2, n]` when `n >= 2`, otherwise commit the
   single action.
9. After every committed action, save `env.em.get_state()` into the on-policy
   anchor buffer.
10. If all rollouts die, sample `rewind_back` uniformly from
    `[1, min(max_rewind, len(committed_actions))]`, rewind to that earlier
    committed snapshot, increase rollout count or temperature temporarily, and
    continue from the rewound state.

Pseudo-code:

```python
committed_state = level_start_state
committed_actions = []
committed_states = []

while training:
    group = [new_empty_traj() for _ in range(group_size)]
    branch_states = [committed_state for _ in range(group_size)]

    for t in range(rollout_len):
        obs_batch = []
        active = []

        for i, branch_state in enumerate(branch_states):
            if group[i].done:
                continue
            env.restore(branch_state)
            obs_batch.append(current_obs(env))
            active.append(i)

        if not active:
            break

        actions, logprobs = policy.sample_batch(
            obs_batch,
            temperature=temperature,
        )

        for i, action, logprob in zip(active, actions, logprobs):
            env.restore(branch_states[i])
            obs, reward, status, info = env.step(action)
            group[i].append(action, logprob, reward, status, info)
            branch_states[i] = env.snapshot()

    normalize_group_advantages(group)
    grpo_update(policy, reference_policy, group)

    best = best_non_death_rollout(group)
    if best is None:
        n = len(committed_actions)
        if n > 0:
            rewind_back = randint(1, min(max_rewind, n))
            rewind_to = n - rewind_back
            committed_state = (
                level_start_state
                if rewind_to == 0
                else committed_states[rewind_to - 1]
            )
            committed_actions = committed_actions[:rewind_to]
            committed_states = committed_states[:rewind_to]
        else:
            committed_state = level_start_state
        temporarily_increase_exploration()
        continue

    n = len(best.actions)
    commit_n = randint(n // 2, n) if n >= 2 else n

    env.restore(committed_state)
    for action in best.actions[:commit_n]:
        obs, reward, status, info = env.step(action)
        if status == "DEAD":
            break
        committed_actions.append(action)
        committed_state = env.snapshot()
        committed_states.append(committed_state)
        on_policy_anchor_buffer.add(committed_state, info)
        if status == "DONE":
            reset_or_start_next_search()
            break
```

The initial implementation should stay close to `mc_search.py`:

- keep one committed search path for the single emulator owner;
- use batched model prediction across the `G` rollout branches, but restore and
  step emulator branch states sequentially;
- store `committed_actions`, `committed_states`, and cumulative rewards;
- on all-death groups, choose the rewind distance randomly from
  `1..min(max_rewind, len(committed_actions))`;
- after choosing the best rollout, choose the commit length randomly from half
  to all of that rollout;
- commit only actions that were replayed from the last committed snapshot;
- rewind to a previous committed snapshot when all sampled futures die;
- log death rate, best return, committed return, event tags, action histogram,
  and current x-scroll after every commit block.

Use the VLA action space during policy rollouts. `mc_search.py` uses a trimmed
random/search action prior, but GRPO must sample and train the full 36-action
vocabulary from `vla.datasets.preprocess` so rollout actions match the model
contract.

Recommended defaults:

```yaml
group_size: 32               # G rollouts from the same committed state
rollout_len: 48              # actions; 144 emulator frames at skip 3
commit_min: 24               # random commit_n is sampled from [24, 48]
commit_max: 48
max_rewind: 32               # random rewind_back is sampled from [1, min(32, path_len)]
grpo_epochs: 1               # avoid overfitting stale rollout data
clip_eps: 0.2
kl_beta: 0.02
entropy_coef: 0.001
lr: 1.0e-6 to 5.0e-6
temperature: 0.8-1.0
max_grad_norm: 1.0
```

## GRPO Objective

For each anchor state `s`, collect `G` sampled trajectories:

```text
tau_i = (obs_t, action_t, logprob_t, reward_i), i = 1..G
```

Compute a group-relative advantage:

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

Then optimize token-level policy ratios:

```text
ratio_t = exp(logprob_theta(a_t | obs_t) - logprob_old(a_t | obs_t))
L_clip = min(ratio_t * A_i, clip(ratio_t, 1-eps, 1+eps) * A_i)
L_kl = beta * KL(policy_theta || reference_policy)
L = -mean(L_clip - L_kl + entropy_coef * entropy)
```

Use the frozen BC checkpoint as `reference_policy`. Use the rollout policy as
`old_policy` for saved logprobs. Update the model for a small number of epochs
per rollout batch, then discard stale rollouts.

## Training Metrics

The scalar GRPO loss is hard to interpret because it mixes clipped policy
ratios, normalized advantages, KL, and entropy. Treat it as an optimization
debug signal, not as the main measure of progress. The main question is whether
the policy-guided search needs less luck and produces better committed paths
over time.

Track these metrics closely:

| Metric | Why it matters |
| --- | --- |
| Best rollout return per group | Shows whether the current policy can still discover a good future from the committed state. |
| Mean rollout return per group | Shows whether the whole sampled group is improving, not just one lucky branch. |
| Return std within group | GRPO needs contrast. Near-zero std means the group gives little learning signal. |
| Death rate per group | Primary survival signal. Should fall over training, especially from common hard states. |
| All-death group rate | Directly measures search failure. High values mean the policy cannot find any safe branch from many states. |
| X-scroll progress per committed action | Measures real level progress, less noisy than raw shaped reward. |
| Action histogram | Detects collapse to right-only, no-fire, no-jump, or other degenerate behavior. |
| Policy entropy | Too low means premature collapse; too high means the sampler may still be random. |
| KL to BC reference | Keeps track of how far the policy has moved from the BC checkpoint. |
| Clip fraction | Fraction of tokens where PPO clipping activates; high values mean updates are too aggressive. |
| Approx policy ratio | Mean and max `ratio_t`; catches unstable policy updates. |
| Grad norm | Debugs exploding updates, especially after high-variance groups. |

Recommended logging cadence:

- per group: best/mean/std return, death rate, all-death flag, action histogram,
  event counters, entropy, KL, clip fraction;
- per commit block: committed actions, commit length, committed return delta,
  x-scroll delta, rewind count, current x-scroll;
- periodic evaluation: greedy clear rate, low-temperature clear rate, average
  x-scroll, death cause breakdown.

Healthy early training usually looks like this:

- best rollout return improves before mean rollout return improves;
- all-death groups become less frequent;
- committed x-scroll advances farther between rewinds;
- action histograms gain more fire/jump/crouch actions without losing the
  rightward movement prior;
- KL rises gradually instead of jumping in a few updates.

Warning signs:

- return std is near zero for many groups, so GRPO has no useful ranking signal;
- death rate falls only because the agent stalls or refuses to move;
- best return improves but committed progress does not, which suggests reward
  hacking or unsafe commit prefixes;
- KL, clip fraction, or grad norm spikes at the same time as clear rate drops;
- action entropy collapses and the histogram returns to mostly right-only.