# ContraVLA RL Post-Training Design

After the baseline behavior-cloning (BC) model is trained on human recordings and Monte-Carlo search traces, we run **online RL post-training** to push performance beyond the demonstration data. The BC model gives us a strong prior; RL fine-tunes it for higher scores, fewer deaths, and faster level completion.

---

## 2. Algorithm: GRPO (Recommended)

We use **Group Relative Policy Optimisation** (GRPO) rather than classic PPO. GRPO is the de-facto standard for RL fine-tuning of VLM/VLA models (DeepSeek-R1, OpenVLA, etc.) because it eliminates the value network.

### 2.1 Why GRPO over PPO?

| Property | PPO | GRPO |
|----------|-----|------|
| Critic network | Required (adds ~40M params + optimizer state) | **Not required** |
| Memory footprint | ~13 GB (batch=24, unfrozen) | **~9 GB** |
| Advantage estimation | GAE (needs value function accuracy) | Group mean baseline (always unbiased) |
| Implementation complexity | High (separate actor/critic loops) | **Low** (single model, sample a group) |
| Action-chunk compatibility | Awkward (value head must predict return for a chunk) | **Natural** (reward is sum over chunk) |

### 2.2 Connection to MC Search

Our existing Monte-Carlo search (`synthetic/mc_search.py`) already does exactly what GRPO needs at the rollout level:

1. **Sample** `N` random action sequences from the current state.
2. **Score** each sequence with `contra.events.compute_reward()`.
3. **Commit** the highest-scoring sequence.

GRPO generalises this idea into a differentiable policy-update rule:
- Instead of *committing* the best rollout, we *nudge* the policy toward all rollouts proportionally to their relative advantage.
- The frozen BC checkpoint acts as a "soft reset" (KL penalty) instead of the hard backtracking (`rewind`) in MC search.

### 2.3 GRPO Objective

For every environment state `s`, sample a **group** of `G` action chunks from the current policy:

```
{a_1, a_2, ..., a_G} ~ π_θ(. | s)
```

Execute each chunk in the emulator and observe returns `{R_1, ..., R_G}`. Compute relative advantages:

```
A_i = (R_i - mean(R)) / std(R)          # group-normalised advantage
```

The GRPO loss is:

```
L_GRPO(θ) = - (1/G) Σ_i  min( r_i(θ) A_i,  clip(r_i(θ), 1-ε, 1+ε) A_i )
            + β_KL · KL( π_θ || π_ref )
```

where `r_i(θ) = π_θ(a_i|s) / π_θ_old(a_i|s)` and `π_ref` is the frozen BC checkpoint.

The **KL penalty** is critical: it prevents the RL policy from forgetting the BC prior and collapsing to degenerate strategies (e.g. standing still to avoid death).

---

## 3. Model Architecture

### 3.1 Actor — The VLA Policy

Reuse the full ContraVLA model (SmolVLM-256M + action transformer). During RL we keep the **entire model trainable** but with a low LR (`2e-5`) because the BC prior is already strong.

```
input:  [2 frames, text instruction, proprio (118-dim)]
        ↓
ContraVLA.forward_vlm_efficient  →  VLM features
        ↓
Action Transformer  →  logits [B, T, 36]
        ↓
sample actions (categorical, temperature=1.0 for exploration)
```

### 3.2 Reference Model — Frozen BC Checkpoint

Load the best BC checkpoint, freeze all parameters, and use it to compute:
- `log π_ref(a|s)` for the KL term
- Per-token log-probabilities for importance-ratio clipping

This is kept in CPU or a separate GPU to save VRAM, or we can compute it once per rollout and cache.

### 3.3 No Critic Network

GRPO replaces the critic with the **group mean baseline**. This avoids:
- Adding a value head to the action transformer
- Training a separate vision-based value network
- GAE hyper-parameter tuning (`λ`, `γ` trade-offs)

The only trade-off is that we need `G ≥ 4` samples per state, which increases rollout cost. We amortise this with chunk execution (see §4).

---

## 4. Environment Interface: VLAEnv

A thin `gymnasium.Wrapper` around `stable_retro` that converts emulator state into the VLA model's input format.

### 4.1 Observation Space

```python
{
    "images":   Tensor[2, 3, 192, 192],   # last 2 frames, resized, ImageNet-norm
    "input_ids": Tensor[L],                # tokenised level instruction
    "proprio":  Tensor[118],               # structured RAM state
}
```

### 4.2 Action Execution Strategy

Our MC search already demonstrates that a horizon of **48 steps** (`rollout_len=48`) is the sweet spot for Contra — long enough to see if a decision leads to death or progress, short enough to remain computationally tractable.

We mirror this in GRPO: each group member executes **K=48 actions** before the group comparison. Because the VLA model predicts chunks of `T=8`, we generate the 48-action sequence autoregressively:

```
chunk_0 = model(obs_0)          # actions [0..7]
obs_1   = env.step(chunk_0)
chunk_1 = model(obs_1)          # actions [8..15]
...
chunk_5 = model(obs_5)          # actions [40..47]
```

This yields a 48-action trajectory whose **total return** is used for the GRPO group advantage, exactly matching the MC search evaluation horizon.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `T` (model chunk) | 8 | Fixed by BC training |
| `K` (execution horizon) | **48** | Matches MC search `rollout_len` |
| Chunks per rollout | 6 | `48 / 8` |
| Replan frequency | Every 48 actions | Amortises inference cost |

**Why not K=4?** A 4-step horizon is too short for meaningful differentiation — most group members would receive nearly identical returns because Contra rewards are sparse (enemy kills, scroll progress) and death penalties arrive unpredictably. MC search learned empirically that 48 steps is where signal emerges.

### 4.3 Rollout Collection

```python
for episode in range(num_episodes):
    obs = env.reset()
    trajectory = []
    
    while not done:
        # Sample G action chunks from current policy
        chunks = policy.sample_group(obs, G=GROUP_SIZE)   # [G, T]
        
        # Execute first K actions of each chunk in parallel envs
        # (or sequentially if single env)
        rewards = []
        for chunk in chunks:
            reward = execute_chunk(env, chunk[:K])
            rewards.append(reward)
        
        # Store (obs, chunk, reward, log_prob_ref) for update
        trajectory.append(VLATransition(...))
```

For efficiency we use `SubprocVecEnv` with `N_envs = G` (one env per group member).

---

## 5. Reward Design

Raw stable-retro score is too sparse. We shape rewards to guide exploration.

### 5.1 Reusing the MC Search Reward

We reuse `contra.events.compute_reward()` directly — it is already level-aware, handles edge cases (L3 falling-rock exclusion, indoor transitions, weapon tracking), and has been battle-tested through thousands of MC search rollouts.

```python
from contra.events import compute_reward

# Inside the env wrapper
pre_ram = env.unwrapped.get_ram().copy()
step_env(env, action)
curr_ram = env.unwrapped.get_ram()
step_reward = compute_reward(pre_ram, curr_ram)   # level-aware, shaped
```

**Event breakdown** (from `contra/events.py`):

| Event | Trigger | Weight | Notes |
|-------|---------|--------|-------|
| `ENEMY_HIT` | Sum of enemy HP decrements | +1.0 / HP | Excludes L3 endless rocks |
| `PUSH_FORWARD` | `Δ scroll_x` pixels | +1/30 per px | Normalised by screen width |
| `PUSH_UP` | `Δ scroll_y` (wrap-aware) | +1/2 per px | Waterfall levels |
| `SPREAD_PICK` | Picked up spread gun | +10 | — |
| `SPREAD_LOST` | Lost spread gun | **-200** | Strong penalty |
| `PLAYER_DIE` | Death (enemy or pit) | **-5000** | Dominant signal |
| `GUN_PICKUP` | Any weapon pickup | +10 | — |
| `GUN_POWERUP` | Rapid-fire flag gained | +10 | — |
| `LEVELUP` | Level complete | +100 | — |
| `GAME_CLEAR` | Final boss defeated | +1000 | L8 only |
| `CORE_BROKEN` | Indoor core destroyed | +10 | L2/L4/L6 |
| `PUSH_INSIDE` | Walking through door | +0.5 / step | Bridges long transitions |

**Return accumulation** (identical to MC search):

```python
R = Σ γ^t · compute_reward(pre_ram_t, curr_ram_t)
```

with `γ = 0.99`.

### 5.2 Additional RL Penalties

| Component | Formula | Purpose |
|-----------|---------|---------|
| **KL penalty** | `-β_KL · KL(π_θ \|\| π_ref)` | Stay close to BC prior |
### 5.3 Reward Normalisation

Maintain a running mean/std of returns across recent rollouts and z-score normalise before computing advantages. This makes GRPO's group mean baseline more stable.

Because `PLAYER_DIE` emits `-5000`, raw returns can have extreme variance. We clip normalised advantages to `[-5, 5]` to prevent a single death from dominating the gradient.

---

## 6. Training Loop

### 6.1 Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  VecEnv(G)  │ ──► │  RolloutBuf  │ ──► │  GRPO Update │
│  (CPU/GPU)  │     │  (obs, acts, │     │  (4 epochs)  │
│             │     │   rewards,   │     │              │
│             │     │   logπ_ref)  │     │              │
└─────────────┘     └──────────────┘     └─────────────┘
```

### 6.2 Update Steps

1. **Collect** `N_rollouts = 64` episodes per iteration (8 parallel envs × 8 episodes each).
2. **Group** transitions by state: for each state we have `G=4` action chunks.
3. **Compute returns** `R = Σ γ^t r_t` with `γ = 0.99`.
4. **Normalise** returns across the group: `A_i = (R_i - μ_R) / σ_R`.
5. **Update policy** for `4` epochs on the rollout buffer with clipped surrogate loss + KL.
6. **Repeat** until convergence.

### 6.3 Mixed Precision & Memory

| Setting | Value | Reason |
|---------|-------|--------|
| Forward dtype | `bfloat16` | Fast, no scaling issues |
| Optimiser dtype | `float32` | Stable gradients |
| Gradient checkpointing | `True` | Saves ~3 GB VRAM |
| Batch size | 24 | Fits in 16 GB GPU (measured) |
| Group size `G` | 4 | Memory/quality trade-off |

---

---

## 8. Implementation Plan

### 8.1 New Files

```
vla/post_training/
├── post_training.md               # This document
├── grpo.yaml                      # Hyperparameters
├── grpo.py                        # GRPO trainer (main loop)
├── env_wrappers.py                # VLAEnv + reward shaping
└── rollout_buffer.py              # Store (obs, chunk, reward, logπ_ref)
```

### 8.2 Key Modules

#### `env_wrappers.VLAEnv`
- Wraps `stable_retro` or `contra_wrapper`
- Converts `em.get_screen()` + RAM → VLA input dict
- Applies reward shaping (scroll progress, survival bonus)
- Supports chunk execution (`execute_chunk(actions, K)`)

#### `rollout_buffer.RolloutBuffer`
- Stores transitions for one iteration
- Groups transitions by state for GRPO advantage computation
- Computes per-token log-probs under both `π_θ` and `π_ref`

#### `grpo.py.GRPOTrainer`
- `collect_rollouts()` — run VecEnv, fill buffer
- `compute_advantages()` — group-mean baseline normalisation
- `update_policy()` — clipped surrogate + KL loss, 4 epochs
- `train()` — outer loop

### 8.3 Reusing Existing Code

| Existing module | Reuse in RL |
|-----------------|-------------|
| `vla.model.ContraVLA` | Actor + reference policy |
| `vla.datasets._preprocess_frames` | Frame resize / norm in env wrapper |
| `contra.game_state.state_from_ram` | Proprio extraction |
| `contra.replay` | Emulator state save/load for resets |
| `pixel2play.model.nes_actions` | Action encoding / decoding |
| `ppo.contra_wrapper` | Base env wrapper (reward shaping, frame skip) |

---

## 9. Evaluation

Track these metrics every `N` iterations:

| Metric | How measured |
|--------|--------------|
| **Win rate** | % of episodes that reach the level-end boss |
| **Mean return** | Average shaped return per episode |
| **Mean KL divergence** | `KL(π_θ \|\| π_ref)` — should stay < 0.5 nats |
| **Action entropy** | `H(π_θ)` — should not collapse to < 1.0 |
| **Death count** | Lives lost per episode |
| **Time to boss** | Emulator frames until boss spawn |

Use **TensorBoard** (`tmp/tf_logs/grpo`) for live plotting.

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Catastrophic forgetting | Strong KL penalty (`β_kl = 0.05`) + keep reference model frozen |
| Reward hacking (e.g. farming points) | Progress-shaping dominates; cap score bonus per enemy type |
| Exploration collapse | Temperature sampling (`τ = 1.0`) + entropy bonus if needed |
| VRAM OOM | Gradient checkpointing + `G = 4` + `K = 4` (not full T=8 replanning) |
| Slow rollouts | Use `SubprocVecEnv` with 8 parallel NES emulators on CPU |

---

## 11. Alternatives (Future Work)

| Method | When to use |
|--------|-------------|
| **DPO** (offline) | If we can generate preference pairs (winning vs losing chunks) from MC search |
| **PPO + value head** | If GRPO group variance is too high; add a lightweight MLP critic on VLM features |
| **Rejection Sampling Fine-Tuning (RFT)** | Simple baseline: keep top-20 % rollouts by return, fine-tune with BC |
| **MCTS-guided RL** | Use the existing MC search graph to provide denser reward signals |

---

## 12. Summary

1. Load the best BC checkpoint as both **actor** and **frozen reference**.
2. Wrap `stable-retro` into `VLAEnv` that feeds 2-frame + RAM observations.
3. Collect rollouts with `SubprocVecEnv`, sample `G=4` chunks per state.
4. Compute group-normalised advantages and update with clipped surrogate + KL.
5. Monitor win rate, KL, and entropy to avoid collapse.
6. Expected outcome: **+10–15 % absolute win rate**, fewer deaths, faster completion.
