# Design: Pretrain the CNN policy on mc_traces to warm-start PPO

Status: **draft / for discussion**
Branch: `dev-cnn-pretrain`

## 1. Hypothesis

PPO from a random init spends a large fraction of its budget in the
"incoherent jitter" regime — the policy is ~uniform and the entropy bonus keeps
pulling it back toward uniform, so early rollouts rarely produce coherent
behaviour (hold right + fire, jump a gap). The `mc_search` bigram prior fixes
this *for search*; the equivalent for PPO is to **start the policy already
playing like the demonstrations**.

**Claim to test:** behavior-cloning the PPO CNN actor on the `mc_search` win
traces, then resuming PPO from that checkpoint, reaches a
target win-rate / `delta_x` in **meaningfully fewer environment steps** than PPO
from scratch — for the same wall-clock and reward config.

For the current actor-model experiments we pretrain **only the actor path**:
`features_extractor` → policy latent → `action_net`. The critic is deliberately
ignored so value regression cannot push the shared visual backbone away from the
action-cloning objective.

## 2. Background: what we're cloning into

PPO uses SB3 `CnnPolicy` (`ActorCriticCnnPolicy`). Relevant submodules:

| submodule | role | BC target |
|---|---|---|
| `features_extractor` (NatureCNN) | 84×84×stack → features | trained (shared) |
| `mlp_extractor` | features → (pi, vf) latents | trained |
| `action_net` (Linear) | pi latent → `Discrete(NUM_ACTIONS)` logits | cross-entropy vs demo action |
| `value_net` (Linear) | vf latent → scalar `V(s)` | ignored for actor-only BC |

Observation contract (must match `ContraWrapper` exactly, see
[contra_wrapper.py](contra_wrapper.py)): `(84, 84, stack)` uint8, channels
`R(t), G(t-1), B(t-3)`, frame-skip `ACTION_SKIP`, max-pool of the last two raw
frames. Any mismatch here silently poisons the warm-start, so the pretraining
**must generate observations through the same wrapper**, not a re-implementation.

Action space is the shared canonical flat space (`contra/action_space.py`), the
same one `mc_search` used to author the traces — so every trace action vector
maps to exactly one `Discrete` index.

## 3. Trace → supervised dataset

Each trace npz (`tmp/mc_trace/level<N>/win_*.npz`) holds:
```
actions       (T, 9) uint8   # raw NES button vectors, one per decision step
initial_state (bytes)        # emulator savestate the trace starts from
level, outcome, fps
```
Traces store actions, **not** observations or rewards — we regenerate those by
replay.

**Pipeline (per trace):**
1. Make a retro env + `ContraWrapper` with the *training* config (stack, skip,
   reward_weights). `reset()`, then overwrite the emulator with `initial_state`
   and re-sync `prev_ram`/`prev_xscroll` (same trick as
   [make_states.py](make_states.py) `replay_snapshots` and
   `RandomStateWrapper`), bypassing warmup so obs align with the trace.
2. Map each `actions[t]` (9-vector) → discrete index `a_t` via an exact-match
   lookup against `ACTION_SPACE.actions_np()`.
3. Step the wrapper with `a_t`. Record:
   - `obs_t` (the stacked observation the policy will see),
   - `a_t` (label for the actor),
   - `r_t` (the wrapper's shaped reward — retained for dataset diagnostics /
     possible future critic experiments).
4. After the rollout, compute discounted **return-to-go**
   `G_t = Σ_{k≥t} γ^{k-t} r_k` using the **same `gamma`** as PPO (0.99). The
   current actor-only trainer does not optimize against `G_t`.

Collect `(obs, a, G)` across all chosen traces into one dataset. Optionally
include traces for multiple levels, or restrict to the level being trained.

Notes / decisions:
- **Scope (decided): level 1 only**, **all available level-1 traces**
  (`tmp/mc_trace/level1/win_level1_*.npz`, currently 11) concatenated into one
  dataset. Multiple traces give state/action **variance** so the actor isn't fit
  to a single trajectory; this directly softens the entropy-collapse risk in
  §7.1.
- **Always use pruned traces.** `prune_actions` only removes button bits whose
  presence/absence leaves the NES RAM *identical* — so the pruned-out presses
  are **RAM-no-ops, i.e. pure noise**: the game plays out the same whatever those
  bits are. Training BC to reproduce them would teach the actor to imitate
  meaningless presses. Crucially, the *decisive* fire/jump (the press that
  actually spawns a bullet or initiates a jump → changes RAM) **survives
  pruning**, so the meaningful signal is intact; it's just genuinely rare. We
  handle that rarity with loss weighting / oversampling (§4), **not** by adding
  noise back via unpruned data.
- Returns-to-go are computed on the replayed reward of the same pruned rollout
  and kept in the cache, but ignored by the current actor-only trainer.
- Even with 11 traces the data is **narrow** (all winning, on-path); the
  anchor-state diversity that helps PPO is not present. That's fine — BC only
  needs to seed a reasonable actor basin; PPO explores out from there.

## 4. Training objective

Per minibatch of `(obs, a, G)`; `G` is loaded but unused by actor-only BC:

```
features = policy.extract_features(obs)
pi_lat, _ = policy.mlp_extractor(features)
logits   = policy.action_net(pi_lat)

L_actor   = CrossEntropy(logits, a, weight=class_w)   # class_w fights imbalance, §7.1
L_entropy = -mean(entropy(logits))                    # monitored, small/zero coeff
L = L_actor + c_e * L_entropy
```

- **`class_w` (critical):** per-action loss weights ∝ inverse frequency
  (e.g. `1/√freq`, clipped) so the rare-but-decisive `fire`/`jump` actions are
  not drowned out by the dominant "right"/"up-right" majority. Without this the
  actor fits only the trivial movement classes (the observed failure, §7.1).
  Alternative: **focal loss** (down-weights easy, frequent, already-correct
  examples). May also **oversample** rare-action steps in the sampler.
- `c_e`: default **0** here (entropy only *monitored*, not fought — §7.2);
  reserve a small floor for later if a warm-started run stalls.
- Standard supervised loop: Adam, a few epochs over the dataset, shuffled
  minibatches. This is cheap (minutes) vs the PPO run.

## 5. Producing a PPO-loadable checkpoint

We want PPO `--resume` to consume the warm-started weights with **zero** changes
to [train.py](train.py)'s resume path. Approach:

1. Build a `PPO("CnnPolicy", env, ...)` with the **exact** hyperparameters /
   spaces the real run uses (so the policy architecture matches).
2. Run the supervised loop above against `model.policy` (it exposes
   `extract_features`, `mlp_extractor`, and `action_net`; use a supervised
   optimizer outside SB3's rollout loop.
3. `model.save(out.zip)` and `save_config_to_model(...)` to embed
   `contra_config.json` (actions/skip/stack) just like training checkpoints, so
   the warm-start zip is self-describing and loadable by `infer`/`benchmark`.
4. Train: `python ppo/train.py --config <level>.yaml --resume <out.zip>`.

This keeps the warm-start a drop-in: same `PPO.load(...)` path already in
[train.py:342](train.py#L342).

## 5b. Phase 0 — confirm the model actually fits the traces (do this first)

From prior BC experience the actor rarely collapses to a deterministic model.
The **dominant, first-order risk is action class imbalance** (§7.1): the actor
learns the trivial frequent moves (right, up-right) but **fails to learn the
decisive rare actions — `jump` and `fire`**. So before any PPO A/B, the gate is:
*can the policy reproduce the rare critical actions, not just the common ones?*
(A net that can't even fit the *frequent* classes points instead at a
data-pipeline / obs-contract bug, §7.5.)

Watch during pretraining (log every epoch):
- **Per-button recall — the real gate (not aggregate accuracy).** Decode each
  predicted action into its `fire`/`jump`/dpad bits and track **`fire`-recall**
  and **`jump`-recall** separately. Aggregate accuracy is *misleading*: the data
  is dominated by "right"/"up-right", so a model that never presses fire/jump
  still scores high accuracy while being useless (the observed failure, §7.1).
  Success = fire- and jump-recall both climb high, not just overall accuracy.
- **Per-action confusion / class accuracy** — confirm the rare classes are
  actually predicted, not collapsed into the majority movement action.
- **Actor CE loss** — should fall steadily; a floor with low rare-class recall =
  the imbalance is winning (turn up `class_w` / oversampling, §4).
- **Greedy replay** from each trace's `initial_state`: the cloned actor, run
  greedily, should reproduce (near-)win progress. This is the real end-to-end
  check that obs + action mapping are correct.

Only once memorization is demonstrated do we move to the speedup experiment.
Entropy is *observed* here (we log mean actor entropy) but not yet fought — see
§7.1.

## 6. Experiment to validate "faster"

Single variable: warm-start vs scratch. Everything else fixed (level, reward
config, anchors, seed set, hyperparams, wall-clock).

- **A (baseline):** PPO from scratch.
- **B (warm-start):** PPO `--resume` from the BC checkpoint.

Metrics (already logged by `TensorboardCallback`): `contra/win_rate`,
`contra/mean_delta_x`, `contra/mean_reward` vs `num_timesteps`. Success =
**B reaches a fixed `win_rate`/`delta_x` threshold at fewer timesteps** and/or a
higher asymptote. Scope is **level 1** (cheap, fast feedback); harder levels are
out of scope for this first pass.

Prerequisite: the Phase-0 gate (§5b) must pass — only run the A/B once the BC
checkpoint demonstrably memorizes and greedily replays the traces.

## 7. Risks & open questions (for discussion)

1. **Action class imbalance — the observed, first-order failure.** Prior BC
   experiments here learned only the *trivial frequent* actions (move right,
   up-right) and **never learned the decisive rare ones (`jump`, `fire`)**. With
   a single flat softmax over `NUM_ACTIONS`, cross-entropy is minimized by
   predicting the majority movement classes; `fire`/`jump` are few-shot and
   high-stakes (miss jump → death, miss fire → no boss damage), so the actor
   scores high aggregate accuracy while being useless. Mitigations (§4):
   **inverse-frequency / focal class weighting** (primary) and **oversampling**
   rare-action steps — applied to the *pruned* labels (the decisive fire/jump
   survive pruning; pruned-out presses are RAM-no-op noise, §3) — plus the key
   instrumentation: **per-button `fire`/`jump` recall** as the gate, never
   aggregate accuracy (§5b). This, not entropy, is what Phase 0 must crack.
   (Genuine *underfitting* of even the frequent classes points further upstream:
   an obs-contract / action-mapping bug, §7.5.)
2. **Entropy collapse (secondary — watch, don't pre-empt).** In theory hard
   cloning drives the actor near-deterministic (entropy ≈ 0), a poor PPO init:
   exploration dies, gradients vanish (`π≈1` ⇒ `log π≈0`, ratio-clip bites), and
   it fights `train.py`'s `ent_coef_initial=0.1` schedule. **In practice this
   rarely happens** (and using all 11 traces, §3, adds spread that further
   guards against it), so we **only monitor** mean actor entropy during Phase 0
   rather than fighting it up front. *If* a warm-started PPO run then stalls with
   near-zero entropy, the cheap levers are an entropy floor in BC
   (`-c_e·entropy`, `c_e≈0.01`), fewer epochs / early-stop, or label smoothing.
3. **Narrow data.** Even 11 winning traces only cover on-path states. PPO
   recovers off-path. Acceptable for the first pass; could fold in anchor-state
   neighbourhoods later.
4. **Distribution shift from frame-skip/flicker.** Must replay through the real
   wrapper (covered in §3) — the #1 correctness trap and the most likely cause
   of the underfitting in §7.1.

## 8. Implementation plan (once design is agreed)

- `ppo/pretrain_dataset.py` — replay traces → `(obs, a, G)` arrays, reusing
  `ContraWrapper` + `ACTION_SPACE` + `reward_components`; stream observations
  to a memmapped `.npy` plus small `.npz` metadata.
- `ppo/pretrain_model.py` — model construction, visual backbones, policy forward
  helper, and actor metric helpers.
- `ppo/pretrain_train.py` — supervised actor-only loop, save a resume-able
  checkpoint (+ embedded config). CLI: `--level`, `--resolution`, `--backbone`,
  `--epochs`, `--ent-coef`, `--out`.
- Config: a `pretrain:` block or a small `pretrain_<level>.yaml`.
- Validation: greedy eval of the checkpoint; then the A/B PPO runs.
- No change required to `train.py` (reuse `--resume`).

## Decisions

- [x] Clone **actor only** for the current visual-backbone experiments.
- [x] Dataset: **all available level-1 traces** (currently 11) for variance.
- [x] Scope: **level 1 only** for the first validation.
- [ ] Entropy floor strength `c_e` — *tuned empirically* (start `≈0.01`, target
      a non-collapsed actor entropy; see §7.1). Not a blocker for coding.
- [ ] Where pretrain checkpoints live — proposing `tmp/ppo/pretrain/level1.zip`.
