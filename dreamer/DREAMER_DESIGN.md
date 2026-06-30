# DreamerV3 for Contra — Architecture & Update Rules

A from-scratch DreamerV3 agent (`dreamer/`). This doc explains **what each
component is** and **how it is trained (the update rule)**, grounded in our code
and shapes. Components are tagged with the **same build-ladder labels** used
throughout (`C3a`, `C3b`, …), not document section numbers, so the doc and the
build status line up 1:1. Where our implementation simplifies canonical
DreamerV3, it says so.

## Component map (build ladder → this doc)

| tag | component | file | section | gate |
|---|---|---|---|---|
| C1 | env adapter | `envs.py` | data plumbing | smoke |
| C2 | replay buffer | `buffer.py` | data plumbing | sanity |
| **C3a** | encoder / decoder | `models.py` | World Model | probe ✅ |
| **C3b** | RSSM (dynamics) | `world_model.py` | World Model | dream ✅* |
| **C3c** | reward + continue heads | `world_model.py` | World Model | r=0.905 ✅ |
| **C4** | critic + λ-returns | (todo) | Behavior | value tracks return |
| **C5** | actor | (todo) | Behavior | real-env return ↑ |
| C6 | full interleaved loop | (todo) | Training | win-rate ↑ |

\* C3b passes for 15-step imagination from active states; see open issue in Notes.

---

## The one-paragraph mechanism

Dreamer learns a **world model** (`C3a`+`C3b`+`C3c`) — a latent simulator of
Contra — from replayed `(frame, action, reward, continue)` data. It then learns
an **actor** (`C5`) and **critic** (`C4`) *entirely inside that simulator*:
starting from real states, it rolls the actor forward in imagination (latent
space, no pixels), scores those dreamed trajectories with the learned
reward/value, and improves the policy by gradient. At deploy the actor is a
feed-forward reflex.

```
   ┌─────────────── WORLD MODEL  (C3a + C3b + C3c) ───────────────┐
   image ─►[encoder C3a]─►embed ─┐
                                 ├►[RSSM C3b]─►(h,z)─►[decoder C3a] ─► image  (recon)
   action ───────────────────────┘                 ├►[reward C3c]  ─► reward (reward)
                                                    └►[continue C3c]─► cont   (continue)
   └───────────────────────────────────────────────────────────────┘
                                 │ state (h,z)
        ┌────────────────────────┴──────── BEHAVIOR (C4 + C5) ───────────┐
        │  imagine H steps with the actor, score with reward + value:    │
        │     actor C5:  a ~ π(a|s)        critic C4: v(s) ≈ λ-return     │
        └─────────────────────────────────────────────────────────────────┘
```

---

## Latent state & notation

At each step the model state is a pair:

| symbol | name | shape (ours) | role |
|---|---|---|---|
| `h_t` | deterministic (GRU) state | `(deter=256)` | the **memory** — folds in the whole past + last action |
| `z_t` | stochastic state | `32 groups × 32 classes` → flat `1024` | the **"what's here now"**, a sample of categorical latents |
| `s_t = (h_t, z_t)` | full state | — | what the heads/decoder/actor/critic read |
| `feat_t = [h_t ; z_t]` | features | `(1280)` | concat fed to decoder / heads / actor / critic |

`z_t` is **categorical with straight-through gradients** (32 one-hot vectors of
size 32), mixed with 1% uniform (`unimix`). Other symbols: `x_t` = image,
`a_t` = action (one-hot over 21), `r_t` = reward, `c_t` = continue (`1−terminal`),
`γ` = discount, `λ` = return mixing, `sg` = stop-gradient.

---

## World Model — C3a + C3b + C3c

The three world-model components share one forward pass and one combined loss,
so they're presented together but each network/loss term is tagged.

### Networks

| tag | network | maps | implements |
|---|---|---|---|
| **C3a** | encoder `q_enc` | image → `embed (1024)` | `ConvEncoder` (5 stride-2 convs, 128→4) |
| **C3a** | decoder `p(x\|s)` | `feat → image` | `ConvDecoder` (mirror) |
| **C3b** | sequence model `f` | `(h_{t-1},z_{t-1},a_{t-1}) → h_t` | GRUCell |
| **C3b** | prior `p(z_t\|h_t)` | `h_t → logits(z_t)` | `prior_net` MLP — **the dynamics** |
| **C3b** | posterior `q(z_t\|h_t,x_t)` | `(h_t, embed_t) → logits(z_t)` | `post_net` MLP |
| **C3c** | reward head `p(r_t\|s)` | `feat → reward` (symlog) | MLP |
| **C3c** | continue head `p(c_t\|s)` | `feat → logit` | MLP |

The **prior** (C3b) and **posterior** (C3b) both produce `z_t`; the prior predicts
it *without* the frame (imagination), the posterior *with* it (training). Training
them to agree is the KL term.

### Two forward passes (C3b)

**Observe (closed-loop, posterior)** — `rssm.observe` over a real sequence:
```
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
z_t ~ q(z_t | h_t, embed_t)            # uses the real frame
```
`is_first` resets `(h,z,a)` to zero at episode boundaries.

**Imagine (open-loop, prior)** — `rssm.imagine`:
```
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
z_t ~ p(z_t | h_t)                     # NO frame — predicted from h alone
```
**Crucial:** imagination stays in latent space — `z` is fed back, never a decoded
pixel. The decoder (C3a) is a read-out, never an input.

### World-model update rule (combined loss)

Trained on a batch of `(B, L)` real subsequences:
```
L_wm =  β_pred · L_pred   +   β_dyn · L_dyn   +   β_rep · L_rep
        (β_pred=1)            (β_dyn=0.5)          (β_rep=0.1)
```

**Prediction loss `L_pred`** — heads reconstruct the data from `feat_t`:
```
L_recon  = Σ_pixels (decoder(feat_t) − x_t)²       # C3a — SUMMED over pixels, mean B,L
L_reward = ( reward_head(feat_t) − symlog(r_t) )²  # C3c
L_cont   = BCE( continue_head(feat_t), c_t )       # C3c
L_pred   = L_recon + L_reward + L_cont
```
> ⚠️ **Gotcha 1 (C3a):** `L_recon` must be **summed** over pixels, not meaned —
> meaning makes it ~50,000× too weak vs KL → KL dominates → blurry dreams.

> ⚠️ **Gotcha 2 (C3c):** `r_t`/`c_t` use the **reward-into-observation**
> convention so `feat_t` (which has seen `a_{t-1}`) can predict them. The
> off-by-one tanked reward correlation 0.905→0.147.

**KL losses `L_dyn`, `L_rep` (C3b)** — align prior & posterior, with **free bits**
(`max(1,·)` nats) so KL never collapses:
```
L_dyn = max(1,  KL[ sg(q(z_t|h_t,x_t))  ‖  p(z_t|h_t) ] )   # train PRIOR → posterior
L_rep = max(1,  KL[ q(z_t|h_t,x_t)  ‖  sg(p(z_t|h_t)) ] )   # train POSTERIOR → prior
```
`L_dyn` is **the** training signal for the dynamics used in imagination (the
prior). KL balancing (0.5 / 0.1) lets the prior chase the posterior faster.

> Our impl: manual categorical KL + manual straight-through (version-proof).
> Canonical DreamerV3 also uses symlog **twohot** regression for reward/value; we
> use plain symlog-MSE for reward.

---

## Critic — C4

The critic `v_ψ(s)` predicts the **expected discounted return** from a state.

### The target: λ-returns

Over an imagined rollout `s_t … s_{t+H}` with predicted rewards `r̂` (C3c) and
continues `ĉ` (C3c):
```
R^λ_t   = r̂_t + γ ĉ_t [ (1−λ) v(s_{t+1}) + λ R^λ_{t+1} ]
R^λ_{H} = v(s_H)                         # bootstrap at the horizon
```
- `γ = 0.997` (discount), `λ = 0.95` (mix of 1-step bootstrap vs Monte-Carlo).
- `ĉ_t` **multiplicatively zeroes future reward past a predicted death** — why C3c
  matters for value.

### Update rule (C4)

```
L_critic = Σ_t ( v_ψ(s_t) − sg(R^λ_t) )²
```
Stabilizers to include: a **slow EMA copy** of the critic for the targets + a
small regularizer toward it. (Canonical uses symlog-twohot for `v`.)

---

## Actor — C5

The actor `π_θ(a|s)` (categorical over 21 actions) maximizes the λ-returns of its
own imagined rollouts.

### Imagination rollout
```
start from every posterior state s_t in the replay batch (≈ B·L starts)
for k in 0..H-1:   a ~ π_θ(·|s);  s ← prior_step(s,a) [C3b];  collect r̂, ĉ [C3c], v [C4]
```
Thousands of short (H=15) dreamed trajectories per batch. No emulator, no pixels.

### Update rule (C5 — discrete → REINFORCE + entropy)
```
A_t = (R^λ_t − v(s_t)) / max(1, S)          # advantage, return-normalized
L_actor = Σ_t  −[  sg(A_t) · ln π_θ(a_t|s_t)   +   η · H(π_θ(·|s_t)) ]
```
- REINFORCE: push up log-prob of actions that beat the critic baseline `v(s_t)`.
- `S` = EMA of `[5th,95th]` percentile range of returns → **return normalization**.
- `η · H(π)` = **entropy bonus** → keeps exploring.

The actor never sees a reward label — it chases the critic's value, grounded in
C3c. The whole behavior loop is reward-shaped imagination.

---

## The interleaved training step — C6

Every gradient step does all three (like a GAN's two nets):
```
1.  batch ← replay.sample(B, L)                         # real subsequences
2.  posts, priors ← observe(batch)                      # C3b closed-loop
3.  update WORLD MODEL on L_wm                           # C3a+C3b+C3c
4.  imagine H steps with the actor from posts (detach)  # C3b open-loop dreams
5.  compute λ-returns from r̂, ĉ, v                       # C3c + C4
6.  update CRITIC on L_critic                            # C4
7.  update ACTOR on L_actor                              # C5
```
World-model grads don't flow into actor/critic (states detached before
imagination) and vice-versa.

**Offline pretrain (Phase 0):** run steps 1–3 alone on the MC traces *before* any
behavior learning — a warm world model most Dreamer setups can't get.

---

## Deploy (reflex)

No search, no imagination at play time:
```
keep a running latent state; each real step:
  h ← f(h,z,a_prev) [C3b];  z ~ q(z|h, encoder(frame)) [C3a/C3b];  a ~ π_θ(·|h,z) [C5]
```
Planning is already baked into the actor by all that imagined training.

---

## Hyperparameters

| symbol | meaning | value |
|---|---|---|
| `deter` | GRU state size | 256 |
| `stoch × classes` | categorical latent | 32 × 32 |
| `unimix` | uniform mix | 1% |
| free bits | KL floor (nats) | 1.0 |
| `β_pred,β_dyn,β_rep` | loss weights | 1, 0.5, 0.1 |
| `H` | imagination horizon | 15 steps (≈0.75s @ 20Hz) |
| `γ` | discount | 0.997 |
| `λ` | return mixing | 0.95 |
| `η` | actor entropy weight | ~3e-4 |

---

## Notes & open issues

- **Decision rate is 20 Hz** (`SKIP=3`), so `H=15` ≈ 0.75 s of real play.
- **Sprite blindness is cosmetic (C3a).** The decoder under-renders the
  player/enemies (background-dominated MSE), but the *latent* encodes them
  (linear probe: player ~7px; reward head r=0.905). The actor reads the latent.
- **⚠️ Open issue — the prior (C3b) under-models slow motion.** Open-loop
  imagination *freezes* from a static start and stalls over long horizons; it
  tracks motion well only over ~15 steps from already-active states. Fine for
  C5's regime, but weaker than ideal. Fixes: longer training, larger `deter`, or
  training the prior on its own multi-step rollouts (scheduled sampling). The C3b
  gate must add a **motion-matching** check (dream motion ≈ actual motion).
- **Continue calibration is soft (C3c)** — add class-weighted BCE + more deaths.

See [[dreamer-contra-build]] for build status; gate scripts are `verify_*.py`.
```
