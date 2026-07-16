# Dreamer-on-Frozen-Encoder — Verification Log

Step-by-step record of *what we verified*, *how*, and *what the numbers mean*, as
we build the world model on a pretrained + frozen encoder. Each step states a
claim, the test, the result, and how to reproduce it. Companion to
[DREAMER_DESIGN.md](DREAMER_DESIGN.md) (which explains the architecture); this file
is the evidence trail.

---

## Step 1 — the latent contains the information the agent needs

**Claim.** The frozen encoder's 1024-d embedding encodes the *task-relevant* content
of a frame — the positions of the player, enemies, and both kinds of bullets — and
it does so on **unseen playthroughs**, not just the training ones.

**Why this matters.** Everything downstream (RSSM dynamics, reward head, actor) reads
the latent, never the pixels. If entities aren't in the latent, the agent is blind to
the things that kill it — pixel reconstruction alone won't put them there, because a
2px bullet is invisible to background-dominated MSE. So this is the first gate.

**How we test it.** During pretraining ([pretrain_ae.py](pretrain_ae.py)) an
`EntityHead` decodes four `32×32` occupancy heatmaps from the embedding — one per
class: `player · player_bullets · enemies · enemy_bullets`. Ground truth comes from
RAM ([contra/entities.py](../contra/entities.py)). We hold out **2 whole traces**
(playthroughs never seen in training) and measure the head on them. Because the head
is a *deterministic function of the embedding*, accurate heatmaps on held-out data
prove the information is in the embedding **and** generalizes.

### What the numbers mean

Held-out eval at the end of training — **one metric, the same for all four classes**:

```
heatmap MSE  [player 0.0002  player_bullets 0.0022  enemies 0.0004  enemy_bullets 0.0003]
```

**Per-class heatmap MSE.** For each class it is the **mean squared error** between the
predicted heatmap and the ground-truth heatmap, averaged over every cell and every
held-out frame:

    MSE_c = mean over (frames × 1024 cells) of ( pred_cell − target_cell )²

Two things to be precise about (easy to misread):

- It's **squared** error (L2), **not** L1 / absolute distance.
- The target is **not** a binary {0,1} vector. It's a continuous **Gaussian blob**
  (peak 1.0, decaying smoothly to 0 around each entity), and the prediction is a
  sigmoid in `[0,1]`. So it's L2 over two continuous `32×32` fields, normalized by the
  1024 cells and averaged over held-out frames — one number per class.

So `player = 0.0002` means: over all 1024 cells of all held-out frames, the predicted
player-occupancy differs from the target by `0.0002` in mean squared error.
Equivalently `√0.0002 ≈ 0.014` → the average per-cell occupancy is off by ~1.4% of the
`[0,1]` range. These started near `0.26` at step 1 (an untrained sigmoid outputs ~0.5
against a mostly-zero target) and dropped **~3 orders of magnitude**.

Reading the whole bracket:

| class | held-out MSE | note |
|---|---|---|
| player | 0.0002 | easiest: one large, always-present sprite |
| enemies | 0.0004 | well localized |
| enemy_bullets | 0.0003 | well localized — the class that matters most for survival |
| player_bullets | 0.0022 | **weakest** — tiny, fast, many, blink in/out; least survival-critical |

### Calibrating the MSE scale (what a value *means* in cells / pixels)

To make the numbers concrete: take one Gaussian blob (σ=1, peak 1.0) as truth and an
identical blob offset from it as the prediction, and compute the MSE over the 1024
cells:

| offset (dx,dy) | center distance | MSE |
|---|---|---|
| (0,0) | 0.00 | 0.000000 |
| (0,1) | 1.00 | 0.00136 |
| (1,1) | 1.41 | 0.00242 |
| (0,2) | 2.00 | 0.00388 |
| (2,2) | 2.83 | 0.00531 |
| (3,3) | 4.24 | 0.00607 |
| *full miss (predict nothing)* | — | 0.00307 |

Two things this shows:

- **MSE saturates at ~2× a full miss.** A completely missed blob costs ~0.003 (error
  only at the true spot). Two *non-overlapping* blobs cost error at **both** the true and
  the predicted spot, so MSE climbs to ~0.006 once they stop overlapping. Note (2,2) =
  0.0053 is already **worse than a full miss** — confidently placing the blob in the
  wrong cell is worse than predicting nothing.
- **Our numbers are in the sub-cell regime.** They sit far below even (0,1) = 0.00136,
  so the model is off by a *fraction* of a cell. Inverting the single-blob curve:

  | reported MSE | ≈ blob offset | in pixels (×8) |
  |---|---|---|
  | player 0.0002 | ~0.36 cell | ~3 px |
  | enemies 0.0004 | ~0.52 cell | ~4 px |
  | player_bullets 0.0022 | ~1.33 cell | ~11 px |

  The player's `0.0002 → ~3px` matches the earlier `peak_err ≈ 2px` cross-check.

Caveat: this offset↔MSE mapping is exact only when a frame has **exactly one** entity of
the class (true for the single-instance player). For enemies/bullets the per-frame error
is roughly the **sum** over blobs, so a low value there means "a few entities each
localized well," not literally "one entity off by X." Reproduce: put two σ=1 Gaussian
blobs on a 32×32 grid offset by δ and take the mean squared difference over the 1024
cells.

**Verdict.** The latent carries entity positions and generalizes to unseen traces —
all four classes reach low heatmap MSE (player/enemies/enemy-bullets ~2–4e-4).
`player_bullets` (2.2e-3) is the one soft spot, and it's the least survival-critical
class. Gate: **PASS.**

### Caveats

- This is **nonlinear** decodability (the head is a small net) — sufficient, because
  the RSSM/actor that consume the latent are also nonlinear. A linear probe would be a
  stronger "cleanly/linearly present" claim; we chose the heatmap gate instead because
  it's per-class and visually checkable (see [verify_ae.py](verify_ae.py) `--ckpt`).
- Scope: **level-1 winning traces only.** Off-distribution states (deaths, other
  levels) are not exercised here.

### Reproduce

```bash
# train (produces tmp/dreamer/ae_pretrained.pt and the held-out metrics above)
python -m dreamer.pretrain_ae --level 1 --steps 8000

# eyeball it: input | recon | predicted-heatmap overlay, 20 frames along one trace
python -m dreamer.verify_ae --ckpt tmp/dreamer/ae_pretrained.pt --level 1 --k 20
```

---

## Step 2 — the RSSM can dream (C3b)

_TODO — verify the RSSM learns action-conditioned dynamics on the frozen latent:
`closed_mse ≈ 0` (posterior tracks reality) and `open_motion ≈ real_motion` (the
15-step open-loop dream moves like the real game, not frozen)._

```bash
python -m dreamer.verify_rssm --enc_ckpt tmp/dreamer/ae_pretrained.pt --train_traces 8 --steps 6000
```
