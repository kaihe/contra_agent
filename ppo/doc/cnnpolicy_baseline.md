# Actor model-structure experiments (BC warm-start)

Status: **living doc** — append a row per experiment.
Related: [cnn_pretrain_design.md](../cnn_pretrain_design.md) (overall warm-start design).

## Goal & scope

Find a model structure whose **actor** clones the *decisive* actions (fire, jump),
not just the trivial majority (move-right / up-right). We **track actor
performance only** — the critic is excluded on purpose: its `explained_variance`
is an easy, optimistic metric here (returns are a smooth function of visible
level position, with no death/off-path variance in the all-wins data), so it
tells us nothing about model structure.

**Baseline model = the current SB3 `CnnPolicy`** (NatureCNN: 3 conv → 512 FC →
single `Discrete(21)` softmax head). All experiments vary *structure* against the
same dataset and training protocol below.

## Dataset basics

Source: **41 level-1 pruned win traces** (`tmp/mc_trace/level1/`), replayed through
the real `ContraWrapper` → **61,372 steps**; all 41 reproduce a win.
Obs `(84,84,3)` stack=3, reward config `stable`, gamma 0.99.
Return-to-go range: min −306.2, max 64.1, mean 7.2.

**Action distribution** (ranked by count, descending). Codes use the canonical
action-space names; the action column is the decoded button combo.

| rank | code | action | count | freq |
|---|---|---|---|---|
| 1 | R | Right | 22,397 | 0.365 |
| 2 | UR | Up+Right | 21,373 | 0.348 |
| 3 | _ | idle (no-op) | 8,545 | 0.139 |
| 4 | URF | Up+Right+**Fire** | 3,293 | 0.054 |
| 5 | RF | Right+**Fire** | 2,441 | 0.040 |
| 6 | D | Down | 1,057 | 0.017 |
| 7 | RJ | Right+**Jump** | 514 | 0.008 |
| 8 | U | Up | 402 | 0.007 |
| 9 | DF | Down+**Fire** | 330 | 0.005 |
| 10 | F | **Fire** | 266 | 0.004 |
| 11 | URJ | Up+Right+**Jump** | 230 | 0.004 |
| 12 | L | Left | 221 | 0.004 |
| 13 | UF | Up+**Fire** | 202 | 0.003 |
| 14 | J | **Jump** | 68 | 0.001 |
| 15 | LJ | Left+**Jump** | 11 | 0.000 |
| 16 | UL | Up+Left | 8 | 0.000 |
| 17 | LF | Left+**Fire** | 5 | 0.000 |
| 18 | DJ | Down+**Jump** | 4 | 0.000 |
| 19 | ULF | Up+Left+**Fire** | 3 | 0.000 |
| 20 | ULJ | Up+Left+**Jump** | 1 | 0.000 |
| 21 | UJ | Up+**Jump** | 1 | 0.000 |

**Imbalance summary:**
- Top 3 (R, UR, idle) = **85.2%** of all steps; R+UR alone = **71.3%**.
- Any action carrying **Fire** = **6,540 steps (10.7%)**, spread over 7 combos.
- Any action carrying **Jump** = **829 steps (1.35%)**; no single jump combo exceeds 0.8%.

This is the imbalance the actor must overcome: the decisive Fire/Jump actions are
a thin tail, while the head is dominated by rightward movement.

## Key metrics (actor only)

We score the actor by how well it reproduces the **decisive Fire/Jump actions**,
not aggregate accuracy. Metrics computed on the full BC dataset (61,372 steps):

- **fire_recall / jump_recall** — of demo steps that pressed the button (via *any*
  action carrying it), fraction the prediction also presses. Primary.
- **fire_prec / jump_prec** — of predicted button-presses, fraction the demo agreed.
  Pairs with recall so "always-press" can't game it.
- **macro_recall** — mean per-action recall over the 21 actions (balanced accuracy);
  single imbalance-robust headline.
- **accuracy** — top-1 over 21 actions; *context only* (~0.66 = the R/UR majority).

### Baseline — `CnnPolicy` (NatureCNN, single Discrete(21) softmax)

Protocol: 40 epochs, Adam lr 3e-4, batch 512, class weight `inv_sqrt`, ent_coef 0.
Checkpoint `tmp/ppo/pretrain/level1_bc.zip`.

| metric | value |
|---|---|
| **fire_recall** | **0.005** (prec 0.153, n=6,540) |
| **jump_recall** | **0.040** (prec 0.171, n=829) |
| macro_recall | 0.170 |
| accuracy | 0.658 |

Per Fire/Jump-action self-recall (predict that *exact* action when the demo did):

| action | n | self-recall |
|---|---|---|
| URF (Up+Right+Fire) | 3,293 | 0.000 |
| RF (Right+Fire) | 2,441 | 0.000 |
| RJ (Right+Jump) | 514 | 0.041 |
| DF (Down+Fire) | 330 | 0.003 |
| F (Fire) | 266 | 0.041 |
| URJ (Up+Right+Jump) | 230 | 0.000 |
| UF (Up+Fire) | 202 | 0.084 |
| J (Jump) | 68 | 0.000 |
| all rarer Fire/Jump combos | ≤11 each | 0.000 |

**Read:** the baseline actor essentially never fires (recall 0.5%) and rarely jumps
(4%), even on the highest-count Fire combos (URF/RF at 0.000). It collapses to the
R/UR/idle head. This is the bar every model structure must beat — on Fire/Jump
recall, not on the 0.658 accuracy.

## Experiments

Same protocol as the baseline (40 epochs, lr 3e-4, batch 512, `inv_sqrt`,
ent_coef 0); one variable changed per row. Metrics from the final checkpoint.

| # | change | macro_recall | fire_recall (prec) | jump_recall (prec) | acc |
|---|---|---|---|---|---|
| 0 | **baseline** 84×84 CnnPolicy | 0.170 | 0.005 (0.15) | 0.040 (0.17) | 0.658 |
| 1 | **192×192** input (same model) | **0.334** | 0.000 (0.14) | **0.330** (0.27) | 0.729 |

### Exp 1 — larger input resolution (192×192)

**Result: jump recall 0.040 → 0.330 (8×), macro_recall 0.170 → 0.334 (2×); fire unchanged (~0).**

This cleanly splits the two decisive actions by *observability*:
- **Jump is observation-determined** (a gap/obstacle ahead). 84×84 was washing out
  those small/thin cues; 192×192 recovers them, so the actor finally learns to
  jump. Confirmed: jump_recall rose with *no* other change.
- **Fire is timing-determined** (the kept presses sit on the gun's cooldown
  cadence, not a pixel cue), so more resolution does nothing — fire_recall stays
  ~0 as predicted. Resolution is not a lever for fire.

Note: jump_recall was noisy across late epochs (0.14–0.46) and slow to appear
(still 0 at epoch 6, climbing only after ~epoch 25) — so judge resolution on the
final checkpoint, not early epochs.

**Takeaway:** input fidelity is a real lever for the *visually-grounded* decisive
action (jump). Fire still needs the label fix (design A/B), not a model change.
A warm-start checkpoint at 192 (`tmp/ppo/pretrain/level1_bc_r192.zip`) would
require PPO training to also run at 192 to stay resume-compatible.
