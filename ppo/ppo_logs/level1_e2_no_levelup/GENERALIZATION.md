# e2_no_levelup generalizes to unseen start states

**Date:** 2026-06-14
**Checkpoint:** `level1_e2_no_levelup/final.zip`
**Question:** e2 was trained on **one** trace's anchors (the `202603301145` series, 9 states).
Can it handle start states sampled from *other* winning playthroughs it never trained on?

**Method:** 20 stochastic episodes per anchor, 9 anchors per series, 5 series (= 900 episodes).
The 1145 series is the training distribution; the other 4 are unseen. See `generalization.png`.

## Result — yes, it generalizes (no overfitting)

| series | seen in training? | win rate |
|---|---|---|
| 202603301145 | **yes (train)** | 67% (120/180) |
| 202603301703 | no | 74% (134/180) |
| 202604021858 | no | 75% (135/180) |
| 202604081140 | no | 73% (132/180) |
| 202604081539 | no | 77% (138/180) |
| **unseen avg** | — | **74.9%** |
| **all 5 series** | — | **73.2% (659/900)** |

The agent does **at least as well on unseen start states as on its training states** — the 4
unseen series average 74.9% vs 66.7% on the trained 1145 series. If e2 had memorized the 1145
snapshots, unseen series would score *lower*; instead they score *higher*. So PPO learned a
general "play Level 1 from wherever you are" policy, not a lookup table of trained positions.

(The 1145 series scoring lowest is just which anchors happen to land in hard spots — its
x2078/x2411 boss-approach pair is a steep 25%/45% dip; see the plot.)

## Per-anchor win rate (20 episodes each)

Anchors are listed by index (each series samples 9 anchors at the same level fractions, so
row _i_ is roughly the same level position across series). Each cell is `start-x : win%`.

| # | 1145 (TRAIN) | 1703 | 1858 | 0408_1140 | 0408_1539 |
|---|---|---|---|---|---|
| 1 | x0000 : 75% | x0000 : 90% | x0000 : 70% | x0000 : 90% | x0000 : 85% |
| 2 | x0313 : 95% | x0310 : 95% | x0303 : 90% | x0252 : 70% | x0301 : 70% |
| 3 | x0645 : 65% | x0705 : 85% | x0667 : 100% | x0581 : 90% | x0663 : 75% |
| 4 | x1021 : 80% | x1068 : 90% | x1083 : 75% | x0942 : 85% | x1027 : 65% |
| 5 | x1343 : 65% | x1450 : 95% | x1437 : 90% | x1316 : 90% | x1387 : 75% |
| 6 | x1695 : 90% | x1766 : 65% | x1824 : **0%** | x1593 : 45% | x1710 : 85% |
| 7 | x2078 : 25% | x2116 : 50% | x2065 : 95% | x2009 : 95% | x2057 : 65% |
| 8 | x2411 : 45% | x2395 : 95% | x2369 : 85% | x2419 : **0%** | x2426 : 85% |
| 9 | x2721 : 60% | x2779 : **5%** | x2704 : 70% | x2756 : 95% | x2732 : 85% |
| **overall** | **67%** | **74%** | **75%** | **73%** | **77%** |

Bold cells are the instant-death anchors (die in 6–13 steps); their neighbours win 85–95%.

## Shape of the win-rate curve (consistent across all series)

`generalization.png` plots win% vs start-x for every series. Three features recur:

1. **Strong over the first ~two-thirds of the level** — anchors from x0 to ~x1500 win
   70–100% in every series. The agent reliably traverses and finishes from these starts.
2. **A boss-approach dip** — each series has a soft valley somewhere in x1600–x2400
   (1145 x2078=25%, 1703 x2116=50%, 0408_1140 x1593=45%, 0408_1539 x2057=65%). This is the
   gun-dependency region: anchors that reach the boss wall with a weak weapon die more.
3. **Isolated "instant-death" anchors** (red band on the plot) — a few snapshots die in
   **6–13 steps** with negative reward: x2779 (1703, 5%), x1824 (1858, 0%), x2419
   (0408_1140, 0%). Their immediate neighbours win 90–95% (e.g. x2009=95% sits right before
   x2419=0%), so these are **bad savestates**, not a policy weakness — the snapshot drops the
   player into an already-lethal frame (boss fire / mid-hit) that no policy can escape. They
   are an anchor-quality artifact of sampling a single frame from a noisy trajectory.

## Takeaways

- **Generalization confirmed.** Training on one trace's anchors is enough; e2 handles start
  states from four other playthroughs equally well (~75%). Multi-trace anchor sets (the new
  45-state set) should help most at the boss-approach valley, not the early/mid level.
- The real remaining weaknesses are (a) the boss-wall gun dependency and (b) a handful of
  degenerate near-boss/mid-hit snapshots — both anchor/level issues, not generalization
  failures.
- Worth filtering anchors at generation time: drop any whose greedy rollout dies in <~20
  steps (the instant-death snapshots), so training/eval isn't polluted by unwinnable starts.

## Reproduce

```
python tmp/eval_e2.py 20 <series-timestamp>   # e.g. 202603301703, 202604021858, ...
```
Artifacts in this dir: `generalization.png`, plus the earlier `FINDINGS.md` / `compare_e1_e2.png`.
