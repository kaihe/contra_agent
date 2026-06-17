---
layout: default
title: "10. Search to Win the Game"
parent: Content
nav_order: 10
---

# 10. Search to Win the Game
{: .no_toc }

**Date:** 2026-03-31 · **Type:** Search / Synthetic Data
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## The Result

All eight levels of Contra can be beaten by a mindless random search. No neural network, no hand-crafted policy, no RL training loop. Just throw random button presses at the game, keep the ones that score well, throw away the ones that don't, and repeat — level after level until the final boss is dead.

That sounds absurd. It works.

The two ingredients that make it possible are the event system from Chapter 9, which turns RAM state transitions into a dense reward signal, and the fixed anchor technique, which ensures that progress is never lost. Put them together and what looked like an impossibly large search problem decomposes into a chain of manageable short-horizon problems.

## The Algorithm

The search lives in `synthetic/mc_search.py` and is built around a single loop. At any point in time there is a **committed prefix** — a sequence of actions that have been permanently accepted — and a **committed state**, the emulator snapshot after replaying that prefix. The loop repeats three steps:

**Step 1 — Monte Carlo lookahead.** From the committed state, `N` independent random rollouts are run in parallel, each of length `L` actions. A rollout stops early and is flagged as a death if the player-die event fires. The rollout with the highest cumulative reward is selected as the candidate.

**Step 2 — Commit or force rewind.** If the best rollout ended in death, nothing is committed and the stale counter is immediately bumped to the patience threshold to force a rewind. Otherwise, a random prefix of the winning rollout (between half its length and its full length) is replayed step-by-step and appended to the committed prefix. The randomised commit length prevents the search from always picking the same greedy trajectory.

**Step 3 — Progress check and backtrack.** If the cumulative reward has not improved for `patience` consecutive commits, the algorithm rewinds: it picks a random number of steps between 1 and `max_rewind` and truncates the committed prefix back to that point. This lets the search escape local plateaus — corridors of actions that look fine but lead to an unavoidable death further ahead.

In code, the core of the loop looks like this:

```python
# 1. Run N parallel rollouts from the current committed state
rollout_results = pool.map(_worker_rollout, [task] * rollouts)

# 2. Pick the best; commit a random prefix if it survived
best_seq, best_reward, best_died = select_best(rollout_results)
if best_died:
    stale_count = patience          # force backtrack
else:
    commit_n = np.random.randint(n // 2, n + 1)
    commit(best_seq[:commit_n])

# 3. Rewind on stagnation
if stale_count >= patience:
    rewind_back = np.random.randint(1, max_rewind + 1)
    truncate_committed_prefix(rewind_back)
```

The committed prefix, the emulator snapshot at each step, and the cumulative reward are all kept in parallel arrays, so rewinding is just slicing them shorter.

## The Action Bigram Prior

Pure uniform-random button mashing is functional but wasteful. Many action transitions are physically nonsensical — switching direction mid-jump, mashing fire while climbing a ladder — and human players never produce them. Chapter 8's human recordings let us build a **bigram prior**: a 21×21 matrix (one row/column per action in the curated `baseline.yaml` set) where entry `[i, j]` holds the empirical probability of action `j` following action `i` in human play.

Critically, the right prior is different for each level. Contra's eight levels span three fundamentally different layouts:

- **Side-scrolling** (Levels 1, 5, 6, 7, 8) — the player must constantly push right. Right dominates the action distribution.
- **Top-down** (Levels 2, 4) — the player walks into the screen. Up is the dominant direction.
- **Vertical climb** (Level 3, the waterfall) — the player scales a waterfall. Jump has a heavy density throughout.

A single uniform prior ignores all of this. A rollout on Level 3 that never presses jump will run straight into the waterfall and die; a rollout on Level 2 that keeps pushing right will walk in circles. Fitting a separate bigram per level from the human recordings captures these layout-specific patterns automatically, so the random search spends its budget on plausible trajectories rather than physically impossible ones.

Each rollout worker samples from this prior instead of uniformly:

```python
prev_idx = int(np.random.choice(ALL_ACTIONS, p=prior[prev_idx]))
```

The prior is stored in `synthetic/action_bigram.npz` with one matrix per level. If the file is absent the search falls back to uniform sampling — it still works, just slower.

## A Shared Action Space and Frame Skip

Running the search across all eight levels surfaced something useful beyond the winning traces themselves: a single action space and frame-skip setting that is the most efficient configuration for finding a win path everywhere. It is checked in as [`contra/action_configs/baseline.yaml`](../../contra/action_configs/baseline.yaml).

Two choices matter:

- **A curated 21-action set.** Rather than enumerate all 2⁹ raw NES button combinations, the search uses the 21 directional + jump/fire combinations that actually do anything in Contra — the eight directions, each with optional jump or fire, plus the no-op. This is small enough that even uniform sampling explores meaningfully, yet expressive enough to clear every level.
- **`skip = 3`.** Each decision is held for three NES frames. This was the sweet spot: short enough that the agent retains fine control for boss dodging and platforming, long enough that a rollout of `L` actions covers enough game time to make Monte Carlo lookahead worthwhile. Larger skips lose precision on the hard sections; smaller skips waste search budget re-deciding identical actions.

Crucially, **this same setup should be optimal for PPO too.** Search and RL are solving the same underlying control problem — same emulator, same levels, same physics — and an action space and time resolution that make the win path *findable* by random search are exactly what make it *learnable* by a policy gradient. A coarser action space throws away reachable behaviors; a finer one inflates the policy's branching factor for no benefit. By having both `mc_search` and PPO load `baseline.yaml`, the win paths discovered by search are reproducible by the trained policy by construction, and we avoid silently optimizing two different problems.

## Running It

```bash
python synthetic/mc_search.py --level 1 --rollouts 64 --rollout-len 48 --max-rewind 30 --max-time 600 --max-actions 6000
```

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `--rollouts` | 64 | Rollouts per step |
| `--rollout-len` | 48 | Actions per rollout (~2.4 s of game time) |
| `--max-rewind` | 30 | Max steps to undo on backtrack |
| `--max-actions` | 6000 | Abandon the trace if committed actions exceed this |
| `--workers` | all CPUs | Parallel worker processes |
| `--goal` | `level_up` | `level_up` stops at level transition; `game_clear` runs to the end |
| `--reward-config` | `stable` | Reward config under `contra/reward_configs/` |

The parameters above — rollout count, rewind range, action budget — are admittedly ad hoc. They are almost certainly not optimal. But on a 32-core laptop, every level falls within about a minute of search time, which is more than fast enough.

Here is the search agent clearing Level 6. It moves with an almost casual confidence — smooth, unhurried, and completely effortless-looking.

<video src="../../assets/recordings/ch10_mc_search_level6.mp4" controls width="100%"></video>

With winning traces in hand for all eight levels, the next step is to use them as behavior cloning data to train a neural network player.
