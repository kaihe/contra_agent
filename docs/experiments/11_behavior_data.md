---
layout: default
title: "11. Behavior Data from Search"
parent: Content
nav_order: 11
---

# 11. Behavior Data from Search
{: .no_toc }

**Date:** 2026-04-27 · **Type:** Search / Synthetic Data
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## The Goal

Chapter 10 showed that the Monte Carlo search in `synthetic/mc_search.py` can beat any single level of Contra. One win is a proof of concept; it is not a dataset. To train a neural player by behavior cloning we need *many* winning traces per level — thousands of them — so the network sees the corridor solved many different ways rather than memorising one brittle path.

So we ran the search as a factory. Each invocation stops at a level-up (`--goal level_up`), saves the committed action sequence to an NPZ, and starts over from a fresh random seed. Run that loop across all eight levels for a couple of weeks of spare machine time and you accumulate a library of independent win traces.

The result, in `synthetic/mc_trace/`:

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **Total** |
|---|---|---|---|---|---|---|---|---|---|
| Win traces | 1716 | 897 | 679 | 923 | 821 | 881 | 934 | 870 | **7721** |

Level 1 is over-represented because it is the cheapest to search and was used to shake out the pipeline; Level 3 (the vertical waterfall climb) is the rarest because it is the hardest corridor and the slowest to solve.

## Statistics of the Win Traces

The length of each trace — the number of committed steps from level start to level-up. One step is `SKIP = 3` NES frames, so 20 steps ≈ 1 second of game time. Measured across all 7721 traces:

| Level | Min | Mean | Max |
|---|---|---|---|
| 1 | 1238 | 1494 | 1882 |
| 2 | 1477 | 1644 | 2231 |
| 3 | 1742 | 2860 | 3989 |
| 4 | 2483 | 2990 | 4126 |
| 5 | 2172 | 2421 | 3085 |
| 6 | 1475 | 1681 | 3559 |
| 7 | 1704 | 1943 | 2376 |
| 8 | 1748 | 2110 | 2993 |

The two top-down levels, **3 and 4**, are the longest — the player walks *into* the screen, which the search covers more slowly than a side-scroll. The long tails (Level 3 up to 3989, Level 6 up to 3559) are traces that backtracked heavily before finally committing through a chokepoint.

## Compute Cost

How much random flailing does one win trace actually cost? The search now reports `total_sampled_actions` — the number of random actions stepped through the emulator across every rollout — so we can read it off directly. Below is one representative search per level: the committed trace it produced, how many random actions it sampled to get there, and the wall-clock time on this machine (32 cores).

| Level | Win-trace steps | Random actions sampled | Wall-clock time |
|---|---|---|---|
| 1 | 1502 | 668,564 | 173.7 s |
| 2 | 1725 | 275,819 | 148.4 s |
| 3 | 4032 | 1,124,269 | 268.9 s |
| 4 | 3052 | 518,336 | 149.5 s |
| 5 | 2464 | 786,997 | 254.1 s |
| 6 | 1707 | 1,584,613 | 433.8 s |
| 7 | 2223 | 2,441,057 | 508.0 s |
| 8 | 2337 | 626,507 | 148.7 s |
| **Total** | **19,042** | **≈ 8.03 M** | **≈ 2085 s (~35 min)** |

We log these numbers to track the total compute cost of beating the game.