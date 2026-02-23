---
layout: default
title: "Monte Carlo Search"
parent: Experiments
nav_order: 4
---

# Experiment: Monte Carlo Search and Playfun
{: .no_toc }

**Date:** 2026-02-23 · **Model:** `None (Agentic Search)`
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## 1. Validating Action Space
- **Goal:** Quick algorithmic check to validate action space before costly RL training.
- **Why:** Avoid bugs like the "B-button release" issue found in prior experiments.

## 2. Playfun vs. Monte Carlo
- **Concept:** Based on Tom Murphy's greedy "Learnfun/Playfun" algorithm.
- **Problem:** Greedy 1-step search fails in Contra due to delayed rewards (e.g., bullet hits 60 frames later).
- **Solution:** Swapped exhaustive 1-step search for **256 random 16-action Monte Carlo rollouts**.
- **Backtracking:** If all 256 futures result in death, it logs the state as an inescapable trap and dynamically **rewinds time** to find an uncorrupted timeline.

## 3. Deep Rollouts
- **Lookahead:** Rollouts consist of 16 actions (skipped 4 frames each), giving the agent a **64-frame lookahead** window to spot incoming enemy bullets.

## 4. Why it Fails at the Boss
- **Early Game:** Excels because of constant score/x-scroll micro-rewards.
- **Boss Fight:** Fails because boss targets have multi-HP. A random 16-action sequence rarely destroys a boss target, so the sequence naturally receives **no immediate score reward**. The agent cannot logically deduce that shooting the boss is working.

## 5. Silver Lining: The Perfect Action Tester
- **Outcome:** We successfully built a rapid testing framework. It can auto-play the base game and expose physics/action mapping bugs instantly.

### Gameplay Recording
Below is the agent using the Monte Carlo backtrack search algorithm, starting from the level 1 beginning:

![Monte Carlo Run]({{ site.baseurl }}/assets/recordings/mc_backtrack_Level1_x0_step1.gif)

## Next Steps
- Implement boss-health memory scraping for micro-rewards on successful multi-HP target hits.
