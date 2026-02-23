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

## 1. Validating the Action Space First
Before embarking on long and costly RL training runs, we needed a way to perform a **quick check of the action space**. Our earlier RL agent in the Boss Fight Mix experiment suffered because it got stuck in bad states (e.g., releasing the B button by accident). We wanted a fast, algorithmic simulation to spot action space bugs immediately without waiting for PPO convergence.

## 2. Stealing Ideas: Learnfun & Playfun → Monte Carlo
We drew inspiration from Tom Murphy's famous **Learnfun/Playfun** algorithm, which solved NES games using lexicographic reward ranking and greedy search. However, Contra has a highly delayed reward structure (especially dodges). A greedy 1-step lookahead couldn't foresee a bullet that was 60 frames away.

To fix this, we modified the Playfun idea into a **Monte Carlo Search**:
- Instead of exhausting every 1-step action, we evaluate **256 random future rollouts**.
- We paired this "long sight" algorithm with the "time-travel" backtracking mechanic from Playfun. If all 256 random futures end in death, the agent realizes the *committed* root state was an inescapable bullet trap, and it instantly rewinds time dynamically to change the timeline and avoid the fatal sequence.

## 3. The Need for Deep Rollouts
To make Monte Carlo Search work in a delayed-reward environment like Contra, we needed to roll out "a little bit further" so that the result of a current action (like a bullet collision) is accurately reflected as a penalty.
- We set the rollout length to **16 actions** (where each action is skipped 4 frames), equating to a **64-frame lookahead**. This gave the agent enough foresight to spot enemy bullets and avoid them.

## 4. Why it Fails at the Boss
This Monte Carlo tracking worked beautifully for navigating the early game! The continuous influx of distance (x-scroll) and score points made evaluating rollouts easy. 

However, **it didn't work for the boss fight**. 
The Contra Level 1 boss has three core targets. The lone shooter on the top is easy to kill and gave points. But the other two targets require *multiple hits* to destroy. A purely random 16-action rollout sequence might randomly shoot the boss core once or twice, but the target wouldn't die. Consequently, **the rollout would receive no score reward**. The agent had absolutely no way of knowing that shooting the boss was "working" based on our current score and distance reward system!

## 5. The Silver Lining: A Fantastic Test Bed
While it couldn't quite beat the boss, the incredible news is that we now have a **highly effective, rapid test method for our action table**. It can automatically play the game in seconds, allowing us to debug our physics, action mappings, and emulator states. 

### Gameplay Recording
Below is the agent using the Monte Carlo backtrack search algorithm against the boss state. Watch it carefully dodge and backtrack!

![Monte Carlo Boss Fight]({{ site.baseurl }}/assets/recordings/mc_backtrack_Level1_x3048_step921.gif)

## Next Steps
We need to rework the reward system—possibly finding a way to read boss health from the RAM so the agent receives micro-rewards for landing individual shots on multi-HP targets.
