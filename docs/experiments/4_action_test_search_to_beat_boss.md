---
layout: default
title: "4. Action Test: MC search to beat boss"
parent: Experiments
nav_order: 4
---

# 4. Action Test: MC search to beat boss
{: .no_toc }

**Date:** 2026-02-24 · **Type:** Algorithmic Validation
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Motivation

There has been growing concern regarding our handcrafted action table (`ACTION_TABLE`). It is highly restricted and somewhat *ad hoc*—severely deviating from how a human actually outputs keystrokes. Before expending computational resources on another full reinforcement learning (PPO) training cycle, we **must answer a fundamental question: Is the Level 1 Boss mathematically beatable using our current action space?**

To verify this, we leveraged our Monte Carlo search framework from Chapter 3 to play out the exact boss fight scene. If the tree search algorithm can't brute-force a sequence of 4-frame skips to dodge the bullets and defeat the boss, then an RL agent certainly never will.

---

## Finding a Dense Output Signal

One major hurdle in solving the boss dynamically via tree search (and concurrently, RL exploration) is **sparse rewards**. The traditional game mechanics only grant points when the boss dies or a cannon breaks. A greedy tree search cannot plan 200 frames ahead with zero feedback to realize that firing bullets is a net positive.

To fix this, we inspected raw human gameplay demonstrations (`.npz` traces) and analyzed the full 2048 bytes of RAM. We were hunting for byte sequences that perfectly correlated with individual bullet impacts. 

We successfully isolated three distinct addresses acting as **Bullet-by-Bullet Hit Counters**:
* `RAM[1412]`: Left Cannon HP (~16 hits)
* `RAM[1414]`: Main Core HP (~32 hits)
* `RAM[1415]`: Right Cannon HP (~16 hits)

Every single time the player lands a shot on any of the three boss targets, the corresponding counter drops precisely by 1. Below is our visual proof demonstrating this perfectly smooth decrement across all winning human traces:

![Boss HP Analysis]({{ site.baseurl }}/assets/boss_analysis.png)

By injecting these absolute hit counters into the Monte Carlo search evaluation logic, we transformed the sparse boss fight into a densely rewarded gradient: the search algorithm now receives explicit reward points (+1) for every single bullet that connects with the boss's hitbox.

---

## MC Search Results: Boss Defeated (Conditionally)

Armed with the dense bullet reward, we initialized the Monte Carlo backtrack tree search at the exact pixel column the boss spawns at, forcing it to find a path using our strictly defined `ACTION_TABLE`.

### The Bad News: Default Rifle Fails
When initialized with the **Default Rifle**, the tree search struggles severely.

![Monte Carlo Search - Normal Rifle]({{ site.baseurl }}/assets/recordings/mc_backtrack_normal.gif)

The slow fire rate and lack of spread mean the algorithm has to string together heavily complex, perfect jumping logic over hundreds of frames to even survive while dealing baseline DPS, ultimately collapsing to inescapable trap sequences long before breaking the core. We suspect that our current action config is indeed problematic under extreme bullet hell constraints, or it simply requires a vast amount of search time to find the exact dodge pixels.

### The Good News: Spread Gun is Viable!
However, when initialized possessing the **Spread Gun**, the raw DPS completely overwhelms the algorithmic limitations! 

![Monte Carlo Search - Spread Gun]({{ site.baseurl }}/assets/recordings/mc_backtrack_spread.gif)

Thanks to the Spread Gun's immense fan of damage, the MC search effortlessly spammed the fire button while performing minimal dodges, breaking both cannons and annihilating the core with devastating speed!

## Conclusion

This experiment confirmed our most crucial hypothesis: **Our handcrafted action table is capable of defeating the boss, provided the agent has the Spread Gun.**
We have proven that if the PPO agent is smart enough to avoid the Laser Rifle trap (reinforcing Chapter 5) and carry the Spread Gun into the boss arena, it absolutely has the mechanical capability within its restricted action space to achieve victory.
