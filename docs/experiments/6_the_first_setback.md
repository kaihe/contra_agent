---
layout: default
title: "6. The First Setback"
parent: Experiments
nav_order: 6
---

# 6. The First Setback
{: .no_toc }

**Date:** 2026-03-18 · **Type:** Reflection
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Reward Engineering Does Not Transfer to Level 2

We spent considerable effort hacking the reward system to make the agent beat Level 1. The core ideas — shaping distance reward from `xscroll`, detecting boss HP decrements from RAM, penalising idle behaviour — eventually worked for Level 1. Moving to Level 2 revealed that none of these ideas transfer cleanly.

Level 2 is a top-down "walk into the screen" perspective. The player moves upward, not rightward. The `xscroll` value no longer represents continuous horizontal progress; it is a discrete room index that jumps by 256 each time the player advances to the next room. A continuous distance reward based on scroll position is meaningless here.

Level 2 also introduces a structural obstacle that Level 1 does not have: each room contains a gate core that must be hit **8 times** to open the door to the next room. Between the moment the player finds the core and the moment the core breaks, there is a long stretch — up to 8 shots — with no observable reward signal. The agent has no feedback that it is doing the right thing.

## Manual Play and RAM Analysis

To address the sparse reward problem, we played Level 2 manually and used the recordings to find the RAM addresses that track each room core's HP. We identified the correct addresses and wired them into the reward function so the agent receives a small positive reward for every hit on the core.

Training on this reward still did not work. The reason turned out to be the same one we observed in Level 1: the learning process requires the agent to stumble upon the correct behaviour through random exploration first. To hit the core reliably, the agent needs to navigate toward it while dodging enemy bullets — a coordination task that takes many random steps to discover. In the meantime, the agent dies repeatedly before it ever lands enough hits to receive a meaningful reward signal.

## Conclusion

At this point it is clear that the standard CNN-policy PPO approach is not suitable for Contra. The game's combination of sparse rewards, dense hazards, and long-horizon coordination requirements pushes PPO well beyond the regime where it can learn efficiently from scratch. We need to look for a fundamentally different learning algorithm.
