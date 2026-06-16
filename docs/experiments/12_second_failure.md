---
layout: default
title: "12. The Second Setback"
parent: Content
nav_order: 12
---

# 12. The Second Setback
{: .no_toc }

**Date:** 2026-06-16 · **Type:** Reflection
{: .fs-5 .fw-300 }

---

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Two Months of VLA Experiments

For the last two months we chased the idea of treating Contra as a vision-language-action (VLA) problem. We benchmarked a few baselines — Pixel2Play and SmolVLA — and then built our own variant: tokenize the game traces from the behavior data (Chapter 11), feed the visual tokens to a transformer backbone, and have it predict the next action token, exactly like a VLA model.

The results were disappointing. No matter how we set it up, the transformer collapsed onto the **prior action distribution** — it simply emitted the most common buttons and ignored the image.

## Why It Failed

Two structural problems in the data made this almost inevitable:

- **Action imbalance.** The win traces are dominated by `right` and `right + up`. The actions that actually matter — `jump` and `fire` — are rare but critical. A model that minimizes average token loss is rewarded for ignoring exactly the moments we care about.
- **Irrelevant actions.** The behavior data was collected by random search, so roughly **20%** of the recorded actions are inert: the button pressed has no effect on the game state (e.g. during a death/level animation). This is pure label noise — the same frame maps to arbitrary actions — which teaches the model that the image does not predict the action.

## What We Tried

We threw the standard toolbox at it, and none of it moved the needle:

- Pruning irrelevant actions based on game state.
- Up-weighting the rare, critical actions in the loss.
- Pre-training followed by DPO.
- GRPO RL post-training.

## The Real Lesson

The honest conclusion is not about any single technique. The problem was *method*: I was not building on a solid foundation and pushing the limit bit by bit. I kept switching between fancy ideas, and burned a lot of time and compute jumping between them without ever consolidating a working baseline.

So we are pulling back. The plan is to return to the **CNN policy with PPO**, make it reliably beat Levels 1–8 of Contra within a reasonable budget, and only then move on to the next game.
