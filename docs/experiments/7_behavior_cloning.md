---
layout: default
title: "7. Behavior Cloning — A New Baseline"
parent: Content
nav_order: 7
---

# 7. Behavior Cloning — A New Baseline
{: .no_toc }

**Date:** 2026-03-18 · **Type:** New Approach
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

Chapter 6 concluded that CNN-policy PPO is not an effective starting point for Contra. The fundamental obstacle is that the agent must discover coordinated behaviour through random exploration in an environment where mistakes are immediately fatal. Reward shaping can help at the margin, but it cannot compensate for a policy that starts from pure noise.

The natural remedy is to give the agent a head start: teach it to imitate human play before any reinforcement signal is introduced. We already have a library of human play recordings for both Level 1 and Level 2, collected during the RAM analysis phase. These recordings are the raw material for **behavior cloning**.

## A New Architecture: Transformers for Game Playing

We draw inspiration from the paper *An Open Model for Real-Time Video Game Playing*, which proposes a fundamentally different architecture for game-playing agents. Rather than a CNN feature extractor feeding into an MLP policy head — the standard setup for PPO on Atari-style games — the model is a **transformer** that consumes a sequence of image tokens and text tokens and produces action tokens as output.

This framing deliberately mirrors the architecture of large language models:

- **Image tokens** encode the current game frame (and optionally a short history of past frames).
- **Text tokens** carry task context — the game name, the current level, a natural-language goal description, or any other structured information.
- **Action tokens** are the output, decoded autoregressively just like text generation.

The transformer's attention mechanism allows the model to reason over temporal context across the full sequence, which is far more expressive than the fixed-window frame stacking used in PPO. It also means the same model architecture can, in principle, generalise across multiple games and tasks without structural changes.

### The Text Tokens as Game Wisdom

The role of text tokens deserves a closer look. They are not just metadata. They are the channel through which strategy enters the model — descriptions of the game, the current goal, what to watch out for, how a particular enemy behaves. In this sense, text tokens carry the *wisdom* about how to play well, while the image tokens carry the raw perception of what is happening right now.

The network itself, then, is less of a strategist and more of an **executor**. Given a description of what to do and a view of the current screen, it converts that into precise button presses — frame by frame, at real-time speed. This is closer to muscle memory than to reasoning: a trained reflex that reliably carries out whatever the text context instructs.

This separation opens up an important architectural possibility. We do not need the game agent itself to be the thing that reasons about strategy. A larger, slower model — a general-purpose language model or a dedicated game-strategy planner — can observe the situation and issue high-level instructions as text. The game agent receives those instructions and executes them faithfully, seeing the plan through at the millisecond timescale that the game demands. The planner thinks; the agent acts.

This is a natural fit for hierarchical control: the planner operates on the timescale of decisions ("head toward the core", "dodge left") while the executor operates on the timescale of frames. Neither needs to do the other's job, and the interface between them is plain language.

## Training in Two Phases

The two-phase training pipeline is a direct analogy to how modern language models are trained.

### Phase 1 — Pretraining via Behavior Cloning

In the first phase, the model is trained purely by imitation. For each frame in the human recording dataset, the input is the game screen and the output is the action the human took. This is standard supervised learning — cross-entropy loss over the action vocabulary — and it requires no reward signal whatsoever.

This is the critical advantage over PPO. The model does not need to discover how to play from scratch. After pretraining, the policy already knows how to navigate rooms, approach cores, and avoid common hazards, because it has seen a human do all of these things. The sparse reward problem that defeated PPO is irrelevant at this stage: the model learns from dense supervision (every frame has a label) rather than sparse environmental feedback.

### Phase 2 — Post-Training with Score-Based Reward

Once the model has learned a competent baseline policy through imitation, it can be further improved using reinforcement learning. The reward signal at this stage is simple and natural: the in-game score. A run that scores higher is better. No hand-crafted reward shaping is needed.

This mirrors the relationship between LLM pretraining and RLHF. Pretraining gives the model broad capability; post-training refines it toward a specific objective. The key insight is that RL is far more effective when it starts from a policy that already behaves reasonably, rather than from random noise. The exploration problem that makes RL so difficult on hard games becomes tractable once the agent already knows the rough shape of good behaviour.


