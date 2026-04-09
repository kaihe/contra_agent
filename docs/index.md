---
layout: default
title: Home
nav_order: 1
nav_exclude: true
---

# Contra Agent

A real-time video game playing agent that takes raw video frames as input and outputs keyboard and mouse actions at 20Hz.

---

## Motivation

Large language models are arguably the greatest innovation of this generation. In the digital world — math, coding, writing, reasoning — they have proven remarkably capable, often matching or exceeding human performance on complex tasks.

But the physical world is a different story.

The core bottleneck is **time**. The physical world is continuous and ever-changing. Decisions must be made in real time to keep pace with it. LLMs, designed to reason deeply over static context, aren't built for this. A model that takes two seconds to respond is useless when the world has already moved on.

This suggests we need a different class of model for physical-world tasks: lightweight, task-specific networks that internalize the context of a particular environment and produce actions reflexively — the way humans develop **muscle memory**. Not a general reasoner, but a fast, specialized actor.

This project is a step in that direction.

---

## What This Is

As a lifelong video game fan, I spend a lot of time playing games. That playtime generates something valuable: behavioral data. This project turns that data into training signal for a real-time game-playing agent.

The pipeline has two parts:

- **Data collection** — screen capture and input logging during human gameplay, converted into (frame, text, action) training pairs
- **Model training** — a vision model trained to map video frames to controller actions, fine-tuned with reinforcement learning against the game environment itself

The long-term goal is an agent that can play any game I care about — RimWorld, Factorio, Stardew Valley. But every ambitious project needs a humble starting point.

That starting point is **Contra on NES**.

The chapters that follow are the development log of this journey — full of trials, errors, and course corrections. The technical roadmap will shift as I learn. I'll document the tricks that actually work, and just as honestly, the dead ends that weren't worth the detour.

---

## Source Code

📦 [GitHub Repository](https://github.com/kaihe/contra_agent)
