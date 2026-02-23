---
layout: default
title: Home
nav_order: 1
---

# Contra Agent

Training an AI to play **Contra (NES)** using Reinforcement Learning.

---

## Project Overview

This project trains a PPO agent (via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) to play the classic NES game Contra. 

## Experiment Log

| Date | Experiment | Summary |
|------|------------|---------|
| 2026-02-22 | [Baseline](experiments/baseline) | First training run — agent learns early-game navigation but fails at the boss |
| 2026-02-23 | [Boss Fight Mix](experiments/boss_fight_mix) | Always-fire action space + boss state — failed due to B-button release bug |
| 2026-02-23 | [Monte Carlo Search & Playfun](experiments/monte_carlo_search) | Monte Carlo simulated trace search to test actions & dodge bullets using far-sight |
| 2026-02-23 | [Gun Advantage](experiments/gun_advantage) | Weapon analysis under constant fire — laser self-cancels, spread dominates |

---

## Source Code

📦 [GitHub Repository](https://github.com/kaihe/contra_agent)
