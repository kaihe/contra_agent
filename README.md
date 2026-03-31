# Contra Agent

Building a real-time AI agent that plays **Contra (NES)** well — from scratch, step by step, and fully documented.

<video src="docs/assets/recordings/ch10_mc_search_level6.mp4" controls width="100%"></video>

## What This Is

This is an educational project that explores how far different AI techniques can get at playing a classic NES game. The goal is not just to beat Contra, but to understand *why* each approach works or fails.

The methods covered so far:

- **Reinforcement Learning** — PPO with pixel and RAM observations, reward shaping, action space design
- **Monte Carlo Search** — random rollouts with backtracking and anchor-based progress locking
- **Behavior Cloning** — supervised learning from human and search-generated demonstrations
- **Neural network backbones** — CNN, PPO policy heads, and Transformer-based architectures (in progress)

## Write-ups

Every experiment has a detailed write-up covering what was tried, what went wrong, and what was learned. They are published on the **[project site](https://kaihe.github.io/contra_agent/)** and written to be readable without prior RL knowledge.

## Project Structure

```
contra_agent/
├── contra/               # Core library: game wrapper, event system, inputs, replay
│   └── start_states/     # Emulator save states for levels 2-8 (spread gun acquired)
├── ppo/                  # PPO training and evaluation scripts
│   └── states/           # Level 1 anchor save states for RL training
├── synthetic/            # Monte Carlo search and synthetic data generation
│   └── action_bigram.npz # Per-level action bigram priors built from human recordings
└── docs/                 # GitHub Pages write-ups (10 chapters and counting)
```

## Quick Start

```bash
# Install
pip install -e .
pip install stable-baselines3 stable-retro gymnasium numpy opencv-python pygame

# Import the Contra ROM (you must supply the ROM file)
python -m retro.import /path/to/Contra.nes

# Run Monte Carlo search on Level 1
python synthetic/mc_search.py --level 1

# Train PPO on Level 1
python ppo/train.py

# Replay a saved trace
python contra/run_npz.py <path/to/trace.npz>
```

## References

- [nes-contra-us](https://github.com/vermiceli/nes-contra-us) — Community reverse-engineering of the Contra ROM; the source of all RAM address knowledge used in the event system
- [learnfun & playfun](https://tom7.org/mario/) — Tom Murphy's beam search approach to NES games; inspiration for the Monte Carlo search design
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO implementation
