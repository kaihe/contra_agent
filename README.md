# Contra Agent 🎮

Training an AI to play **Contra (NES)** using Reinforcement Learning (PPO).

![Reference gameplay](docs/assets/video-1.gif)

## Overview

This project trains a PPO agent via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) to beat Level 1 of the classic NES game Contra. The agent learns from raw pixels (84×84 grayscale, 4-frame stack) and outputs discrete controller actions.

## Project Structure

```
contra_agent/
├── main/
│   ├── train.py              # PPO training script
│   ├── test.py               # Test & record model gameplay
│   ├── contra_wrapper.py     # Gym wrapper: action table, reward shaping, frame processing
│   ├── trained_models/       # Saved model checkpoints (.zip)
│   └── states/               # Custom emulator save states (.state)
├── inspect_action/
│   ├── inspect_model.py      # Step-by-step gameplay navigator (matplotlib)
│   └── run_sequence.py       # Run & record predefined action sequences
└── docs/                     # GitHub Pages experiment write-ups
```

## Quick Start

```bash
# Setup
conda create -n vllm-env python=3.10
conda activate vllm-env
pip install stable-baselines3 stable-retro gymnasium numpy opencv-python pygame imageio

# Import ROM
python -m retro.import /path/to/Contra.nes

# Train
cd main
python train.py --name run_and_gun --timesteps 32000000

# Test
python test.py --model trained_models/run_and_gun_final.zip --render
```

## Experiment Log

Detailed write-ups with gameplay GIFs and analysis are available on the **[project site](https://kaihe.github.io/contra_agent/)**.

| # | Experiment | Key Finding |
|---|------------|-------------|
| 1 | [Baseline](https://kaihe.github.io/contra_agent/experiments/baseline) | Agent navigates early game well but fails at boss — fire and movement were mutually exclusive |
| 2 | [Boss Fight Mix](https://kaihe.github.io/contra_agent/experiments/boss_fight_mix) | Always-fire action space + boss state training — discovered B-button release bug (zero kills) |
| 3 | [Gun Advantage](https://kaihe.github.io/contra_agent/experiments/gun_advantage) | Weapon analysis under rapid fire — laser self-cancels, spread dominates, training plan for run-and-gun |

## Key Design Decisions

- **Always-fire action space** — `B=1` on all 8 actions so the agent can run-and-gun like a human player
- **B-release fix** — Release the fire button on the last frame of each 4-frame skip to enable rapid fire
- **Multi-state training** — Random start from 4 save states (level start, spread gun, laser, boss) for balanced experience
- **Embedded model config** — Each model `.zip` contains a `contra_config.json` with the action table it was trained with, ensuring old models remain compatible when the action space changes

## References

- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [vietnh1009/Contra-PPO-pytorch](https://github.com/vietnh1009/Contra-PPO-pytorch) — Reference PPO implementation for Contra
- [learnfun & playfun](https://tom7.org/mario/) — Tom Murphy's beam search approach to NES games
