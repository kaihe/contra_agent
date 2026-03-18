# Project: Contra NES AI (RL)
- Game: Contra (NES) in experimental dir of stable-retro
- Action Space: MultiBinary (NES controller buttons)
- Library: Stable-Baselines3 + Gym Retro
- Goal: Train PPO agent to maximize score in Level 1
- Style: Python 3.10, Type Hints, Modular Code
- Python Env: Use `conda run -n vllm-env` for all Python commands

## Game Variables
- `lives`: Number of lives remaining (starts at 2)
- `score`: Current score (main reward signal)

## Key Files
- `main/train.py`: Training script with PPO
- `main/test.py`: Testing and recording script
- `main/contra_wrapper.py`: Custom gym wrapper

```
