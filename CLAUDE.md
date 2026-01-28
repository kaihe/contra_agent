# Project: Contra AI (RL)
- Game: Contra (NES) - experimental integration
- Action Space: MultiBinary (NES controller buttons)
- Library: Stable-Baselines3 + Gym Retro
- Goal: Train PPO agent to complete Level 1
- Style: Python 3.10, Type Hints, Modular Code
- Python Env: Use `conda run -n vllm-env` for all Python commands

## Game Variables
- `lives`: Number of lives remaining
- `score`: Current score
- `xscroll`: Horizontal progress (main reward signal)
- `level`: Current level number
- `player_state`: Player state (15 = dead)

## Key Files
- `main/train.py`: Training script with PPO
- `main/test.py`: Testing and recording script
- `main/contra_wrapper.py`: Custom gym wrapper

## ROM Setup
The Contra ROM must be imported manually:
```bash
python -m retro.import /path/to/contra.nes
```
