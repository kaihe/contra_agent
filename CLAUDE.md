# Project: Contra Force AI (RL)
- Game: Contra Force (NES)
- Action Space: MultiBinary (NES controller buttons)
- Library: Stable-Baselines3 + Gym Retro
- Goal: Train PPO agent to maximize score in Level 1
- Style: Python 3.10, Type Hints, Modular Code
- Python Env: Use `conda run -n vllm-env` for all Python commands

## Game Variables
- `lives`: Number of lives remaining (starts at 2)
- `score`: Current score (main reward signal)

## Import Reference
- lexicographic : technique described in paper <learnfun & playfun: A general technique for automating NES games>

## Key Files
- `main/train.py`: Training script with PPO
- `main/test.py`: Testing and recording script
- `main/contra_wrapper.py`: Custom gym wrapper

## ROM Setup
The Contra Force ROM must be imported manually:
```bash
python -m retro.import /path/to/Contra\ Force\ \(USA\).nes
```
