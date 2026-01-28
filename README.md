# Contra AI Agent

Train a PPO agent to play Contra (NES) using Stable-Baselines3 and Gym Retro.

## Setup

### 1. Install Dependencies

```bash
cd main
pip install -r requirements.txt
```

### 2. Import ROM

The Contra ROM is not included. You need to import it manually:

```bash
python -m retro.import /path/to/contra.nes
```

The game uses the experimental integration, so make sure stable-retro can find it.

## Usage

### Training

```bash
cd main

# Default training (10M timesteps)
python train.py

# Custom timesteps
python train.py --timesteps 5000000

# Resume from checkpoint
python train.py --resume trained_models/ppo_contra_1000000_steps.zip

# Train on different level
python train.py --state Level2

# Add random start frames for robustness
python train.py --random-start 30
```

### Testing

```bash
cd main

# Test default model
python test.py

# Test specific model
python test.py --model trained_models/ppo_contra_1000000_steps.zip

# Test with random agent
python test.py --random --episodes 10

# Record video
python test.py --record --episodes 5

# Test with random start frames
python test.py --random-start 30 --episodes 20
```

## Project Structure

```
contra_agent/
├── main/
│   ├── train.py              # Training script
│   ├── test.py               # Testing and recording
│   ├── contra_wrapper.py     # Custom environment wrapper
│   ├── requirements.txt      # Python dependencies
│   ├── logs/                 # TensorBoard logs
│   └── trained_models/       # Saved model checkpoints
├── CLAUDE.md                 # Project instructions
├── README.md                 # This file
└── .gitignore
```

## Game Info

- **Game**: Contra (NES) - Experimental integration
- **Levels**: 8 levels available (Level1-Level8)
- **Observation**: 84x84 RGB (downsampled from 256x224)
- **Action**: MultiBinary (NES controller: A, B, Select, Start, Up, Down, Left, Right)
- **Frame Skip**: 4 frames per action

## Reward Design

- **Progress**: +1.0 per unit of horizontal scroll (xscroll)
- **Death**: -100.0 penalty for losing a life
- **Level Complete**: +500.0 bonus for completing a level
- **Normalization**: All rewards scaled by 0.01

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Policy | CnnPolicy |
| Learning Rate | 2.5e-4 → 2.5e-6 (linear decay) |
| Clip Range | 0.15 → 0.025 (linear decay) |
| Gamma | 0.99 |
| N Steps | 512 |
| Batch Size | 512 |
| N Epochs | 4 |
| Parallel Envs | 16 |

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir main/logs
```
