"""
Contra PPO Training with Stable-Baselines3
==========================================

Usage:
    python train.py                           # Default training
    python train.py --timesteps 10000000      # Custom timesteps
    python train.py --resume trained_models/ppo_contra_1000000_steps.zip
"""

import os

import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from contra_wrapper import ContraWrapper

# =============================================================================
# CONFIG
# =============================================================================

NUM_ENV = 16
LOG_DIR = "logs"
SAVE_DIR = "trained_models"
GAME = "ContraForce-Nes-v0"
STATE = "Level1"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# CUSTOM CALLBACK FOR TENSORBOARD LOGGING
# =============================================================================

class TensorboardCallback(BaseCallback):
    """Custom callback for logging episode stats to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_distances = []
        self.episode_scores = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check for episode end in each environment
        for i, info in enumerate(self.locals.get("infos", [])):
            if "episode_distance" in info:
                self.episode_distances.append(info["episode_distance"])
                self.episode_scores.append(info["episode_score"])
                self.episode_rewards.append(info["episode_reward"])

        # Log averages every 100 episodes
        if len(self.episode_distances) >= 100:
            self.logger.record("contra/avg_distance", np.mean(self.episode_distances))
            self.logger.record("contra/avg_score", np.mean(self.episode_scores))
            self.logger.record("contra/avg_reward", np.mean(self.episode_rewards))
            self.logger.record("contra/max_distance", np.max(self.episode_distances))
            self.logger.record("contra/max_score", np.max(self.episode_scores))
            # Clear for next batch
            self.episode_distances = []
            self.episode_scores = []
            self.episode_rewards = []

        return True


# =============================================================================
# LEARNING RATE & CLIP RANGE SCHEDULES
# =============================================================================

def linear_schedule(initial_value, final_value=0.0):
    """Linear interpolation between initial_value and final_value."""
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(game, state, seed=0, random_start_frames=0):
    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        env = ContraWrapper(env, random_start_frames=random_start_frames)
        env = Monitor(env)
        return env

    return _init


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contra PPO Training")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total training timesteps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--state", type=str, default=STATE,
                        help="Game state to train on")
    parser.add_argument("--random-start", type=int, default=0,
                        help="Max random no-op frames at episode start (0=disabled)")
    args = parser.parse_args()

    print("=" * 70)
    print("Contra Force - PPO Training")
    print("=" * 70)
    print(f"  Game:         {GAME}")
    print(f"  State:        {args.state}")
    print(f"  Envs:         {NUM_ENV}")
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  Random start: {args.random_start} frames")
    print(f"  Log dir:      {LOG_DIR}")
    print(f"  Save dir:     {SAVE_DIR}")
    if args.resume:
        print(f"  Resume:       {args.resume}")
    print("=" * 70)

    # Create vectorized environment
    env = SubprocVecEnv(
        [make_env(GAME, state=args.state, seed=i, random_start_frames=args.random_start)
         for i in range(NUM_ENV)]
    )

    # Learning rate and clip range schedules
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    if args.resume:
        # Load existing model
        print(f"Loading model from {args.resume}")
        custom_objects = {
            "learning_rate": lr_schedule,
            "clip_range": clip_range_schedule,
            "n_steps": 512,
        }
        model = PPO.load(args.resume, env=env, device="cuda", custom_objects=custom_objects)
    else:
        # Create new model
        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=512,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,  # Higher gamma for longer-term planning
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR,
        )

    # Callbacks
    checkpoint_interval = 62500  # 62500 * 16 envs = 1M steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=SAVE_DIR,
        name_prefix="ppo_contra",
    )
    tensorboard_callback = TensorboardCallback()

    # Training
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, tensorboard_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(SAVE_DIR, "ppo_contra_final.zip")
    model.save(final_path)
    print(f"Final model saved: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
