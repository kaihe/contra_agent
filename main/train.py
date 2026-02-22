"""
Contra (NES) PPO Training with Stable-Baselines3
=================================================

Usage:
    python train.py                           # Default training
    python train.py --timesteps 10000000      # Custom timesteps
    python train.py --resume trained_models/ppo_contra_1000000_steps.zip
"""

import os
import gzip

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import gymnasium as gym
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from contra_wrapper import create_env

# =============================================================================
# CONFIG
# =============================================================================

NUM_ENV = 32
BATCH_SIZE = 2048
LOG_DIR = "logs"
SAVE_DIR = "trained_models"
GAME = "Contra-Nes"
STATE = "Level1"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# CUSTOM CALLBACK FOR TENSORBOARD LOGGING
# =============================================================================

class EntropyScheduleCallback(BaseCallback):
    """Custom callback for scheduling entropy coefficient during training."""

    def __init__(self, entropy_schedule, verbose=0):
        super().__init__(verbose)
        self.entropy_schedule = entropy_schedule

    def _on_step(self) -> bool:
        # Update entropy coefficient based on training progress
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)
        self.model.ent_coef = self.entropy_schedule(progress_remaining)
        # Log current entropy coefficient
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True


class TensorboardCallback(BaseCallback):
    """Custom callback for logging episode stats to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_max_x = []
        self.episode_scores = []
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_distance_rewards = []
        self.episode_score_rewards = []
        self.end_reasons = {"time_out": 0, "game_over": 0, "win": 0}

    def _on_step(self) -> bool:
        # Check for episode end in each environment
        for i, info in enumerate(self.locals.get("infos", [])):
            if "episode_max_x" in info:
                self.episode_max_x.append(info["episode_max_x"])
                self.episode_scores.append(info["episode_score"])
                self.episode_rewards.append(info["episode_reward"])
                self.episode_steps.append(info.get("episode_steps", 1))
                self.episode_distance_rewards.append(info.get("episode_distance_reward", 0))
                self.episode_score_rewards.append(info.get("episode_score_reward", 0))
                reason = info.get("episode_end_reason", "")
                if reason in self.end_reasons:
                    self.end_reasons[reason] += 1

        # Log averages every 100 episodes
        if len(self.episode_max_x) >= 100:
            self.logger.record("contra/mean_max_x", np.mean(self.episode_max_x))
            self.logger.record("contra/mean_actions", np.mean(self.episode_steps))
            self.logger.record("contra/mean_score", np.mean(self.episode_scores))
            self.logger.record("contra/mean_reward", np.mean(self.episode_rewards))
            self.logger.record("contra/reward_distance", np.mean(self.episode_distance_rewards))
            self.logger.record("contra/reward_score", np.mean(self.episode_score_rewards))
            total = sum(self.end_reasons.values()) or 1
            self.logger.record("contra/end_time_out", self.end_reasons["time_out"] / total)
            self.logger.record("contra/end_game_over", self.end_reasons["game_over"] / total)
            self.logger.record("contra/end_win", self.end_reasons["win"] / total)
            # Clear for next batch
            self.episode_max_x = []
            self.episode_scores = []
            self.episode_rewards = []
            self.episode_steps = []
            self.episode_distance_rewards = []
            self.episode_score_rewards = []
            self.end_reasons = {"time_out": 0, "game_over": 0, "win": 0}

        return True


# =============================================================================
# LEARNING RATE & CLIP RANGE SCHEDULES
# =============================================================================

def linear_schedule(initial_value, final_value=0.0):
    """Linear interpolation between initial_value and final_value.

    Args:
        initial_value: Starting value (at progress=1.0, beginning of training)
        final_value: Ending value (at progress=0.0, end of training)

    Note: progress goes from 1.0 -> 0.0 during training
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        # progress: 1.0 at start -> 0.0 at end
        return final_value + progress * (initial_value - final_value)

    return scheduler


# =============================================================================
# RANDOM STATE WRAPPER
# =============================================================================

class RandomStateWrapper(gym.Wrapper):
    """On each reset, load a randomly chosen state from a list."""

    def __init__(self, env, game, states):
        super().__init__(env)
        self.game = game
        self.states = states
        # Preload all state data
        self.state_data = []
        for s in states:
            path = retro.data.get_file_path(game, f"{s}.state",
                                            inttype=retro.data.Integrations.ALL)
            with gzip.open(path, "rb") as f:
                self.state_data.append(f.read())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Load a random state
        idx = np.random.randint(len(self.state_data))
        self.unwrapped.em.set_state(self.state_data[idx])
        self.unwrapped.data.update_ram()
        # Step once to sync observation with loaded state
        obs, _, _, _, info = self.env.step(
            np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        )
        return obs, info


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(game, states, seed=0, random_start_frames=0):
    def _init():
        env = retro.make(
            game=game,
            state=states[0],
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
            inttype=retro.data.Integrations.ALL,
        )
        if len(states) > 1:
            env = RandomStateWrapper(env, game=game, states=states)
        env = create_env(env, random_start_frames=random_start_frames)
        env = Monitor(env)
        return env

    return _init


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contra PPO Training")
    parser.add_argument("--timesteps", type=int, default=32_000_000,
                        help="Total training timesteps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--state", type=str, nargs="+", default=[STATE],
                        help="Game state(s) to train on (randomly selected per episode)")
    parser.add_argument("--random-start", type=int, default=0,
                        help="Max random no-op frames at episode start (0=disabled)")
    parser.add_argument("--name", type=str, default="ppo_contra",
                        help="Experiment name (used for tensorboard and checkpoints)")
    args = parser.parse_args()

    print("=" * 70)
    print("Contra (NES) - PPO Training")
    print("=" * 70)
    print(f"  Experiment:   {args.name}")
    print(f"  Game:         {GAME}")
    print(f"  State(s):     {', '.join(args.state)}")
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
        [make_env(GAME, states=args.state, seed=i, random_start_frames=args.random_start)
         for i in range(NUM_ENV)]
    )

    # Learning rate, clip range, and entropy schedules
    clip_range_schedule = linear_schedule(0.2, 0.05)
    # Start with high exploration (0.1), decay to lower exploration (0.005)
    entropy_schedule = linear_schedule(0.1, 0.005)

    if args.resume:
        # Load existing model
        print(f"Loading model from {args.resume}")
        custom_objects = {
            "learning_rate": 1e-4,
            "clip_range": clip_range_schedule,
            "n_steps": BATCH_SIZE,
        }
        model = PPO.load(args.resume, env=env, device="cuda", custom_objects=custom_objects)
        # Set initial entropy (will be updated by EntropyScheduleCallback)
        model.ent_coef = 0.1
    else:
        # Create new model
        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=0,
            n_steps=BATCH_SIZE,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.1,  # Initial value (will be scheduled by callback)
            learning_rate=1e-4,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR,
        )

    # Callbacks
    checkpoint_interval = 125000  # 125000 * 16 envs = 2M steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=SAVE_DIR,
        name_prefix=args.name,
    )
    entropy_callback = EntropyScheduleCallback(entropy_schedule)
    tensorboard_callback = TensorboardCallback()

    # Training
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, entropy_callback, tensorboard_callback],
        tb_log_name=args.name,
        progress_bar=True,
    )

    # Save final model and normalizer stats
    final_path = os.path.join(SAVE_DIR, f"{args.name}_final.zip")
    model.save(final_path)
    print(f"Final model saved: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
