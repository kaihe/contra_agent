"""
Contra PPO Training with Stable-Baselines3
==========================================

Usage:
    python train.py                           # Default training
    python train.py --timesteps 10000000      # Custom timesteps
    python train.py --resume trained_models/ppo_contra_1000000_steps.zip
"""

import os
import gzip
import shutil

import gymnasium as gym
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from contra_wrapper import ContraWrapper
# from curriculum_wrapper import CurriculumCallback

# =============================================================================
# CONFIG
# =============================================================================

NUM_ENV = 32
BATCH_SIZE = 512
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
        self.episode_steps = []

    def _on_step(self) -> bool:
        # Check for episode end in each environment
        for i, info in enumerate(self.locals.get("infos", [])):
            if "episode_distance" in info:
                self.episode_distances.append(info["episode_distance"])
                self.episode_scores.append(info["episode_score"])
                self.episode_rewards.append(info["episode_reward"])
                self.episode_steps.append(info.get("episode_steps", 1))

        # Log averages every 10 episodes
        if len(self.episode_distances) >= 10:
            self.logger.record("contra/avg_distance", np.mean(self.episode_distances))
            self.logger.record("contra/avg_score", np.mean(self.episode_scores))
            self.logger.record("contra/avg_ep_reward", np.mean(self.episode_rewards))
            self.logger.record("contra/avg_step_reward",
                               np.mean([r / s for r, s in zip(self.episode_rewards, self.episode_steps)]))
            self.logger.record("contra/max_distance", np.max(self.episode_distances))
            self.logger.record("contra/max_score", np.max(self.episode_scores))
            # Clear for next batch
            self.episode_distances = []
            self.episode_scores = []
            self.episode_rewards = []
            self.episode_steps = []

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
            path = retro.data.get_file_path(game, f"{s}.state")
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
        )
        if len(states) > 1:
            env = RandomStateWrapper(env, game=game, states=states)
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
    parser.add_argument("--timesteps", type=int, default=8_000_000,
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
    print("Contra Force - PPO Training")
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
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Learning rate and clip range schedules
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.05)

    if args.resume:
        # Load existing model
        print(f"Loading model from {args.resume}")
        # Load normalizer stats if available
        norm_path = args.resume.replace(".zip", "").rsplit("_steps", 1)[0] + "_vecnormalize.pkl"
        if not os.path.exists(norm_path):
            norm_path = os.path.join(SAVE_DIR, f"{args.name}_vecnormalize.pkl")
        if os.path.exists(norm_path):
            print(f"Loading VecNormalize from {norm_path}")
            env = VecNormalize.load(norm_path, env.venv)
        custom_objects = {
            "learning_rate": lr_schedule,
            "clip_range": clip_range_schedule,
            "n_steps": BATCH_SIZE,
        }
        model = PPO.load(args.resume, env=env, device="cuda", custom_objects=custom_objects)
    else:
        # Create new model
        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=0,
            n_steps=BATCH_SIZE,
            batch_size=BATCH_SIZE,
            n_epochs=4,
            gamma=0.99,  # Higher gamma for longer-term planning
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR,
        )

    # Callbacks
    checkpoint_interval = 62500  # 62500 * 32 envs ~ 2M steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=SAVE_DIR,
        name_prefix=args.name,
    )
    tensorboard_callback = TensorboardCallback()
    # Save VecNormalize stats alongside checkpoints
    class SaveNormalizeCallback(BaseCallback):
        def __init__(self, save_freq, save_path, name_prefix):
            super().__init__()
            self.save_freq = save_freq
            self.save_path = save_path
            self.name_prefix = name_prefix
        def _on_step(self):
            if self.n_calls % self.save_freq == 0:
                self.training_env.save(
                    os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize.pkl"))
            return True
    save_norm_callback = SaveNormalizeCallback(checkpoint_interval, SAVE_DIR, args.name)
    # curriculum_callback = CurriculumCallback(
    #     game=GAME,
    #     state=args.state,
    #     eval_freq=1_000_000,
    #     random_start_frames=args.random_start,
    # )

    # Training
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, tensorboard_callback, save_norm_callback],
        tb_log_name=args.name,
        progress_bar=True,
    )

    # Save final model and normalizer stats
    final_path = os.path.join(SAVE_DIR, f"{args.name}_final.zip")
    model.save(final_path)
    env.save(os.path.join(SAVE_DIR, f"{args.name}_vecnormalize.pkl"))
    print(f"Final model saved: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
