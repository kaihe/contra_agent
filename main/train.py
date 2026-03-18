"""
Contra (NES) PPO Training with Stable-Baselines3
=================================================

Usage:
    python train.py                           # Default training
    python train.py --timesteps 10000000      # Custom timesteps
    python train.py --resume trained_models/ppo_contra_1000000_steps.zip
"""

import glob
import os

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import gymnasium as gym
import numpy as np
import torch as th
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import obs_as_tensor


class LatestCheckpointCallback(CheckpointCallback):
    """Saves a checkpoint every save_freq steps, keeps only the latest one,
    and embeds contra_config.json into each saved file."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
            checkpoints = sorted(glob.glob(pattern))
            if checkpoints:
                save_config_to_model(checkpoints[-1])
            for old in checkpoints[:-1]:
                os.remove(old)
                print(f"  Removed old checkpoint: {os.path.basename(old)}")
        return result
from stable_baselines3.common.vec_env import SubprocVecEnv

from contra_wrapper import create_env, save_config_to_model

# =============================================================================
# CONFIG
# =============================================================================

NUM_ENV = 16
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
        self.episode_death_rewards = []
        self.episode_game_result_rewards = []
        self.episode_enemy_progress_rewards = []
        self.end_reasons = {"time_out": 0, "game_over": 0, "win": 0}

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            if "episode_max_x" in info:
                self.episode_max_x.append(info["episode_max_x"])
                self.episode_scores.append(info["episode_score"])
                self.episode_rewards.append(info["episode_reward"])
                self.episode_steps.append(info.get("episode_steps", 1))
                self.episode_distance_rewards.append(info.get("episode_distance_reward", 0))
                self.episode_score_rewards.append(info.get("episode_score_reward", 0))
                self.episode_death_rewards.append(info.get("episode_death_reward", 0))
                self.episode_game_result_rewards.append(info.get("episode_game_result_reward", 0))
                self.episode_enemy_progress_rewards.append(info.get("episode_enemy_progress_reward", 0))
                reason = info.get("episode_end_reason", "")
                if reason in self.end_reasons:
                    self.end_reasons[reason] += 1

        if len(self.episode_max_x) >= 100:
            self.logger.record("contra/mean_max_x", np.mean(self.episode_max_x))
            self.logger.record("contra/mean_actions", np.mean(self.episode_steps))
            self.logger.record("contra/mean_score", np.mean(self.episode_scores))
            self.logger.record("contra/mean_reward", np.mean(self.episode_rewards))
            self.logger.record("contra/reward_distance", np.mean(self.episode_distance_rewards))
            self.logger.record("contra/reward_score", np.mean(self.episode_score_rewards))
            self.logger.record("contra/reward_death", np.mean(self.episode_death_rewards))
            self.logger.record("contra/reward_game_result", np.mean(self.episode_game_result_rewards))
            self.logger.record("contra/reward_enemy_progress", np.mean(self.episode_enemy_progress_rewards))
            total = sum(self.end_reasons.values()) or 1
            self.logger.record("contra/end_time_out", self.end_reasons["time_out"] / total)
            self.logger.record("contra/end_game_over", self.end_reasons["game_over"] / total)
            self.logger.record("contra/end_win", self.end_reasons["win"] / total)
            self.episode_max_x = []
            self.episode_scores = []
            self.episode_rewards = []
            self.episode_steps = []
            self.episode_distance_rewards = []
            self.episode_score_rewards = []
            self.episode_death_rewards = []
            self.episode_game_result_rewards = []
            self.episode_enemy_progress_rewards = []
            self.end_reasons = {"time_out": 0, "game_over": 0, "win": 0}

        return True


# =============================================================================
# RND CURIOSITY REWARD
# =============================================================================

class _RNDEnvAdapter:
    """Minimal adapter so rllte.RND can infer obs/action shape from SB3 VecEnv.

    rllte expects channels-first obs, but ContraWrapper exposes (84, 84, 4).
    We report (4, 84, 84) here so the CNN encoder is built correctly.
    """
    def __init__(self, vec_env, stack: int = 4):
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(stack, 84, 84), dtype=np.uint8
        )
        self.action_space = vec_env.action_space
        self.num_envs = vec_env.num_envs
        self.unwrapped = self


class RNDCallback(BaseCallback):
    """Augments rollout rewards with RND intrinsic curiosity before each policy update.

    Flow (per rollout):
      1. Transpose buffer observations to channels-first (n_steps, n_envs, 4, 84, 84)
      2. Call rnd.compute() → normalised intrinsic rewards (n_steps, n_envs)
      3. Add beta * intrinsic to rollout_buffer.rewards
      4. Recompute GAE returns/advantages with the modified rewards
    """

    def __init__(self, rnd, beta: float = 0.5, verbose: int = 0):
        super().__init__(verbose)
        self.rnd = rnd
        self.beta = beta
        self._last_dones = None

    def _on_step(self) -> bool:
        # Capture dones from the most recent env step for GAE recomputation.
        self._last_dones = self.locals.get("dones")
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        n_steps, n_envs = buf.rewards.shape

        # (n_steps, n_envs, 84, 84, 4) → (n_steps, n_envs, 4, 84, 84), float32 [0,1]
        obs_chw = np.transpose(buf.observations, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
        obs_t = th.as_tensor(obs_chw, device=self.rnd.device)

        samples = {
            "observations": obs_t,
            "next_observations": obs_t,          # RND only uses next_obs
            "actions": th.as_tensor(buf.actions),
            "rewards": th.as_tensor(buf.rewards),
            "terminateds": th.zeros(n_steps, n_envs, device=self.rnd.device),
            "truncateds": th.zeros(n_steps, n_envs, device=self.rnd.device),
        }

        intrinsic = self.rnd.compute(samples, sync=True).cpu().numpy()
        buf.rewards += self.beta * intrinsic

        # Recompute GAE returns/advantages with modified rewards
        with th.no_grad():
            last_obs_t = obs_as_tensor(self.model._last_obs, self.model.device)
            last_values = self.model.policy.predict_values(last_obs_t)
        dones = self._last_dones if self._last_dones is not None \
            else np.zeros(n_envs, dtype=bool)
        buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

        if self.verbose:
            self.logger.record("rnd/mean_intrinsic", float(intrinsic.mean()))


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
# ENVIRONMENT FACTORY
# =============================================================================

def infer_level(states: list) -> int:
    """Infer level number from state names (e.g. 'Level2' or 'Level2_x100.state' -> 2)."""
    import re
    for s in states:
        m = re.search(r"Level(\d+)", os.path.basename(s), re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 1  # default


def make_env(game, states, seed=0, random_start_frames=0):
    level = infer_level(states)

    def _init():
        # Use first non-file state for retro.make, or fallback to default
        init_state = None
        for s in states:
            if not (s.endswith(".state") and os.path.isfile(s)):
                init_state = s
                break
        if init_state is None:
            init_state = f"Level{level}"

        env = retro.make(
            game=game,
            state=init_state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
            inttype=retro.data.Integrations.ALL,
        )
        env = create_env(env, random_start_frames=random_start_frames, level=level)
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
    parser.add_argument("--rnd", action="store_true",
                        help="Enable RND curiosity reward (rllte-core)")
    parser.add_argument("--rnd-beta", type=float, default=0.5,
                        help="Intrinsic reward injection scale (0=off, 1=full weight)")
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
    if args.rnd:
        print(f"  RND:          enabled (beta={args.rnd_beta})")
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
        print(f"  Resumed at timestep {model.num_timesteps:,}")
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

    # RND curiosity module (optional)
    callbacks = []
    if args.rnd:
        from rllte.xplore.reward import RND
        rnd = RND(
            envs=_RNDEnvAdapter(env),
            device="cuda",
            beta=1.0,          # internal normalization scale; injection scale is rnd_beta
            kappa=0.0,
            obs_norm_type=None, # skip costly normalization warmup
        )
        callbacks.append(RNDCallback(rnd, beta=args.rnd_beta, verbose=1))

    # Callbacks
    checkpoint_interval = 125000  # 125000 * 32 envs = 4M steps
    checkpoint_callback = LatestCheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=SAVE_DIR,
        name_prefix=args.name,
    )
    entropy_callback = EntropyScheduleCallback(entropy_schedule)
    tensorboard_callback = TensorboardCallback()
    callbacks += [checkpoint_callback, entropy_callback, tensorboard_callback]

    # Training
    # When resuming, use cumulative total_timesteps so schedules continue from
    # where they left off rather than restarting (reset_num_timesteps=False).
    start_timesteps = model.num_timesteps if args.resume else 0
    total_timesteps = start_timesteps + args.timesteps
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=args.name,
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume),
    )

    env.close()


if __name__ == "__main__":
    main()
