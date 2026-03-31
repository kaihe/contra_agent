"""
Contra (NES) PPO Training — RAM observation, hp-loss + xscroll reward
======================================================================

Usage:
    python train_ram.py                           # Default training
    python train_ram.py --timesteps 10000000
    python train_ram.py --resume trained_models_ram/ppo_contra_ram_1000000_steps.zip
"""

import glob
import os
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from contra_ram_wrapper import create_env

# =============================================================================
# CONFIG
# =============================================================================

NUM_ENV    = 16
STACK      = 120   # RAM history buffer length
SAMPLE     = 12    # uniformly-sampled frames fed to the policy
BATCH_SIZE = 2048
LOG_DIR   = "logs_ram"
SAVE_DIR  = "trained_models_ram"
GAME      = "Contra-Nes"
STATE     = "Level1"

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# CALLBACKS
# =============================================================================

class LatestCheckpointCallback(CheckpointCallback):
    """Saves a checkpoint every save_freq steps and keeps only the latest."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
            checkpoints = sorted(glob.glob(pattern))
            for old in checkpoints[:-1]:
                os.remove(old)
                print(f"  Removed old checkpoint: {os.path.basename(old)}")
        return result


class TensorboardCallback(BaseCallback):
    """Log episode stats to TensorBoard every 100 episodes."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_max_x = []
        self.episode_rewards = []
        self.end_reasons = {"time_out": 0, "game_over": 0, "win": 0}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_max_x" in info:
                self.episode_max_x.append(info["episode_max_x"])
                self.episode_rewards.append(info.get("episode_reward", 0))
                
                reason = info.get("episode_end_reason", "")
                if reason in self.end_reasons:
                    self.end_reasons[reason] += 1

        if len(self.episode_max_x) >= 100:
            self.logger.record("contra/mean_max_x", np.mean(self.episode_max_x))
            self.logger.record("contra/mean_reward", np.mean(self.episode_rewards))
            
            total = sum(self.end_reasons.values()) or 1
            self.logger.record("contra/end_time_out",  self.end_reasons["time_out"]  / total)
            self.logger.record("contra/end_game_over", self.end_reasons["game_over"] / total)
            self.logger.record("contra/end_win",       self.end_reasons["win"]       / total)
            
            self.episode_max_x = []
            self.episode_rewards = []
            self.end_reasons = {"time_out": 0, "game_over": 0, "win": 0}

        return True


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(game: str, state: str, stack: int = 120, sample: int = 12):
    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.RAM,
            render_mode=None,
            inttype=retro.data.Integrations.ALL,
        )
        env = create_env(env, stack=stack, sample=sample)
        env = Monitor(env)
        return env
    return _init


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contra PPO Training (RAM)")
    parser.add_argument("--timesteps", type=int, default=32_000_000)
    parser.add_argument("--resume",    type=str, default=None)
    parser.add_argument("--state",     type=str, default=STATE)
    parser.add_argument("--name",      type=str, default="ppo_contra_ram")
    args = parser.parse_args()

    print("=" * 70)
    print("Contra (NES) - PPO Training [RAM obs]")
    print("=" * 70)
    print(f"  Experiment: {args.name}")
    print(f"  State:      {args.state}")
    print(f"  Envs:       {NUM_ENV}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Stack:      {STACK}  (sample {SAMPLE})")
    print(f"  Log dir:    {LOG_DIR}")
    print(f"  Save dir:   {SAVE_DIR}")
    if args.resume:
        print(f"  Resume:     {args.resume}")
    print("=" * 70)

    env = SubprocVecEnv(
        [make_env(GAME, state=args.state, stack=STACK, sample=SAMPLE) for _ in range(NUM_ENV)]
    )

    def clip_schedule(progress):
        return 0.05 + progress * (0.2 - 0.05)

    if args.resume:
        model = PPO.load(
            args.resume, env=env, device="cuda",
            custom_objects={"learning_rate": 1e-4, "clip_range": clip_schedule,
                            "n_steps": BATCH_SIZE},
        )
        print(f"  Resumed at timestep {model.num_timesteps:,}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device="cuda",
            verbose=0,
            n_steps=BATCH_SIZE,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.05,
            learning_rate=1e-4,
            clip_range=clip_schedule,
            tensorboard_log=LOG_DIR,
        )

    checkpoint_callback = LatestCheckpointCallback(
        save_freq=125_000,
        save_path=SAVE_DIR,
        name_prefix=args.name,
    )

    start_timesteps = model.num_timesteps if args.resume else 0
    model.learn(
        total_timesteps=start_timesteps + args.timesteps,
        callback=[checkpoint_callback, TensorboardCallback()],
        tb_log_name=args.name,
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume),
    )

    env.close()


if __name__ == "__main__":
    main()
