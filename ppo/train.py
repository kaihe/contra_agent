"""
Contra (NES) PPO training with Stable-Baselines3.

Usage:
    python ppo/train.py
    python ppo/train.py --config ppo/ppo.yaml --timesteps 10000000
    python ppo/train.py --resume tmp/ppo/checkpoints/ppo_contra/ppo_contra_final.zip
"""

import argparse
import dataclasses
import glob
import os
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import contra  # registers custom ROM integration
import numpy as np
import stable_retro as retro
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from contra_wrapper import (
    DEFAULT_REWARD_WEIGHTS,
    RandomStateWrapper,
    create_env,
    save_config_to_model,
)


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "ppo.yaml")


@dataclass(frozen=True)
class PPOConfig:
    game: str = "Contra-Nes"
    state: str = "Level1"
    # Optional .state file paths sampled uniformly per episode (multi-state
    # training). When empty, every episode starts from `state`.
    states: list[str] = field(default_factory=list)
    name: str = "ppo_contra"
    timesteps: int = 32_000_000
    resume: str | None = None
    seed: int = 0
    device: str = "cuda"

    num_envs: int = 16
    random_start_frames: int = 0
    warmup_frames: int = 120
    max_episode_steps: int = 10000
    enemy_hp_cap_per_region: float = 5.0
    skip: int = 3
    stack: int = 4

    n_steps: int = 2048
    batch_size: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    learning_rate: float = 1e-4
    clip_range_initial: float = 0.2
    clip_range_final: float = 0.05
    ent_coef_initial: float = 0.1
    ent_coef_final: float = 0.005

    log_dir: str = "tmp/ppo/logs"
    save_dir: str = "tmp/ppo/checkpoints"
    save_freq: int = 125000
    reward_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_REWARD_WEIGHTS.copy()
    )


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def _config_from_mapping(data: dict) -> PPOConfig:
    valid = {field.name for field in dataclasses.fields(PPOConfig)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown PPO config field(s): {unknown}")

    merged = dict(data)
    if "reward_weights" in merged:
        weights = DEFAULT_REWARD_WEIGHTS.copy()
        unknown_weights = sorted(set(merged["reward_weights"]) - set(weights))
        if unknown_weights:
            raise ValueError(f"Unknown reward weight(s): {unknown_weights}")
        weights.update(merged["reward_weights"])
        merged["reward_weights"] = weights
    return PPOConfig(**merged)


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Contra PPO Training")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to PPO YAML config")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Training timesteps to run this invocation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--state", type=str, default=None,
                        help="Single game state to train on")
    parser.add_argument("--random-start", type=int, default=None,
                        help="Max random no-op frames at episode start")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")
    args = parser.parse_args()

    config_data = _load_yaml(args.config)
    overrides = {
        "timesteps": args.timesteps,
        "resume": args.resume,
        "state": args.state,
        "random_start_frames": args.random_start,
        "name": args.name,
    }
    config_data.update({k: v for k, v in overrides.items() if v is not None})
    return _config_from_mapping(config_data)


class LatestCheckpointCallback(CheckpointCallback):
    """Save one rolling checkpoint and embed the PPO/wrapper config."""

    def __init__(self, *args, train_config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_config = train_config

    def _step_number(self, path: str) -> int:
        # Filenames are "{name_prefix}_{steps}_steps.zip"; sort by the integer
        # step, not lexicographically (else "8000000" > "28000000").
        name = os.path.basename(path)
        try:
            return int(name[len(self.name_prefix) + 1: -len("_steps.zip")])
        except ValueError:
            return -1

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
            checkpoints = sorted(glob.glob(pattern), key=self._step_number)
            if checkpoints:
                save_config_to_model(
                    checkpoints[-1],
                    skip=self.train_config["skip"],
                    stack=self.train_config["stack"],
                    train_config=self.train_config,
                )
            for old in checkpoints[:-1]:
                os.remove(old)
                print(f"  Removed old checkpoint: {os.path.basename(old)}")
        return result


class EntropyScheduleCallback(BaseCallback):
    """Schedule entropy coefficient during training."""

    def __init__(self, entropy_schedule, verbose=0):
        super().__init__(verbose)
        self.entropy_schedule = entropy_schedule

    def _on_step(self) -> bool:
        progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)
        self.model.ent_coef = self.entropy_schedule(progress_remaining)
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True


class TensorboardCallback(BaseCallback):
    """Log Contra episode stats to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_delta_x = []
        self.episode_enemy_hp_cost = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_delta_x" in info:
                self.episode_delta_x.append(info["episode_delta_x"])
                self.episode_enemy_hp_cost.append(
                    info.get("episode_enemy_hp_cost", 0.0)
                )
                self.episode_rewards.append(info.get("episode_reward", 0))

        if len(self.episode_delta_x) >= 100:
            self.logger.record("contra/mean_delta_x", float(np.mean(self.episode_delta_x)))
            self.logger.record(
                "contra/mean_enemy_hp_cost",
                float(np.mean(self.episode_enemy_hp_cost)),
            )
            self.logger.record("contra/mean_reward", float(np.mean(self.episode_rewards)))

            self.episode_delta_x = []
            self.episode_enemy_hp_cost = []
            self.episode_rewards = []

        return True


def linear_schedule(initial_value, final_value=0.0):
    """Linear interpolation. SB3 progress goes from 1.0 to 0.0."""
    initial_value = float(initial_value)
    final_value = float(final_value)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def infer_level(state: str) -> int:
    """Infer level number from a state name such as Level2 or Level2_x100.state."""
    import re
    match = re.search(r"Level(\d+)", os.path.basename(state), re.IGNORECASE)
    return int(match.group(1)) if match else 1


def make_env(config: PPOConfig, rank: int):
    level = infer_level(config.state)

    def _init():
        env = retro.make(
            game=config.game,
            state=config.state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        if config.states:
            env = RandomStateWrapper(env, states=config.states)
        env = create_env(
            env,
            random_start_frames=config.random_start_frames,
            warmup_frames=config.warmup_frames,
            skip=config.skip,
            stack=config.stack,
            level=level,
            max_episode_steps=config.max_episode_steps,
            enemy_hp_cap_per_region=config.enemy_hp_cap_per_region,
            reward_weights=config.reward_weights,
        )
        np.random.seed(config.seed + rank)
        env.action_space.seed(config.seed + rank)
        env = Monitor(env)
        return env

    return _init


def main():
    config = parse_args()
    train_config = dataclasses.asdict(config)
    checkpoint_dir = os.path.join(config.save_dir, config.name)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("Contra (NES) - PPO Training")
    print("=" * 70)
    print(f"  Experiment:   {config.name}")
    print(f"  Game:         {config.game}")
    print(f"  State:        {config.state}")
    if config.states:
        print(f"  States:       {len(config.states)} anchors sampled per episode")
    print(f"  Envs:         {config.num_envs}")
    print(f"  Steps to run: {config.timesteps:,}")
    print(f"  Random start: {config.random_start_frames} frames")
    print(f"  Log dir:      {config.log_dir}")
    print(f"  Save dir:     {checkpoint_dir}")
    if config.resume:
        print(f"  Resume:       {config.resume}")
    print("=" * 70)

    env = SubprocVecEnv([make_env(config, i) for i in range(config.num_envs)])

    clip_range_schedule = linear_schedule(
        config.clip_range_initial,
        config.clip_range_final,
    )
    entropy_schedule = linear_schedule(
        config.ent_coef_initial,
        config.ent_coef_final,
    )

    if config.resume:
        print(f"Loading model from {config.resume}")
        custom_objects = {
            "learning_rate": config.learning_rate,
            "clip_range": clip_range_schedule,
            "n_steps": config.n_steps,
        }
        model = PPO.load(
            config.resume,
            env=env,
            device=config.device,
            custom_objects=custom_objects,
        )
        print(f"  Resumed at timestep {model.num_timesteps:,}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            device=config.device,
            verbose=0,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            ent_coef=config.ent_coef_initial,
            learning_rate=config.learning_rate,
            clip_range=clip_range_schedule,
            tensorboard_log=config.log_dir,
        )

    checkpoint_callback = LatestCheckpointCallback(
        save_freq=config.save_freq,
        save_path=checkpoint_dir,
        name_prefix=config.name,
        train_config=train_config,
    )
    callbacks = [
        checkpoint_callback,
        EntropyScheduleCallback(entropy_schedule),
        TensorboardCallback(),
    ]

    model.learn(
        total_timesteps=config.timesteps,
        callback=callbacks,
        tb_log_name=config.name,
        progress_bar=True,
        reset_num_timesteps=not bool(config.resume),
    )

    final_path = os.path.join(checkpoint_dir, f"final.zip")
    model.save(final_path)
    save_config_to_model(
        final_path,
        skip=config.skip,
        stack=config.stack,
        train_config=train_config,
    )
    print(f"Saved final model: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
