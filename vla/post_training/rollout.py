"""Grouped rollout collection for ContraVLA GRPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .env_wrappers import VLAEnv
from .policy import VLAPolicy
from .trajectory import Trajectory


@dataclass
class RolloutConfig:
    level: str = "Level1"
    group_size: int = 32
    rollout_len: int = 48
    max_rewind: int = 32
    max_actions: int = 4000
    max_groups: int = 10_000
    temperature: float = 0.9
    high_temperature: float = 1.2
    save_every: int = 25
    out_dir: str = "tmp/vla_grpo"


def _detach_obs(obs: dict) -> dict:
    return {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in obs.items()}


def collect_group(
    env: VLAEnv,
    policy: VLAPolicy,
    committed_state: bytes,
    cfg: RolloutConfig,
    *,
    temperature: float,
) -> list[Trajectory]:
    group = [Trajectory() for _ in range(cfg.group_size)]
    branch_states = [committed_state for _ in range(cfg.group_size)]

    for _ in range(cfg.rollout_len):
        obs_batch, active = [], []
        for i, state in enumerate(branch_states):
            if group[i].done:
                continue
            obs_batch.append(env.restore(state))
            active.append(i)

        if not active:
            break

        actions, logprobs = policy.sample_batch(obs_batch, temperature=temperature)
        for idx, obs, action, logprob in zip(active, obs_batch, actions, logprobs):
            env.restore(branch_states[idx])
            _, reward, status, info = env.step(action)
            group[idx].append(_detach_obs(obs), int(action), float(logprob), reward, status, info)
            branch_states[idx] = env.snapshot()

    env.restore(committed_state)
    return group
