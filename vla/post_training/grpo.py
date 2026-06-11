"""GRPO update for one grouped ContraVLA rollout batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .policy import VLAPolicy
from .trajectory import Trajectory


@dataclass
class GRPOConfig:
    epochs: int = 1
    batch_size: int = 8
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0


@dataclass
class GRPOStats:
    loss: float = 0.0
    entropy: float = 0.0
    kl: float = 0.0
    clip_frac: float = 0.0
    ratio_mean: float = 0.0
    ratio_max: float = 0.0
    raw_grad_norm: float = 0.0
    grad_norm: float = 0.0
    tokens: int = 0


def _training_rows(group: list[Trajectory]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, proprio, actions, old_logprobs, advantages = [], [], [], [], []
    returns = torch.tensor([traj.total_reward for traj in group], dtype=torch.float32)
    adv = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
    for traj, traj_adv in zip(group, adv.tolist()):
        for step in traj.steps:
            images.append(step.obs["images"])
            proprio.append(step.obs["proprio"])
            actions.append(step.action)
            old_logprobs.append(step.logprob)
            advantages.append(traj_adv)

    if not actions:
        empty = torch.empty(0)
        return empty, empty, empty.long(), empty, empty

    return (
        torch.stack(images),
        torch.stack(proprio),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(old_logprobs, dtype=torch.float32),
        torch.tensor(advantages, dtype=torch.float32),
    )


def grpo_update(
    policy: VLAPolicy,
    reference: VLAPolicy,
    optimizer: torch.optim.Optimizer,
    group: list[Trajectory],
    cfg: GRPOConfig,
) -> GRPOStats:
    images, proprio, actions, old_logprobs, advantages = _training_rows(group)
    if actions.numel() == 0:
        return GRPOStats()

    device = policy.device
    images = images.to(device)
    proprio = proprio.to(device)
    actions = actions.to(device)
    old_logprobs = old_logprobs.to(device)
    advantages = advantages.to(device)

    stats = GRPOStats(tokens=int(actions.numel()))
    policy.model.train()
    reference.model.eval()

    n = actions.numel()
    for _ in range(cfg.epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = policy.logits_for(images[idx], proprio[idx])
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(actions[idx])
                ratio = torch.exp(logprobs - old_logprobs[idx])
                unclipped = ratio * advantages[idx]
                clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                with torch.no_grad():
                    ref_logits = reference.logits_for(images[idx], proprio[idx])
                log_policy = torch.log_softmax(logits.float(), dim=-1)
                log_ref = torch.log_softmax(ref_logits.float(), dim=-1)
                kl = (log_policy.exp() * (log_policy - log_ref)).sum(dim=-1).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.kl_beta * kl - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                stats.loss += float(loss.detach())
                stats.entropy += float(entropy.detach())
                stats.kl += float(kl.detach())
                stats.clip_frac += float((ratio.sub(1.0).abs() > cfg.clip_eps).float().mean())
                stats.ratio_mean += float(ratio.mean())
                stats.ratio_max = max(stats.ratio_max, float(ratio.max()))
                raw_grad_norm = float(grad_norm)
                stats.raw_grad_norm = max(stats.raw_grad_norm, raw_grad_norm)
                stats.grad_norm = max(stats.grad_norm, min(raw_grad_norm, cfg.max_grad_norm))

    denom = max(1, cfg.epochs * ((n + cfg.batch_size - 1) // cfg.batch_size))
    stats.loss /= denom
    stats.entropy /= denom
    stats.kl /= denom
    stats.clip_frac /= denom
    stats.ratio_mean /= denom
    return stats
