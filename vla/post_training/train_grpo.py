"""Run Level-1 MC-GRPO post-training from a BC checkpoint."""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .env_wrappers import VLAEnv
from .grpo import GRPOConfig, grpo_update
from .policy import VLAPolicy, load_bc_policy
from .rollout import RolloutConfig, collect_group
from .trajectory import Trajectory

DEFAULT_CONFIG = "vla/post_training/grpo.yaml"


def _load_yaml_defaults(config_path: str) -> dict:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        defaults = yaml.safe_load(f) or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return defaults


def _best_safe(group: list[Trajectory]) -> Trajectory | None:
    safe = [traj for traj in group if traj.steps and not traj.died]
    if not safe:
        return None
    return max(safe, key=lambda traj: traj.total_reward)


def _log_group(
    group_idx: int,
    group: list[Trajectory],
    stats,
    rewinds: int,
    accumulated_reward: float,
) -> None:
    returns = np.array([traj.total_reward for traj in group], dtype=np.float32)
    death_rate = sum(traj.died for traj in group) / max(1, len(group))
    best = float(returns.max()) if len(returns) else 0.0
    max_distance = max(
        (step.info.get("xscroll", 0) for traj in group for step in traj.steps),
        default=0,
    )
    print(
        f"{group_idx:05d}\t{best:.1f}\t{max_distance}\t{accumulated_reward:.1f}\t{death_rate:.2f}\t"
        f"{stats.loss:.4f}\t{stats.kl:.4f}\t{stats.entropy:.3f}\t"
        f"{stats.clip_frac:.2f}\t{stats.grad_norm:.2f}\t{stats.raw_grad_norm:.2f}\t{rewinds}",
        flush=True,
    )


def _log_tensorboard(
    writer: SummaryWriter | None,
    group_idx: int,
    group: list[Trajectory],
    stats,
    rewinds: int,
    accumulated_reward: float,
    committed_actions: int,
    commit_len: int,
    clears: int,
    temperature: float,
    collect_s: float,
    update_s: float,
) -> None:
    if writer is None:
        return

    returns = np.array([traj.total_reward for traj in group], dtype=np.float32)
    death_rate = sum(traj.died for traj in group) / max(1, len(group))
    max_distance = max(
        (step.info.get("xscroll", 0) for traj in group for step in traj.steps),
        default=0,
    )
    actions = np.array([step.action for traj in group for step in traj.steps], dtype=np.int64)

    writer.add_scalar("rollout/return_best", float(returns.max()) if len(returns) else 0.0, group_idx)
    writer.add_scalar("rollout/return_mean", float(returns.mean()) if len(returns) else 0.0, group_idx)
    writer.add_scalar("rollout/return_std", float(returns.std()) if len(returns) else 0.0, group_idx)
    writer.add_scalar("rollout/death_rate", death_rate, group_idx)
    writer.add_scalar("rollout/all_death", float(death_rate >= 1.0), group_idx)
    writer.add_scalar("rollout/max_xscroll", max_distance, group_idx)
    writer.add_scalar("rollout/tokens", stats.tokens, group_idx)

    writer.add_scalar("train/loss", stats.loss, group_idx)
    writer.add_scalar("train/kl", stats.kl, group_idx)
    writer.add_scalar("train/entropy", stats.entropy, group_idx)
    writer.add_scalar("train/clip_frac", stats.clip_frac, group_idx)
    writer.add_scalar("train/ratio_mean", stats.ratio_mean, group_idx)
    writer.add_scalar("train/ratio_max", stats.ratio_max, group_idx)
    writer.add_scalar("train/grad_norm", stats.grad_norm, group_idx)
    writer.add_scalar("train/raw_grad_norm", stats.raw_grad_norm, group_idx)

    writer.add_scalar("search/rewinds", rewinds, group_idx)
    writer.add_scalar("search/clears", clears, group_idx)
    writer.add_scalar("search/committed_actions", committed_actions, group_idx)
    writer.add_scalar("search/commit_len", commit_len, group_idx)
    writer.add_scalar("search/accumulated_reward", accumulated_reward, group_idx)
    writer.add_scalar("search/temperature", temperature, group_idx)

    writer.add_scalar("time/collect_s", collect_s, group_idx)
    writer.add_scalar("time/update_s", update_s, group_idx)
    writer.add_scalar("time/group_s", collect_s + update_s, group_idx)

    if actions.size:
        writer.add_histogram("rollout/actions", actions, group_idx)


def _log_header() -> None:
    print(
        "group\tbest\tmax_d\trwd\tdeath\tloss\tkl\tent\tclip\tgrad\traw_grad\trewinds",
        flush=True,
    )


def _save_checkpoint(path: str, policy: VLAPolicy, optimizer: torch.optim.Optimizer, group_idx: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "group": group_idx,
            "model": policy.model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def train_with_search(
    env: VLAEnv,
    policy: VLAPolicy,
    reference: VLAPolicy,
    optimizer: torch.optim.Optimizer,
    rollout_cfg: RolloutConfig,
    grpo_cfg: GRPOConfig,
    writer: SummaryWriter | None = None,
) -> None:
    os.makedirs(rollout_cfg.out_dir, exist_ok=True)
    env.reset(rollout_cfg.level)
    level_start_state = env.snapshot()
    committed_state = level_start_state
    committed_actions: list[int] = []
    committed_states: list[bytes] = []
    committed_rewards: list[float] = []
    accumulated_reward = 0.0
    clears = 0
    rewinds = 0
    temperature = rollout_cfg.temperature
    t0 = time.time()
    _log_header()

    for group_idx in range(1, rollout_cfg.max_groups + 1):
        if len(committed_actions) >= rollout_cfg.max_actions:
            break

        collect_t0 = time.time()
        group = collect_group(env, policy, committed_state, rollout_cfg, temperature=temperature)
        collect_s = time.time() - collect_t0

        update_t0 = time.time()
        stats = grpo_update(policy, reference, optimizer, group, grpo_cfg)
        update_s = time.time() - update_t0
        best = _best_safe(group)

        if best is None:
            n = len(committed_actions)
            rewind_to = 0
            if n:
                rewind_back = random.randint(1, min(rollout_cfg.max_rewind, n))
                rewind_to = n - rewind_back
            committed_actions = committed_actions[:rewind_to]
            committed_states = committed_states[:rewind_to]
            committed_rewards = committed_rewards[:rewind_to]
            committed_state = committed_states[-1] if committed_states else level_start_state
            accumulated_reward = committed_rewards[-1] if committed_rewards else 0.0
            rewinds += 1
            temperature = rollout_cfg.high_temperature
            _log_group(group_idx, group, stats, rewinds, accumulated_reward)
            _log_tensorboard(
                writer,
                group_idx,
                group,
                stats,
                rewinds,
                accumulated_reward,
                len(committed_actions),
                0,
                clears,
                temperature,
                collect_s,
                update_s,
            )
            continue

        temperature = rollout_cfg.temperature
        n = len(best.actions)
        commit_n = random.randint(n // 2, n) if n >= 2 else n
        actual_commit_n = 0
        env.restore(committed_state)
        for action in best.actions[:commit_n]:
            _, reward, status, _ = env.step(action)
            if status == VLAEnv.DEAD:
                break
            committed_actions.append(action)
            actual_commit_n += 1
            accumulated_reward += reward
            committed_state = env.snapshot()
            committed_states.append(committed_state)
            committed_rewards.append(accumulated_reward)
            if status == VLAEnv.DONE:
                clears += 1
                print(
                    f"cleared {rollout_cfg.level} #{clears} after "
                    f"{len(committed_actions)} committed actions",
                    flush=True,
                )
                _save_checkpoint(os.path.join(rollout_cfg.out_dir, "clear.pt"), policy, optimizer, group_idx)
                env.reset(rollout_cfg.level)
                level_start_state = env.snapshot()
                committed_state = level_start_state
                committed_actions = []
                committed_states = []
                committed_rewards = []
                accumulated_reward = 0.0
                temperature = rollout_cfg.temperature
                break

        _log_group(group_idx, group, stats, rewinds, accumulated_reward)
        _log_tensorboard(
            writer,
            group_idx,
            group,
            stats,
            rewinds,
            accumulated_reward,
            len(committed_actions),
            actual_commit_n,
            clears,
            temperature,
            collect_s,
            update_s,
        )
        if group_idx % rollout_cfg.save_every == 0:
            _save_checkpoint(os.path.join(rollout_cfg.out_dir, "last.pt"), policy, optimizer, group_idx)
            if writer is not None:
                writer.flush()

    _save_checkpoint(os.path.join(rollout_cfg.out_dir, "last.pt"), policy, optimizer, rollout_cfg.max_groups)
    print(
        f"stopped after {time.time() - t0:.1f}s, "
        f"clears={clears}, current_run_actions={len(committed_actions)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="ContraVLA Level-1 MC-GRPO trainer")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file")
    p.add_argument("--checkpoint", default=None, help="BC checkpoint, usually tmp/vla/<name>/best.pt")
    p.add_argument("--out_dir", default="tmp/vla_grpo")
    p.add_argument("--log_dir", default=None, help="TensorBoard log directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--rollout_len", type=int, default=48)
    p.add_argument("--max_groups", type=int, default=10_000)
    p.add_argument("--max_actions", type=int, default=4000)
    p.add_argument("--max_rewind", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--high_temperature", type=float, default=1.2)
    p.add_argument("--grpo_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.02)
    p.add_argument("--entropy_coef", type=float, default=0.001)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=25)

    partial_args, _ = p.parse_known_args()
    yaml_defaults = _load_yaml_defaults(partial_args.config)
    if yaml_defaults:
        p.set_defaults(**yaml_defaults)

    args = p.parse_args()
    if args.checkpoint is None:
        p.error("--checkpoint is required, either on the CLI or in the YAML config")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.log_dir is None:
        args.log_dir = os.path.join(args.out_dir, "tensorboard")

    model = load_bc_policy(args.checkpoint, device, dropout=0.0, freeze_vlm=True)
    ref_model = load_bc_policy(args.checkpoint, device, dropout=0.0, freeze_vlm=True).eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    policy = VLAPolicy(model, device, level_id=0)
    reference = VLAPolicy(ref_model, device, level_id=0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    rollout_cfg = RolloutConfig(
        group_size=args.group_size,
        rollout_len=args.rollout_len,
        max_rewind=args.max_rewind,
        max_actions=args.max_actions,
        max_groups=args.max_groups,
        temperature=args.temperature,
        high_temperature=args.high_temperature,
        save_every=args.save_every,
        out_dir=args.out_dir,
    )
    grpo_cfg = GRPOConfig(
        epochs=args.grpo_epochs,
        batch_size=args.batch_size,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )

    env = VLAEnv(level=rollout_cfg.level)
    writer = SummaryWriter(args.log_dir)
    writer.add_text("config/checkpoint", args.checkpoint, 0)
    writer.add_text("config/out_dir", args.out_dir, 0)
    writer.add_text("config/args", f"```yaml\n{yaml.safe_dump(vars(args), sort_keys=True)}\n```", 0)
    print(f"TensorBoard logs: {args.log_dir}", flush=True)
    try:
        train_with_search(env, policy, reference, optimizer, rollout_cfg, grpo_cfg, writer)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
