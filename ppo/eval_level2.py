"""
Contra (NES) Level-2 checkpoint evaluation
==========================================

Evaluate a trained Level-2 model across the indoor anchor states, reporting per
state mean reward, mean enemy HP hits, mean core_broken, and win rate.

Usage:
    python ppo/eval_level2.py --model tmp/ppo/level2_e1_baseline/level2_e1_baseline_77000000_steps.zip
    python ppo/eval_level2.py --model <ckpt> --episodes 20 --deterministic
"""

import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import contra  # noqa: F401  registers custom ROM integration
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO

from contra_wrapper import (
    ACTION_SKIP,
    RandomStateWrapper,
    apply_config,
    create_env,
    load_config_from_model,
)

GAME = "Contra-Nes"
DEFAULT_STATES_GLOB = "ppo/states/level2_*.state"


def _ckpt_step(path: str) -> int:
    base = os.path.basename(path)
    try:
        return int(base.rsplit("_steps.zip", 1)[0].rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_states(arg_states, train_config) -> list[str]:
    for candidate in (arg_states, train_config.get("states"),
                      sorted(glob.glob(DEFAULT_STATES_GLOB))):
        if not candidate:
            continue
        expanded = []
        for s in candidate:
            expanded.extend(sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s])
        existing = [s for s in expanded if os.path.isfile(s)]
        if existing:
            return existing
    raise FileNotFoundError("No level-2 anchor .state files found to evaluate on.")


def run_episode(env, model, deterministic: bool):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return {
        "reward": info.get("episode_reward", 0.0),
        "enemy_hp": info.get("episode_enemy_hp_cost", 0.0),
        "core_broken": info.get("episode_core_broken", 0.0),
        "end": info.get("episode_end_reason", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Contra Level-2 checkpoint per anchor state")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--states", type=str, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per anchor state")
    parser.add_argument("--deterministic", action="store_true", help="Greedy actions instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    config = load_config_from_model(args.model) or {}
    train_config = config.get("train_config", {})
    if config:
        apply_config(config)
    states = resolve_states(args.states, train_config)

    np.random.seed(args.seed)

    print("=" * 84)
    print("Contra (NES) - Level-2 Checkpoint Evaluation")
    print("=" * 84)
    print(f"  Model:        {args.model}  (step {_ckpt_step(args.model):,})")
    print(f"  Anchors:      {len(states)}")
    print(f"  Episodes:     {args.episodes} per anchor  ({'greedy' if args.deterministic else 'stochastic'})")
    print(f"  skip={config.get('skip', ACTION_SKIP)} stack={config.get('stack', 3)} "
          f"max_steps={train_config.get('max_episode_steps', 4000)}")
    print("=" * 84)

    model = PPO.load(args.model)

    base_env = retro.make(
        game=GAME, state="Level1",
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    rsw = RandomStateWrapper(base_env, states=states)
    all_data = list(rsw.state_data)
    env = create_env(
        rsw,
        random_start_frames=0,
        warmup_frames=train_config.get("warmup_frames", 240),
        skip=config.get("skip", ACTION_SKIP),
        stack=config.get("stack", 3),
        max_episode_steps=train_config.get("max_episode_steps", 4000),
        reward_weights=train_config.get("reward_weights"),
    )

    header = (f"{'anchor':<36}{'eps':>5}{'win%':>7}{'reward':>10}"
              f"{'enemy_hp':>10}{'core':>8}")
    rows = []
    tot_reward, tot_hp, tot_core, tot_win, tot_n = [], [], [], 0, 0

    for idx, state_path in enumerate(states):
        rsw.state_data = [all_data[idx]]  # pin this anchor for every reset
        name = os.path.basename(state_path).replace(".state", "")

        rewards, hps, cores, wins = [], [], [], 0
        for _ in range(args.episodes):
            r = run_episode(env, model, args.deterministic)
            rewards.append(r["reward"])
            hps.append(r["enemy_hp"])
            cores.append(r["core_broken"])
            wins += int(r["end"] == "win")

        rows.append(
            f"{name:<36}{args.episodes:>5}{wins / args.episodes * 100:>6.0f}%"
            f"{np.mean(rewards):>10.1f}{np.mean(hps):>10.1f}{np.mean(cores):>8.2f}"
        )
        tot_reward += rewards
        tot_hp += hps
        tot_core += cores
        tot_win += wins
        tot_n += args.episodes

    env.close()

    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(row)
    print("-" * len(header))
    print(f"{'OVERALL (mean)':<36}{tot_n:>5}{tot_win / max(tot_n, 1) * 100:>6.0f}%"
          f"{np.mean(tot_reward):>10.1f}{np.mean(tot_hp):>10.1f}{np.mean(tot_core):>8.2f}")


if __name__ == "__main__":
    main()
