"""
Contra (NES) checkpoint evaluation
==================================

Evaluate a trained model across the Level 1 anchor states, reporting per-state
progress (level-aware: xscroll pixels on side-scroll levels, screen/room number
indoors & climbing), enemy HP cost, episode reward, and game result.

Usage:
    # Newest level1_win checkpoint, anchors embedded in the model, 10 eps each
    python ppo/test.py

    python ppo/test.py --model tmp/ppo/checkpoints/level1_win/level1_win_8000000_steps.zip
    python ppo/test.py --episodes 20 --deterministic
    python ppo/test.py --states ppo/states/Level1_x0_step1.state --gif
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
    Monitor,
    RandomStateWrapper,
    apply_config,
    create_env,
    load_config_from_model,
)

GAME = "Contra-Nes"
DEFAULT_CKPT_GLOB = "tmp/ppo/checkpoints/level1_win/level1_win_*_steps.zip"
DEFAULT_STATES_GLOB = "ppo/states/*.state"
GIF_DIR = "tmp/ppo/eval"


def _ckpt_step(path: str) -> int:
    """Integer step from '<prefix>_<steps>_steps.zip' (for numeric sorting)."""
    base = os.path.basename(path)
    try:
        return int(base.rsplit("_steps.zip", 1)[0].rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_model(arg: str | None) -> str:
    if arg:
        return arg
    candidates = glob.glob(DEFAULT_CKPT_GLOB)
    final = "tmp/ppo/checkpoints/level1_win/level1_win_final.zip"
    if os.path.exists(final):
        candidates.append(final)
    if not candidates:
        raise FileNotFoundError(
            f"No model given and none found at {DEFAULT_CKPT_GLOB}"
        )
    return max(candidates, key=_ckpt_step)


def resolve_states(arg_states: list[str] | None, train_config: dict) -> list[str]:
    # Priority: explicit --states, then the list embedded in the checkpoint,
    # then whatever anchor files exist on disk.
    for candidate in (arg_states, train_config.get("states"), sorted(glob.glob(DEFAULT_STATES_GLOB))):
        if candidate:
            existing = [s for s in candidate if os.path.isfile(s)]
            if existing:
                return existing
    raise FileNotFoundError("No anchor .state files found to evaluate on.")


def run_episode(env, model, deterministic: bool):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return {
        "progress": info.get("episode_progress", 0),
        "enemy_hp_cost": info.get("episode_enemy_hp_cost", 0.0),
        "reward": info.get("episode_reward", 0.0),
        "steps": info.get("episode_steps", 0),
        "end": info.get("episode_end_reason", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Contra checkpoint per anchor state")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint path (default: newest level1_win checkpoint)")
    parser.add_argument("--states", type=str, nargs="+", default=None,
                        help="Anchor .state files (default: embedded in model, else ppo/states/*)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per anchor state")
    parser.add_argument("--deterministic", action="store_true",
                        help="Greedy actions instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true",
                        help="Record one episode per state to tmp/ppo/eval/")
    args = parser.parse_args()

    model_path = resolve_model(args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load embedded config first: it sets the action tables and the wrapper
    # settings the model was trained with (skip, stack, reward weights, ...).
    config = load_config_from_model(model_path) or {}
    train_config = config.get("train_config", {})
    if config:
        apply_config(config)
    states = resolve_states(args.states, train_config)

    np.random.seed(args.seed)

    print("=" * 78)
    print("Contra (NES) - Checkpoint Evaluation")
    print("=" * 78)
    print(f"  Model:        {model_path}  (step {_ckpt_step(model_path):,})")
    print(f"  Anchors:      {len(states)}")
    print(f"  Episodes:     {args.episodes} per anchor  ({'greedy' if args.deterministic else 'stochastic'})")
    print(f"  skip={config.get('skip', ACTION_SKIP)} stack={config.get('stack', 3)} "
          f"max_steps={train_config.get('max_episode_steps', 2000)}")
    print("=" * 78)

    model = PPO.load(model_path)

    # One emulator instance for the whole run (stable_retro allows only one per
    # process). RandomStateWrapper holds all anchors; we pin it to one anchor at
    # a time by swapping its preloaded state_data.
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
        max_episode_steps=train_config.get("max_episode_steps", 2000),
        reward_weights=train_config.get("reward_weights"),
    )

    os.makedirs(GIF_DIR, exist_ok=True)
    header = (f"{'anchor':<34}{'win':>5}{'die':>5}{'over':>5}{'tout':>5}"
              f"{'prog(mean)':>10}{'prog(max)':>9}{'hp':>8}{'reward':>9}")
    rows = []
    overall = {"win": 0, "death": 0, "game_over": 0, "time_out": 0, "n": 0}

    for idx, state_path in enumerate(states):
        rsw.state_data = [all_data[idx]]  # pin this anchor for every reset
        name = os.path.basename(state_path).replace(".state", "")

        monitor = None
        if args.gif:
            monitor = Monitor(240, 224,
                              saved_path=os.path.join(GIF_DIR, f"{name}.gif"),
                              render=False)
            monitor.skip = config.get("skip", ACTION_SKIP)

        ends = {"win": 0, "death": 0, "game_over": 0, "time_out": 0}
        progresses, hp_costs, rewards = [], [], []
        for ep in range(args.episodes):
            env.monitor = monitor if (monitor and ep == 0) else None
            r = run_episode(env, model, args.deterministic)
            ends[r["end"]] = ends.get(r["end"], 0) + 1
            progresses.append(r["progress"])
            hp_costs.append(r["enemy_hp_cost"])
            rewards.append(r["reward"])
        if monitor:
            monitor.close()

        for k in overall:
            if k in ends:
                overall[k] += ends[k]
        overall["n"] += args.episodes

        rows.append(
            f"{name:<34}{ends['win']:>5}{ends['death']:>5}{ends['game_over']:>5}{ends['time_out']:>5}"
            f"{np.mean(progresses):>10.0f}{np.max(progresses):>9.0f}"
            f"{np.mean(hp_costs):>8.1f}{np.mean(rewards):>9.1f}"
        )

    env.close()

    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(row)
    print("-" * len(header))
    win_rate = overall["win"] / max(overall["n"], 1)
    print(f"\nOverall: {overall['n']} episodes | "
          f"win {overall['win']} ({win_rate:.0%}) | "
          f"death {overall['death']} | "
          f"game_over {overall['game_over']} | time_out {overall['time_out']}")
    if args.gif:
        print(f"GIFs: {GIF_DIR}/<anchor>.gif")


if __name__ == "__main__":
    main()
