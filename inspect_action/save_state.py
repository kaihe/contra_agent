"""
Save Boss-Fight Emulator State
===============================

Runs the trained model for 680 deterministic steps from Level1,
then saves the emulator state as a gzip-compressed .state file
into the retro game data directory for use as a training start point.

Usage:
    python save_state.py
    python save_state.py --steps 680 --model ../main/trained_models/distance_0.1_final.zip
"""

import argparse
import gzip
import os
import sys

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

import stable_retro as retro
from stable_baselines3 import PPO

from contra_wrapper import ContraWrapper

GAME = "Contra-Nes"
STATE = "Level1"
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "main", "trained_models", "distance_0.1_final.zip"
)


def save_boss_state(model_path: str, num_steps: int, state_name: str) -> str:
    """Run model for num_steps, then save emulator state to retro data dir."""

    base_env = retro.make(
        game=GAME,
        state=STATE,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )

    env = ContraWrapper(base_env, monitor=None, reset_round=True,
                        random_start_frames=0, skip=4, stack=4)

    model = PPO.load(model_path)

    obs, info = env.reset()
    print(f"Start — xscroll: {info.get('xscroll', 0)}, lives: {info.get('lives', 0)}")

    for step in range(1, num_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        if step % 100 == 0 or step == num_steps:
            print(f"  step {step:4d} — xscroll: {info.get('xscroll', 0)}, "
                  f"score: {env.episode_score}, lives: {info.get('lives', 0)}")

        if terminated or truncated:
            print(f"Episode ended at step {step} ({env.episode_end_reason}). "
                  "Cannot reach target step.")
            env.close()
            return ""

    # Grab raw emulator state
    raw_state = base_env.em.get_state()

    # Find the retro data directory for this game
    game_data_dir = retro.data.get_file_path(GAME, "", inttype=retro.data.Integrations.ALL)
    out_path = os.path.join(game_data_dir, f"{state_name}.state")

    with gzip.open(out_path, "wb") as f:
        f.write(raw_state)

    print(f"\nSaved state to: {out_path}")
    print(f"  xscroll at save: {info.get('xscroll', 0)}")
    print(f"  score at save:   {env.episode_score}")
    print(f"  lives at save:   {info.get('lives', 0)}")

    env.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Save emulator state for boss fight training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to trained model .zip")
    parser.add_argument("--steps", type=int, default=680,
                        help="Number of agent steps to run before saving state")
    parser.add_argument("--name", type=str, default="BossFight",
                        help="Name for the .state file (without extension)")
    args = parser.parse_args()

    print("=" * 60)
    print("Save Boss-Fight Emulator State")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Steps:  {args.steps}")
    print(f"  State:  {args.name}")
    print("=" * 60)

    path = save_boss_state(args.model, args.steps, args.name)
    if path:
        print(f"\nDone. Train with:")
        print(f"  python main/train.py --state {args.name} --name ppo_boss --timesteps 16000000")


if __name__ == "__main__":
    main()
