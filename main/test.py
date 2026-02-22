"""
Contra (NES) Model Testing
===========================

Usage:
    python test.py                                    # Test default model
    python test.py --model trained_models/ppo_contra_1000000_steps.zip
    python test.py --render                           # Also show live gameplay
"""

import os

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO

from contra_wrapper import create_env, Monitor


# =============================================================================
# CONFIG
# =============================================================================

GAME = "Contra-Nes"
STATE = "Level1"
MODEL_DIR = "trained_models"
DEFAULT_MODEL = "ppo_contra_final.zip"
OUTPUT_DIR = "recordings"


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contra Model Testing")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file")
    parser.add_argument("--state", type=str, default=STATE,
                        help="Game state to test on")
    parser.add_argument("--render", action="store_true",
                        help="Show live gameplay window")
    args = parser.parse_args()

    # Resolve model path and name
    model_path = args.model or os.path.join(MODEL_DIR, DEFAULT_MODEL)
    model_name = os.path.basename(model_path).replace(".zip", "")

    # Always record a GIF named after the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}.gif")

    print("=" * 70)
    print("Contra (NES) - Model Testing")
    print("=" * 70)
    print(f"  Game:         {GAME}")
    print(f"  State:        {args.state}")
    print(f"  Model:        {model_path}")
    print(f"  Render:       {args.render}")
    print(f"  Recording to: {output_path}")
    print("=" * 70)

    # Setup monitor (always record, optionally render)
    # NES resolution: 256x224
    monitor = Monitor(240, 224, saved_path=output_path, render=args.render)

    # Create environment
    base_env = retro.make(
        game=GAME,
        state=args.state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env = create_env(base_env, monitor=monitor)

    # Load model
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"Model not found: {model_path}")
        return

    print("\nGame Start!\n")

    # Run single episode
    obs, info = env.reset()

    done = False
    episode_reward = 0
    episode_actions = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_actions += 1

    episode_score = info.get("score", 0)
    episode_max_x = info.get("episode_max_x", 0)
    dist_rwd = info.get("episode_distance_reward", 0)
    score_rwd = info.get("episode_score_reward", 0)
    episode_steps = info.get("episode_steps", 0)
    end_reason = info.get("episode_end_reason", "unknown")

    print(f"Result: {end_reason} | Score: {episode_score} | "
          f"Reward: {episode_reward:.3f} (dist:{dist_rwd:.2f} score:{score_rwd:.2f}) | "
          f"Steps: {episode_steps} | Distance: {episode_max_x}")

    env.close()

    # Close monitor and report
    monitor.close()
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nGIF saved: {output_path} ({file_size:.1f} MB)")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Score:        {episode_score}")
    print(f"  Reward:       {episode_reward:.3f}")
    print(f"  Distance:     {episode_max_x}")
    print(f"  Model:        {model_path}")
    print(f"  GIF:          {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

