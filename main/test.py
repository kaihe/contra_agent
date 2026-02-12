"""
Contra (NES) Model Testing
===========================

Usage:
    python test.py                                    # Test default model
    python test.py --model trained_models/ppo_contra_1000000_steps.zip
    python test.py --episodes 50                      # 50 episodes
    python test.py --random                           # Random agent baseline
    python test.py --record                           # Save video recording
    python test.py --random-start 30                  # Random freeze at start
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
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of model")
    parser.add_argument("--state", type=str, default=STATE,
                        help="Game state to test on")
    parser.add_argument("--random-start", type=int, default=0,
                        help="Max random no-op frames at episode start (0=disabled)")
    parser.add_argument("--render", action="store_true",
                        help="Show live gameplay window")
    parser.add_argument("--record", action="store_true",
                        help="Save video recording via ffmpeg")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video filename (only with --record)")
    args = parser.parse_args()

    print("=" * 70)
    print("Contra (NES) - Model Testing")
    print("=" * 70)
    print(f"  Game:         {GAME}")
    print(f"  State:        {args.state}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Random:       {args.random}")
    print(f"  Random start: {args.random_start} frames")
    print(f"  Render:       {args.render}")
    print(f"  Record:       {args.record}")
    print("=" * 70)

    # Setup monitor (render and/or record)
    monitor = None
    output_path = None
    if args.render or args.record:
        if args.record:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model_path = args.model or os.path.join(MODEL_DIR, DEFAULT_MODEL)
            if args.output:
                output_path = os.path.join(OUTPUT_DIR, args.output)
            else:
                model_name = os.path.basename(model_path).replace(".zip", "")
                if args.random:
                    model_name = "random"
                output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{args.episodes}ep.gif")
            print(f"Recording to: {output_path}")
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
    env = create_env(base_env, monitor=monitor,
                     random_start_frames=args.random_start)

    # Load model
    model = None
    model_path = args.model or os.path.join(MODEL_DIR, DEFAULT_MODEL)
    if not args.random:
        if os.path.exists(model_path):
            print(f"Loading model: {model_path}")
            model = PPO.load(model_path)
        else:
            print(f"Model not found: {model_path}")
            print("Running with random actions instead.")
            args.random = True

    print("\nGame Start!\n")

    # Run episodes
    total_score = 0
    max_score = 0

    for episode in range(args.episodes):
        episode_num = episode + 1
        obs, info = env.reset()

        done = False
        episode_reward = 0
        episode_actions = 0

        while not done:
            if args.random:
                action = env.action_space.sample()
            else:
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
        total_score += episode_score
        if episode_score > max_score:
            max_score = episode_score

        print(f"Episode {episode_num}: {end_reason} | Score: {episode_score} | "
              f"Reward: {episode_reward:.3f} (dist:{dist_rwd:.2f} score:{score_rwd:.2f}) | "
              f"Steps: {episode_steps} | Distance: {episode_max_x}")

    env.close()

    # Close monitor
    if monitor:
        monitor.close()
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nVideo saved: {output_path} ({file_size:.1f} MB)")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    avg_score = total_score / args.episodes
    print(f"  Avg Score:        {avg_score:.0f}")
    print(f"  Max Score:        {max_score}")
    print(f"  Total Score:      {total_score}")
    if args.random:
        print(f"  Agent:            Random")
    else:
        print(f"  Agent:            {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
