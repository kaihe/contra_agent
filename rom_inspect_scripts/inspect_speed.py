"""
Measure per-step speed (x_pos diff) using a trained agent.

Based on test.py - uses the same env setup so agent behaves identically.
Speed = x_pos difference between consecutive wrapper steps.

Usage:
    python inspect_speed.py --model ../main/trained_models/ppo_contra_final.zip
    python inspect_speed.py --model ../interesting_checkpoints/score_cheater.zip --episodes 5
    python inspect_speed.py --model model.zip --render --fps 30
    python inspect_speed.py --random --episodes 3
"""

import os
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
from contra_wrapper import ContraWrapper
from stable_baselines3 import PPO

GAME = "ContraForce-Nes-v0"
STATE = "Level1"
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'main', 'trained_models')
DEFAULT_MODEL = "ppo_contra_final.zip"
DEFAULT_FPS = 15


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure agent speed (x_pos diff per step)")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--random", action="store_true", help="Use random actions")
    parser.add_argument("--state", type=str, default=STATE, help="Game state")
    parser.add_argument("--random-start", type=int, default=0, help="Max random no-op frames at start")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Render FPS (default: 15)")
    args = parser.parse_args()

    print("=" * 60)
    print("Contra - Speed Inspector")
    print("=" * 60)

    base_env = retro.make(
        game=GAME,
        state=args.state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
    )

    env = ContraWrapper(
        base_env,
        reset_round=True,
        rendering=args.render,
        random_start_frames=args.random_start,
        render_fps=args.fps,
    )

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

    print(f"  Episodes: {args.episodes}")
    print(f"  Render:   {args.render}")
    print("=" * 60)

    all_speeds = []

    for episode in range(args.episodes):
        ep_num = episode + 1
        obs, info = env.reset()
        prev_x = info.get("x_pos", 0)
        done = False
        step = 0
        ep_speeds = []

        print(f"\nEpisode {ep_num}/{args.episodes}")
        print(f"{'Step':>6} | {'x_pos':>6} | {'diff':>6}")
        print("-" * 30)

        while not done:
            if args.random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            curr_x = info.get("x_pos", prev_x)
            diff = curr_x - prev_x
            ep_speeds.append(diff)

            print(f"{step:6d} | {curr_x:6d} | {diff:+6d}")

            prev_x = curr_x

        all_speeds.extend(ep_speeds)
        pos = [s for s in ep_speeds if s > 0]

        print(f"\nEpisode {ep_num} stats ({len(ep_speeds)} steps):")
        print(f"  Max diff:  {max(ep_speeds)}")
        print(f"  Min diff:  {min(ep_speeds)}")
        if pos:
            print(f"  Avg (pos): {np.mean(pos):.2f}")
            print(f"  p50 (pos): {np.percentile(pos, 50):.1f}")
            print(f"  p90 (pos): {np.percentile(pos, 90):.1f}")
            print(f"  p99 (pos): {np.percentile(pos, 99):.1f}")

    env.close()

    if not all_speeds:
        return

    pos_all = [s for s in all_speeds if s > 0]
    print(f"\n{'='*60}")
    print("OVERALL")
    print(f"{'='*60}")
    print(f"  Total steps:  {len(all_speeds)}")
    print(f"  Max diff:     {max(all_speeds)}")
    print(f"  Min diff:     {min(all_speeds)}")
    if pos_all:
        print(f"  Avg (pos):    {np.mean(pos_all):.2f}")
        print(f"  p50 (pos):    {np.percentile(pos_all, 50):.1f}")
        print(f"  p90 (pos):    {np.percentile(pos_all, 90):.1f}")
        print(f"  p99 (pos):    {np.percentile(pos_all, 99):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
