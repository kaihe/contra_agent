"""
Contra Model Testing
====================

Usage:
    python test.py                                    # Test default model
    python test.py --model trained_models/ppo_contra_1000000_steps.zip
    python test.py --episodes 50 --render             # 50 episodes with rendering
    python test.py --random                           # Random agent baseline
    python test.py --record                           # Save video recording
    python test.py --random-start 30                  # Random freeze at start
"""

import os

import cv2
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO

from contra_wrapper import ContraWrapper

# =============================================================================
# CONFIG
# =============================================================================

GAME = "Contra-Nes"
STATE = "Level1"
MODEL_DIR = "trained_models"
DEFAULT_MODEL = "ppo_contra_final.zip"
OUTPUT_DIR = "recordings"

# NES runs at 60fps, with 4-frame skip = 15 actions per second
DEFAULT_FPS = 15


# =============================================================================
# VIDEO OVERLAY
# =============================================================================

def add_overlay(frame, episode_num, total_episodes, info=None, result=None):
    """Add episode number, stats, and result overlay to frame."""
    frame = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    shadow_color = (0, 0, 0)

    # Episode number (bottom-left)
    text = f"Game {episode_num}/{total_episodes}"
    x, y = 10, frame.shape[0] - 10
    cv2.putText(frame, text, (x+1, y+1), font, font_scale, shadow_color, thickness)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    # Lives and level (top-left)
    if info:
        lives = info.get("lives", 0)
        level = info.get("level", 1)
        text = f"Lives: {lives} | Level: {level}"
        x, y = 10, 20
        cv2.putText(frame, text, (x+1, y+1), font, font_scale, shadow_color, thickness)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    # Result overlay (center)
    if result:
        result_color = (0, 255, 0) if "Complete" in result else (0, 0, 255)
        text_size = cv2.getTextSize(result, font, 1.0, 2)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = frame.shape[0] // 2
        cv2.putText(frame, result, (x+2, y+2), font, 1.0, shadow_color, 2)
        cv2.putText(frame, result, (x, y), font, 1.0, result_color, 2)

    return frame


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
    parser.add_argument("--render", action="store_true",
                        help="Render the game")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of model")
    parser.add_argument("--state", type=str, default=STATE,
                        help="Game state to test on")
    parser.add_argument("--random-start", type=int, default=0,
                        help="Max random no-op frames at episode start (0=disabled)")
    parser.add_argument("--record", action="store_true",
                        help="Save video recording of gameplay")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video filename (only with --record)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Video FPS (default: 15 for real-time speed)")
    args = parser.parse_args()

    # Check for imageio if recording
    if args.record:
        try:
            import imageio
        except ImportError:
            print("Please install imageio: pip install imageio imageio-ffmpeg")
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Contra - Model Testing")
    print("=" * 70)
    print(f"  Game:         {GAME}")
    print(f"  State:        {args.state}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Random:       {args.random}")
    print(f"  Random start: {args.random_start} frames")
    print(f"  Record:       {args.record}")
    print("=" * 70)

    # Create base environment
    base_env = retro.make(
        game=GAME,
        state=args.state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )

    # Wrap environment
    env = ContraWrapper(
        base_env,
        reset_round=True,
        rendering=args.render,
        random_start_frames=args.random_start
    )

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

    # Determine output filename if recording
    output_path = None
    if args.record:
        if args.output:
            output_path = os.path.join(OUTPUT_DIR, args.output)
        else:
            model_name = os.path.basename(model_path).replace(".zip", "")
            if args.random:
                model_name = "random"
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{args.episodes}ep.mp4")
        print(f"Recording to: {output_path}")

    print("\nGame Start!\n")

    # Run episodes
    total_score = 0
    max_progress = 0
    levels_completed = 0
    all_frames = []

    for episode in range(args.episodes):
        episode_num = episode + 1
        obs, info = env.reset()

        # Capture initial frame if recording
        if args.record:
            raw_frame = base_env.get_screen()
            frame_with_overlay = add_overlay(raw_frame, episode_num, args.episodes, info)
            all_frames.append(frame_with_overlay)

        done = False
        episode_reward = 0
        episode_progress = 0
        result = None

        while not done:
            if args.random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Track progress
            xscroll = info.get("xscroll", 0)
            if xscroll > episode_progress:
                episode_progress = xscroll

            # Check level completion or death
            if done:
                curr_level = info.get("level", 1)
                curr_lives = info.get("lives", 0)
                if curr_lives < 0:
                    result = "Game Over"
                else:
                    result = f"Level {curr_level} Complete!"
                    levels_completed += 1

            # Capture frame if recording
            if args.record:
                raw_frame = base_env.get_screen()
                frame_with_overlay = add_overlay(
                    raw_frame, episode_num, args.episodes, info,
                    result if done else None
                )
                all_frames.append(frame_with_overlay)

        # Add frames showing the result for 1 second
        if args.record:
            for _ in range(args.fps):
                frame_with_overlay = add_overlay(raw_frame, episode_num, args.episodes, info, result)
                all_frames.append(frame_with_overlay)

        total_score += info.get("score", 0)
        if episode_progress > max_progress:
            max_progress = episode_progress

        print(f"Episode {episode_num}: {result} | Progress: {episode_progress} | Reward: {episode_reward:.3f}")

    env.close()

    # Save video if recording
    if args.record:
        import imageio
        print("=" * 70)
        print(f"Saving video with {len(all_frames)} frames...")

        frames_array = []
        for frame in all_frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            frames_array.append(frame_rgb)

        imageio.mimsave(output_path, frames_array, fps=args.fps)
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved: {output_path} ({file_size:.1f} MB)")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Levels Completed: {levels_completed}/{args.episodes}")
    print(f"  Max Progress:     {max_progress}")
    print(f"  Total Score:      {total_score}")
    if args.random:
        print(f"  Agent:            Random")
    else:
        print(f"  Agent:            {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
