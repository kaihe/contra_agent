"""
Interactive Contra Force RAM inspector with keyboard control.

Usage:
    python inspect_x_pos.py                              # Keyboard control, watch level_x
    python inspect_x_pos.py --vars level_x,x_pos,score  # Watch multiple vars
    python inspect_x_pos.py --agent path/to/model.zip   # Let trained agent play
    python inspect_x_pos.py --vars level_x --speed 1     # Original speed

Controls:
    Arrow keys: Move (UP, DOWN, LEFT, RIGHT)
    D: B button (shoot)
    Space: A button (jump)
    Enter: START
    Backspace: SELECT
    Q: Quit
"""

import argparse
import os
import sys

import stable_retro as retro
import matplotlib.pyplot as plt
import numpy as np


# NES button layout: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
KEY_MAP = {
    'd':         0,  # B (shoot)
    ' ':         8,  # A (jump)
    'up':        4,
    'down':      5,
    'left':      6,
    'right':     7,
    'enter':     3,  # START
    'backspace': 2,  # SELECT
}


def main():
    parser = argparse.ArgumentParser(description="Contra Force RAM Inspector")
    parser.add_argument("--vars", type=str, default="level_x",
                        help="Comma-separated RAM variables to track (default: level_x)")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Game speed multiplier (default: 2.0)")
    parser.add_argument("--agent", type=str, default=None,
                        help="Path to trained model (.zip) for agent play")
    args = parser.parse_args()

    GAME = "ContraForce-Nes-v0"
    STATE = "Level1"
    var_names = [v.strip() for v in args.vars.split(",")]

    # Load agent model if specified
    model = None
    if args.agent:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
        from contra_wrapper import ContraWrapper
        from stable_baselines3 import PPO
        if not os.path.exists(args.agent):
            print(f"Model not found: {args.agent}")
            return
        print(f"Loading agent: {args.agent}")
        model = PPO.load(args.agent)

    print(f"Loading {GAME}...")
    print(f"Tracking: {', '.join(var_names)}")
    base_env = retro.make(game=GAME, state=STATE,
                          use_restricted_actions=retro.Actions.FILTERED,
                          obs_type=retro.Observations.IMAGE,
                          render_mode=None)

    if model:
        env = ContraWrapper(base_env)
    else:
        env = base_env

    ret = env.reset()
    obs, info = ret if isinstance(ret, tuple) else (ret, {})

    pressed_keys = set()

    # Setup matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 7))
    raw_obs = base_env.get_screen() if model else obs
    im = ax.imshow(raw_obs)
    title = "Contra Force (Agent)" if model else "Contra Force (Keyboard)"
    ax.set_title(f"{title} - Tracking: {', '.join(var_names)}")
    info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def on_key_press(event):
        if event.key in KEY_MAP:
            pressed_keys.add(event.key)
        elif event.key == 'q':
            pressed_keys.add('quit')

    def on_key_release(event):
        if event.key in KEY_MAP:
            pressed_keys.discard(event.key)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    if model:
        print("Mode: Agent play (press Q to quit)")
    else:
        print(f"Controls: Arrow keys=Move, D=Shoot, Space=Jump, Enter=Start, Q=Quit")
    print("-" * 60)

    # NES runs at 60fps; step multiple frames per render to achieve target speed
    # e.g. 2x speed = step 4 frames per render (~30 renders/sec)
    frames_per_render = max(1, round(args.speed * 2))

    frame = 0
    try:
        while 'quit' not in pressed_keys:
            if model:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action_size = env.action_space.shape[0]
                action = np.zeros(action_size, dtype=int)
                for key in pressed_keys:
                    if key in KEY_MAP:
                        action[KEY_MAP[key]] = 1

            done = False
            for _ in range(frames_per_render):
                obs, reward, terminated, truncated, info = env.step(action)
                frame += 1
                done = terminated or truncated
                if done:
                    break

            curr_vals = {v: info.get(v, 0) for v in var_names}

            # Update display with raw screen (not preprocessed)
            raw_obs = base_env.get_screen() if model else obs
            im.set_data(raw_obs)
            var_strs = [f'{v}: {curr_vals[v]}' for v in var_names]
            info_text.set_text(f'Frame: {frame} | {" | ".join(var_strs)}')
            fig.canvas.flush_events()
            plt.pause(0.001)

            # Print values every render tick
            all_parts = [f"{v}={curr_vals[v]}" for v in var_names]
            print(f"Frame {frame:6d} | {' | '.join(all_parts)}")

            prev_vals = curr_vals

            if done:
                print("Episode finished. Resetting...")
                ret = env.reset()
                obs, info = ret if isinstance(ret, tuple) else (ret, {})
                prev_vals = {v: info.get(v, 0) for v in var_names}
                frame = 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()
        plt.close()
        print("Done.")


if __name__ == "__main__":
    main()
'''
    python rom_inspect_scripts/inspect_address.py --agent interesting_checkpoints/score_cheater.zip --vars score,x_pos
'''