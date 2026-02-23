"""
Inspect Action Sequence
=======================

Run a predefined action sequence through the exact same wrapped env
used in training (create_env) and generate a GIF.
Prints score and xscroll changes at each agent step.

Edit ACTIONS below to define the sequence, then run:
    python run_sequence.py
"""

import os
import sys
import gzip

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro

# Add parent so we can import from main/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
from contra_wrapper import ACTION_NAMES, Monitor, create_env


GAME = "Contra-Nes"
STATE = "Level1"
GIFS_DIR = os.path.join(os.path.dirname(__file__), "gifs")

# Map nickname -> discrete action index (used by ContraWrapper)
NAME_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def run_sequence(actions, output_path, state=STATE):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Validate action names
    for name in actions:
        if name not in NAME_TO_IDX:
            print(f"Unknown action '{name}'. Valid: {', '.join(ACTION_NAMES)}")
            sys.exit(1)

    monitor = Monitor(240, 224, saved_path=output_path)

    # Detect custom state file path
    custom_state_data = None
    if state.endswith(".state") and os.path.isfile(state):
        with gzip.open(state, "rb") as f:
            custom_state_data = f.read()
        init_state = STATE
    else:
        init_state = state

    base_env = retro.make(
        game=GAME,
        state=init_state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env = create_env(base_env, monitor=monitor)

    obs, info = env.reset()

    # Load custom state after reset
    if custom_state_data:
        base_env.em.set_state(custom_state_data)
        base_env.data.update_ram()
        no_op = np.zeros(base_env.action_space.shape, dtype=base_env.action_space.dtype)
        base_env.step(no_op)
        info_ram = base_env.data.lookup_all()
        env.prev_xscroll = info_ram.get("xscroll", 0)
        env.prev_score = info_ram.get("score", 0)
        env.prev_lives = info_ram.get("lives", 0)
        env.max_x_reached = env.prev_xscroll
        print(f"Loaded custom state: {state} (xscroll={env.prev_xscroll})")

    prev_score = env.prev_score
    prev_xscroll = env.prev_xscroll

    print(f"Sequence: {' '.join(actions)} ({len(actions)} actions)\n")
    print(f"{'Step':<6} {'Action':<8} {'Score':>6} {'dScore':>7} {'xscroll':>8} {'dScroll':>8} {'Reward':>8}")
    print("-" * 65)
    print(f"{'init':<6} {'--':<8} {prev_score:>6} {'--':>7} {prev_xscroll:>8} {'--':>8} {'--':>8}")

    for i, name in enumerate(actions):
        action_idx = NAME_TO_IDX[name]

        obs, reward, done, trunc, info = env.step(action_idx)

        curr_score = env.episode_score
        curr_xscroll = env.prev_xscroll
        d_score = curr_score - prev_score
        d_scroll = curr_xscroll - prev_xscroll

        print(f"{i + 1:<6} {name:<8} {curr_score:>6} {d_score:>+7} {curr_xscroll:>8} {d_scroll:>+8} {reward:>+8.2f}")

        prev_score = curr_score
        prev_xscroll = curr_xscroll

        if done:
            end_reason = info.get("episode_end_reason", "unknown")
            print(f"  ** Episode ended: {end_reason} at step {i + 1} **")
            break

    env.close()
    monitor.close()

    file_size = os.path.getsize(output_path) / 1024
    print(f"\nGIF saved: {output_path} ({file_size:.1f} KB, {len(monitor.frames)} frames)")



if __name__ == "__main__":
    # Test laser rapid fire: Jump+Fire x 10 from a state with the laser equipped
    LASER_STATE = os.path.join(os.path.dirname(__file__), "..", "main", "states", "Level1_x0_step1.state")
    actions = ["JF",'F'] * 60

    seq_str = 'JF_F_60_normal'
    output_path = os.path.join(GIFS_DIR, f"seq_{seq_str}.gif")
    run_sequence(actions, output_path, state=LASER_STATE)
