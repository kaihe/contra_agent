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

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro

# Add parent so we can import from main/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
from contra_wrapper import ACTION_NAMES, Monitor, create_env

# =====================================================================
# DEFINE YOUR ACTION SEQUENCE HERE
# Valid actions: N(oop), F(ire), L(eft), R(ight), U(p), D(own), J(ump)
# =====================================================================


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

    base_env = retro.make(
        game=GAME,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env = create_env(base_env, monitor=monitor)

    obs, info = env.reset()

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


def make_filename(actions):
    """Compress action list into run-length name: ["R"]*30 + ["N"] -> "R30_N"."""
    parts = []
    i = 0
    while i < len(actions):
        action = actions[i]
        count = 1
        while i + count < len(actions) and actions[i + count] == action:
            count += 1
        parts.append(f"{action}{count}" if count > 1 else action)
        i += count
    return "_".join(parts)


if __name__ == "__main__":
    PREFIX = ["R"] * 20

    SUFFIX = ["N"] * 20
    actions =  PREFIX + ["R",'F'] * 30 + SUFFIX 

    
    seq_str = make_filename(actions)
    output_path = os.path.join(GIFS_DIR, f"seq_{seq_str}.gif")
    run_sequence(actions, output_path)
