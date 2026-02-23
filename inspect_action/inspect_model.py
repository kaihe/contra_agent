"""
Contra Force – Game State Inspector
====================================

Runs a trained model for one episode, captures every agent step's raw frame
with metadata, and opens an interactive matplotlib navigator (left/right
arrow keys). Press 'S' to save the emulator state at the current frame.

Usage:
    python inspect_model.py --model ../main/trained_models/ppo_contra_final.zip
"""

import argparse
import gzip
import os
import sys

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np

# Allow imports from ../main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

import stable_retro as retro
from stable_baselines3 import PPO

from contra_wrapper import ACTION_NAMES, ContraWrapper, load_config_from_model, apply_config

# =============================================================================
# CONSTANTS
# =============================================================================

GAME = "Contra-Nes"
STATE = "Level1"
STATES_DIR = os.path.join(os.path.dirname(__file__), "..", "main", "states")


# =============================================================================
# RUN EPISODE & CAPTURE
# =============================================================================

def run_episode(model_path: str) -> list[dict]:
    """Run one episode with the model deterministically, return per-step records."""

    # Load embedded config if present (action table, skip, etc.)
    config = load_config_from_model(model_path)
    if config:
        apply_config(config)
        print(f"Loaded embedded config: {config['action_names']}")

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

    records: list[dict] = []
    cumulative_reward = 0.0
    step_num = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward
        step_num += 1

        # Grab the last raw RGB frame and emulator state
        raw_frame = base_env.get_screen()  # (H, W, 3) RGB array
        emu_state = base_env.em.get_state()  # raw emulator state bytes

        records.append({
            "step": step_num,
            "action": ACTION_NAMES[action],
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "xscroll": info.get("xscroll", 0),
            "score": env.episode_score,
            "lives": info.get("lives", 0),
            "frame": raw_frame.copy(),
            "emu_state": emu_state,
        })

    env.close()
    print(f"Episode finished: {step_num} steps, score={env.episode_score}, "
          f"reward={cumulative_reward:.2f}, reason={env.episode_end_reason}")
    return records


# =============================================================================
# MATPLOTLIB NAVIGATOR
# =============================================================================

def _format_info(record: dict) -> str:
    """Build multi-line info string for a record."""
    return (
        f"Step: {record['step']}    Action: {record['action']}    "
        f"Lives: {record['lives']}\n"
        f"Reward: {record['reward']:+.2f}    "
        f"Total: {record['cumulative_reward']:.2f}    "
        f"Score: {record['score']}    xscroll: {record['xscroll']}"
    )


def launch_navigator(records: list[dict]) -> None:
    """Open matplotlib window; left/right arrows navigate frames."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    total = len(records)
    idx = [0]  # mutable container for closure

    # Size figure so the game frame renders at 1x native pixels (240x224).
    # Text panel adds ~60px below.
    dpi = 100
    frame_h, frame_w = records[0]["frame"].shape[:2]
    text_h = 60  # pixels for info box
    fig_w = frame_w / dpi
    fig_h = (frame_h + text_h) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = GridSpec(2, 1, height_ratios=[frame_h, text_h], hspace=0.05)

    ax_img = fig.add_subplot(gs[0])
    ax_img.axis("off")
    ax_img.set_position([0, text_h / (frame_h + text_h), 1, frame_h / (frame_h + text_h)])
    im = ax_img.imshow(records[0]["frame"], interpolation="nearest")

    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    txt = ax_txt.text(
        0.5, 0.5, _format_info(records[0]),
        transform=ax_txt.transAxes, fontsize=8, fontfamily="monospace",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85),
        color="white",
    )

    fig.suptitle(f"Frame 1 / {total}  |  ←/→ navigate, S = save state", fontsize=10)

    def on_key(event):
        if event.key == "right":
            idx[0] = min(idx[0] + 1, total - 1)
        elif event.key == "left":
            idx[0] = max(idx[0] - 1, 0)
        elif event.key == "home":
            idx[0] = 0
        elif event.key == "end":
            idx[0] = total - 1
        elif event.key == "s":
            _save_state(records[idx[0]])
            return
        else:
            return
        im.set_data(records[idx[0]]["frame"])
        txt.set_text(_format_info(records[idx[0]]))
        fig.suptitle(f"Frame {idx[0] + 1} / {total}  |  ←/→ navigate, S = save state", fontsize=10)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# =============================================================================
# SAVE STATE
# =============================================================================

def _save_state(record: dict) -> None:
    """Save the emulator state from a record to the project states/ dir."""
    os.makedirs(STATES_DIR, exist_ok=True)
    xscroll = record["xscroll"]
    step = record["step"]
    filename = f"Level1_x{xscroll}_step{step}.state"
    filepath = os.path.join(STATES_DIR, filename)
    with gzip.open(filepath, "wb") as f:
        f.write(record["emu_state"])
    print(f"✅ State saved: {filepath}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inspect trained model gameplay step-by-step")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .zip file")
    args = parser.parse_args()

    print("=" * 70)
    print("Contra Force – Game State Inspector")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print("=" * 70)

    records = run_episode(args.model)
    launch_navigator(records)


if __name__ == "__main__":
    main()
