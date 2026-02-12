"""
Contra Force – Game State Inspector
====================================

Runs a trained model for one episode, captures every agent step's raw frame
with metadata, and opens an interactive matplotlib navigator (left/right
arrow keys).

Usage:
    python inspect_model.py --model ../main/trained_models/ppo_contra_final.zip
"""

import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np

# Allow imports from ../main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

import stable_retro as retro
from stable_baselines3 import PPO

from contra_wrapper import ACTION_NAMES, ContraWrapper

# =============================================================================
# CONSTANTS
# =============================================================================

GAME = "Contra-Nes"
STATE = "Level1"


# =============================================================================
# RUN EPISODE & CAPTURE
# =============================================================================

def run_episode(model_path: str) -> list[dict]:
    """Run one episode with the model deterministically, return per-step records."""

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

        # Grab the last raw RGB frame from the inner env
        # env.env is the base retro env; its last observation is the raw frame
        # We can reconstruct from the wrapper internals – the easiest way is to
        # use the retro env's screen directly.
        raw_frame = base_env.get_screen()  # (H, W, 3) RGB array

        records.append({
            "step": step_num,
            "action": ACTION_NAMES[action],
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "xscroll": info.get("xscroll", 0),
            "score": env.episode_score,
            "lives": info.get("lives", 0),
            "frame": raw_frame.copy(),
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

    fig.suptitle(f"Frame 1 / {total}", fontsize=12)

    def on_key(event):
        if event.key == "right":
            idx[0] = min(idx[0] + 1, total - 1)
        elif event.key == "left":
            idx[0] = max(idx[0] - 1, 0)
        elif event.key == "home":
            idx[0] = 0
        elif event.key == "end":
            idx[0] = total - 1
        else:
            return
        im.set_data(records[idx[0]]["frame"])
        txt.set_text(_format_info(records[idx[0]]))
        fig.suptitle(f"Frame {idx[0] + 1} / {total}", fontsize=12)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


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
