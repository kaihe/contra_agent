"""
Contra Force – Game State Inspector
====================================

Runs a trained model or replays a human trace, captures every step's raw frame
with metadata, and opens an interactive matplotlib navigator (arrow keys).
Press 'S' to save the emulator state at the current frame.

Usage:
    python inspector.py --model ../main/trained_models/ppo_contra_final.zip
    python inspector.py --trace human_trace/win_D3072_S196_02152328.npz
    python inspector.py --trace          # auto-picks first win trace
"""

import argparse
import glob
import gzip
import os
import sys

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np

# Allow imports from ../main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

import stable_retro as retro

# =============================================================================
# CONSTANTS
# =============================================================================

GAME = "Contra-Nes"
STATE = "Level1"
STATES_DIR = os.path.join(os.path.dirname(__file__), "..", "main", "states")


# =============================================================================
# MODE 1: MODEL EPISODE
# =============================================================================

def run_model_episode(model_path: str) -> list[dict]:
    """Run one episode with the model deterministically, return per-step records."""
    from stable_baselines3 import PPO
    from contra_wrapper import ACTION_NAMES, ContraWrapper, load_config_from_model, apply_config

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

        records.append({
            "step": step_num,
            "action": ACTION_NAMES[action],
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "xscroll": info.get("xscroll", 0),
            "score": env.episode_score,
            "lives": info.get("lives", 0),
            "frame": base_env.get_screen().copy(),
            "emu_state": base_env.em.get_state(),
        })

    env.close()
    print(f"Episode finished: {step_num} steps, score={env.episode_score}, "
          f"reward={cumulative_reward:.2f}, reason={env.episode_end_reason}")
    return records


def _format_model_record(record: dict) -> str:
    return (
        f"Step: {record['step']}    Action: {record['action']}    "
        f"Lives: {record['lives']}\n"
        f"Reward: {record['reward']:+.2f}    "
        f"Total: {record['cumulative_reward']:.2f}    "
        f"Score: {record['score']}    xscroll: {record['xscroll']}"
    )


# =============================================================================
# MODE 2: HUMAN TRACE REPLAY
# =============================================================================

def replay_trace(trace_path: str) -> list[dict]:
    """Replay a recorded human trace deterministically, return per-frame records."""
    print(f"Loading trace: {trace_path}")
    data = np.load(trace_path)
    # CRITICAL: first action is a dummy zero-action from env.reset(); skip it
    # or replay will be 1 frame desynced and butterfly-effect into a death.
    actions = data["actions"][1:] if len(data["actions"]) > 1 else []

    env = retro.make(
        game=GAME,
        state=STATE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env.reset()

    records: list[dict] = []
    print(f"Replaying {len(actions)} frames...")

    for step_num, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        records.append({
            "step": step_num + 1,
            "action": action.tolist(),
            "xscroll": info.get("xscroll", 0),
            "score": info.get("score", 0),
            "lives": info.get("lives", 0),
            "frame": obs.copy(),
            "emu_state": env.em.get_state(),
        })
        if terminated or truncated:
            break

    env.close()
    print(f"Replay finished: {len(records)} frames captured.")
    return records


def _format_trace_record(record: dict) -> str:
    keys = "".join(str(x) for x in record["action"])
    return (
        f"Frame: {record['step']}    Keys: [{keys}]    "
        f"Lives: {record['lives']}\n"
        f"Score: {record['score']}    xscroll: {record['xscroll']}"
    )


# =============================================================================
# SHARED: SAVE STATE
# =============================================================================

def _save_state(record: dict) -> None:
    """Save the emulator state at the current record to main/states/."""
    os.makedirs(STATES_DIR, exist_ok=True)
    filename = f"Level1_x{record['xscroll']}_step{record['step']}.state"
    filepath = os.path.join(STATES_DIR, filename)
    with gzip.open(filepath, "wb") as f:
        f.write(record["emu_state"])
    print(f"State saved: {filepath}")


# =============================================================================
# SHARED: MATPLOTLIB NAVIGATOR
# =============================================================================

def launch_navigator(records: list[dict], format_fn) -> None:
    """Open matplotlib window. Keys: ←/→ step, PgUp/PgDn jump 60, S save state."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Prevent matplotlib's built-in 's' keybinding from opening a save-figure dialog
    matplotlib.rcParams["keymap.save"] = []

    total = len(records)
    idx = [0]

    dpi = 100
    frame_h, frame_w = records[0]["frame"].shape[:2]
    text_h = 60
    fig = plt.figure(figsize=(frame_w / dpi, (frame_h + text_h) / dpi), dpi=dpi)
    gs = GridSpec(2, 1, height_ratios=[frame_h, text_h], hspace=0.05)

    ax_img = fig.add_subplot(gs[0])
    ax_img.axis("off")
    ax_img.set_position([0, text_h / (frame_h + text_h), 1, frame_h / (frame_h + text_h)])
    im = ax_img.imshow(records[0]["frame"], interpolation="nearest")

    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    txt = ax_txt.text(
        0.5, 0.5, format_fn(records[0]),
        transform=ax_txt.transAxes, fontsize=10, fontfamily="monospace",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85),
        color="white",
    )

    def _refresh():
        rec = records[idx[0]]
        im.set_data(rec["frame"])
        txt.set_text(format_fn(rec))
        fig.suptitle(
            f"Frame {idx[0] + 1} / {total}  |  ←/→ step  PgUp/Dn jump  S save",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    _refresh()

    def on_key(event):
        if event.key == "right":
            idx[0] = min(idx[0] + 1, total - 1)
        elif event.key == "left":
            idx[0] = max(idx[0] - 1, 0)
        elif event.key == "pagedown":
            idx[0] = min(idx[0] + 60, total - 1)
        elif event.key == "pageup":
            idx[0] = max(idx[0] - 60, 0)
        elif event.key == "home":
            idx[0] = 0
        elif event.key == "end":
            idx[0] = total - 1
        elif event.key == "s":
            _save_state(records[idx[0]])
            return
        else:
            return
        _refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inspect model or human trace gameplay frame-by-frame"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str,
                       help="Path to trained model .zip file")
    group.add_argument("--trace", type=str, nargs="?", const="auto",
                       help="Path to .npz human trace (omit path to auto-pick first win trace)")
    args = parser.parse_args()

    print("=" * 70)

    if args.model:
        print("Contra Force – Model Inspector")
        print("=" * 70)
        print(f"  Model: {args.model}")
        print("=" * 70)
        records = run_model_episode(args.model)
        format_fn = _format_model_record

    else:
        trace_path = args.trace
        if trace_path == "auto":
            traces = glob.glob(
                os.path.join(os.path.dirname(__file__), "human_trace", "win_*.npz")
            )
            if not traces:
                print("No win traces found in human_trace/. Pass an explicit --trace path.")
                sys.exit(1)
            trace_path = traces[0]
        print("Contra Force – Human Trace Inspector")
        print("=" * 70)
        print(f"  Trace: {trace_path}")
        print("=" * 70)
        records = replay_trace(trace_path)
        format_fn = _format_trace_record

    print("\nStarting GUI  (←/→ step · PgUp/Dn jump 60 · S save state)")
    launch_navigator(records, format_fn)


if __name__ == "__main__":
    main()
