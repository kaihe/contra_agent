"""
plot_trace_map.py — Draw a winning trace as the agent's 2D path through a level.

Replays a saved mc_search trace (actions + initial_state) and records the
player's absolute position at every step, with the level-start pinned to the
origin (0, 0). The result is the agent's movement as a 2D trace on a map; points
where specific buttons are pressed (fire / jump by default) are marked on top of
the path, so you can see *where* in the level the agent shoots or jumps.

Coordinates (player 1):
  world_x = xscroll camera (ram[$64:$65], big-endian) + on-screen sprite x ($0334)
  world_y = -on-screen sprite y ($031A)  (NES sprite y grows downward, so we negate
                                          it: height is up-positive, like a normal plot)
Both are offset by their step-0 value, so the trace starts at (0, 0). world_x is
the same horizontal-progress quantity the reward uses for "forward" levels, so
the x axis is "pixels into the level".

Artifacts are written under tmp/ (CLAUDE.md).

Usage:
    python synthetic/plot_trace_map.py --input "tmp/mc_trace/level1/*.npz"
    python synthetic/plot_trace_map.py --input TRACE.npz --buttons F,J,U,D,L
    python synthetic/plot_trace_map.py --input "..." --no-path   # button scatter only
"""

import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore", message=".*Gym.*")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: render straight to a file
import matplotlib.pyplot as plt
import stable_retro as retro

from contra.replay import rewind_state, step_env, GAME
from contra.reward import xscroll

# Player-1 on-screen sprite position (ram.asm: SPRITE_X_POS $0334, SPRITE_Y_POS $031A).
ADDR_SPRITE_X = 0x0334
ADDR_SPRITE_Y = 0x031A

# Action-vector bit per button nickname (matches the action table / search_reward).
BUTTON_BIT = {"F": 0, "U": 4, "D": 5, "L": 6, "R": 7, "J": 8}

# How each marked button is drawn: (matplotlib marker, colour, label).
_MARKER_STYLE = {
    "F": ("o", "#d62728", "fire"),
    "J": ("^", "#1f77b4", "jump"),
    "U": ("v", "#2ca02c", "up"),
    "D": ("s", "#9467bd", "down"),
    "L": ("D", "#ff7f0e", "left"),
}


def replay_positions(actions: np.ndarray, initial_state: bytes) -> np.ndarray:
    """Replay `actions` from `initial_state`, return (N, 2) world (x, y) per step.

    Position is read *before* each action, i.e. the spot the player was standing
    when that step's buttons were pressed — so a fire/jump marker lands where the
    shot/jump was made.
    """
    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial_state)

    pos = np.empty((len(actions), 2), dtype=np.int64)
    for i, act in enumerate(actions):
        ram = env.unwrapped.get_ram()
        # Negate sprite y so the stored coordinate is up-positive (NES y is down-positive).
        pos[i] = (xscroll(ram) + int(ram[ADDR_SPRITE_X]), -int(ram[ADDR_SPRITE_Y]))
        step_env(env, act)
    env.close()

    pos -= pos[0]  # pin the level start to the origin
    return pos


def plot_traces(traces: list[tuple[np.ndarray, np.ndarray]], buttons: list[str],
                out_path: str, title: str, show_path: bool = True) -> None:
    """Overlay one or more (pos, actions) traces on a shared map.

    Each trace's path gets its own colour (unless ``show_path`` is False);
    fire/jump (etc.) markers are pooled across all traces, so the marker clouds
    show *where* in the level a button tends to be pressed across runs.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = plt.colormaps["turbo"]
    n = len(traces)

    handles = [Line2D([], [], color="k", marker="*", ls="", ms=11, label="start")]

    # Paths: one colour per trace (faint so overlaps stay readable). Skipped when
    # show_path is False, leaving a pure button-press distribution scatter.
    if show_path:
        for i, (pos, _) in enumerate(traces):
            ax.plot(pos[:, 0], pos[:, 1], "-", lw=0.9, alpha=0.55 if n > 1 else 0.9,
                    color=cmap((i + 0.5) / n), zorder=1)
        handles.insert(0, Line2D([], [], color="0.4", lw=1.2, label=f"path × {n}"))

    # Markers: pool every position where the button is pressed, across traces.
    for b in buttons:
        marker, color, name = _MARKER_STYLE[b]
        hits = [pos[actions[:, BUTTON_BIT[b]] == 1] for pos, actions in traces]
        pts = np.concatenate(hits) if hits else np.empty((0, 2))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker, s=16,
                       alpha=0.45 if n > 1 else 0.8, lw=0, zorder=3)
            handles.append(Line2D([], [], color=color, marker=marker, ls="",
                                  ms=6, label=f"{name} ({len(pts)})"))

    ax.scatter(0, 0, c="k", marker="*", s=130, zorder=4)  # shared origin
    ax.set_xlabel("x — pixels into the level (level start = 0)")
    ax.set_ylabel("y — height relative to start (up +)")
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Trace NPZ or glob; every match is replayed and overlaid")
    ap.add_argument("--output", default=None,
                    help="Output PNG (default: tmp/trace_map_<name>.png)")
    ap.add_argument("--buttons", default="F,J",
                    help="Comma-separated buttons to mark (F,J,U,D,L); default F,J")
    ap.add_argument("--no-path", action="store_true",
                    help="Hide the movement path; show only the button-press scatter")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.input))
    if not matches:
        raise SystemExit(f"no trace matches {args.input!r}")

    buttons = [b.strip().upper() for b in args.buttons.split(",") if b.strip()]
    unknown = [b for b in buttons if b not in _MARKER_STYLE]
    if unknown:
        raise SystemExit(f"unknown button(s) {unknown}; choose from {list(_MARKER_STYLE)}")

    traces, level = [], "?"
    for path in matches:
        data = np.load(path, allow_pickle=True)
        actions = data["actions"]
        level = str(data["level"]) if "level" in data.files else level
        traces.append((replay_positions(actions, bytes(data["initial_state"])), actions))
        print(f"  replayed {len(actions):>5} steps  {os.path.basename(path)}")

    if len(matches) == 1:
        stem = os.path.splitext(os.path.basename(matches[0]))[0]
    else:
        stem = f"{level.lower()}_all{len(matches)}"
    out_path = args.output or os.path.join("tmp", f"trace_map_{stem}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = sum(len(a) for _, a in traces)
    title = (f"{level}  ·  {len(matches)} trace(s)  ·  {total} steps"
             if len(matches) > 1 else
             f"{level}  ·  {total} steps  ·  {os.path.basename(matches[0])}")
    plot_traces(traces, buttons, out_path, title, show_path=not args.no_path)

    print(f"\n{len(matches)} trace(s), {total} steps total")
    for b in buttons:
        cnt = sum(int((a[:, BUTTON_BIT[b]] == 1).sum()) for _, a in traces)
        print(f"  {_MARKER_STYLE[b][2]:<5} presses: {cnt}")
    print(f"Saved map to: {out_path}")


if __name__ == "__main__":
    main()
