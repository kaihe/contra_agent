"""
replay.py — replay any Contra action sequence through the emulator.

Public API
----------
    # From a NPZ file:
    result = replay_actions(fpath, want_video=False, verbose=True)

    # From a raw action array:
    result = replay_actions(
        actions,               # np.ndarray (N, 9)
        initial_state = ...,   # bytes  — required
        level         = "Level1",
        want_video    = False,
        verbose       = True,
    )

    # Persist the returned frames to disk:
    save_video(result["video"], path)

Returns
-------
    "result" : "game_clear" | "level_up" | "lose"
    "video"  : np.ndarray (N+1, H, W, 3) uint8, or None if want_video=False

Supported NPZ formats
---------------------
  bc_data   — keys: actions (N,9), initial_state, level, outcome, fps
  mc_trace  — keys: actions (N,2), ram, xscroll, score
"""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.events import scan_events, EV_LEVELUP, EV_GAME_CLEAR, get_level
import stable_retro as retro

_levelup_ev = EV_LEVELUP

GAME = "Contra-Nes"
SKIP = 3
FPS  = 20   # logical fps (= 60 NES fps / SKIP)


def rewind_state(env, emu_state: bytes) -> None:
    env.em.set_state(emu_state)
    env.data.update_ram()


def step_env(env, act: np.ndarray) -> None:
    """Step the environment SKIP times with the given action."""
    act = np.asarray(act, dtype=np.uint8)
    for _ in range(SKIP):
        env.step(act.copy())


def save_video(frames: np.ndarray, path: str) -> None:
    """Save a (N, H, W, 3) uint8 frame array to an MP4 file via ffmpeg."""
    import subprocess
    _, h, w, _ = frames.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(FPS), "-i", "-",
            "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
            path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def replay_actions(source, *,
                   initial_state: bytes = None,
                   level:         str   = "Level1",
                   want_video:    bool  = False,
                   verbose:       bool  = True) -> dict:
    """Replay a Contra action sequence.

    Parameters
    ----------
    source : str | np.ndarray
        Path to a .npz file, or an (N, 9) action array.
    initial_state : bytes
        Required when source is an array. Emulator save-state to rewind to.
    level : str
        Level name (e.g. "Level1"). Used when source is an array.
    want_video : bool
        If True, collect frames and return as "video" (N+1, H, W, 3) uint8.
    """
    if isinstance(source, (str, os.PathLike)):
        ckpt = np.load(source, allow_pickle=True)
        actions = ckpt["actions"]
        initial_emu_state = bytes(ckpt["initial_state"])
        level_str = str(ckpt["level"]) if "level" in ckpt else level
        if verbose:
            print(f"  Load from {source} (replaying {len(actions)} actions)")
    else:
        actions = np.asarray(source)
        if initial_state is None:
            raise ValueError("initial_state is required when source is an array")
        initial_emu_state = bytes(initial_state)
        level_str = level
        if verbose:
            print(f"  Replaying {len(actions)} actions from array")

    env = retro.make(
        game=GAME,
        state=level_str,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    obs, _ = env.reset()
    rewind_state(env, initial_emu_state)

    if want_video:
        # Inject the first image with no action (nes_0)
        frames = [env.em.get_screen().copy()]
    else:
        frames = None

    all_events:  list[dict] = []
    leveled_up   = False
    game_cleared = False

    for step, act in enumerate(actions):
        pre_ram = env.unwrapped.get_ram().copy()
        act_arr = np.asarray(act, dtype=np.uint8)
        for i in range(SKIP):
            obs, _, _, _, _ = env.step(act_arr.copy())
            if i == 0 and frames is not None:
                frames.append(obs.copy())  # first NES frame: model sees state right after action
        curr_ram = env.unwrapped.get_ram()
        all_events.extend(scan_events(pre_ram, curr_ram, step))
        if _levelup_ev.trigger(pre_ram, curr_ram):
            leveled_up = True
        if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
            game_cleared = True

    env.close()

    outcome = "game_clear" if game_cleared else "level_up" if leveled_up else "lose"

    if verbose:
        for e in all_events:
            print(f"  step {e['step']:4d}  {e['tag']:<12}  {e['detail']}")

    return {
        "result": outcome,
        "video":  np.stack(frames) if frames is not None else None,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Replay a Contra NPZ recording")
    parser.add_argument("npz",     help="Path to .npz file")
    parser.add_argument("--video", default=None, help="Save MP4 to this path")
    args = parser.parse_args()

    result = replay_actions(args.npz, want_video=bool(args.video), verbose=True)
    print(f"\n{result['result']}")
    if args.video:
        save_video(result["video"], args.video)
        print(f"  MP4 saved → {args.video}")


if __name__ == "__main__":
    main()
