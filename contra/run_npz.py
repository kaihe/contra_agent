"""
run_npz.py — replay any Contra NPZ recording through the emulator.

Public API
----------
    result = replay_npz(
        fpath,
        want_video = None,   # path to save .mp4, or None
        verbose    = True,   # print header / events to stdout
    )

Returns
-------
    "result"   : "WIN" | "LOSE"
    "steps"    : int
    "score"    : int
    "events"   : list[dict]   narrative events (GUN_PICKUP, LIFE_LOST, WIN …)
    "rewards"  : list[float]  per-step weighted reward (level-aware)
    "ram"      : np.ndarray   (steps+1, 2048) uint8
    "video"    : str | None   path to saved MP4, or None

Supported NPZ formats
---------------------
  bc_data   — keys: actions (N,9), initial_state, level, outcome, fps
  mc_trace  — keys: actions (N,2), ram, xscroll, score
"""

import os
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from contra.events import scan_events, compute_reward, EV_LEVELUP, EV_GAME_CLEAR, get_level
import stable_retro as retro

_levelup_ev = EV_LEVELUP

GAME = "Contra-Nes"
SKIP = 3
FPS  = 20   # logical fps for video output (= 60 NES fps / SKIP)

def rewind_state(env, emu_state: bytes) -> None:
    env.em.set_state(emu_state)
    env.data.update_ram()

def step_env(env, action, skip=SKIP):
    act = np.asarray(action, dtype=np.uint8)
    for _ in range(skip):
        env.step(act.copy())

def replay_npz(fpath: str, *,
               want_video: str  = None,
               verbose:    bool = True) -> dict:
    

    ckpt = np.load(fpath, allow_pickle=True)
    actions = ckpt["actions"]
    initial_emu_state = bytes(ckpt["initial_state"])
    level_str = str(ckpt["level"]) if "level" in ckpt else "Level1"
    print(f"  Load from {fpath} (replaying {len(actions)} actions)")

    env = retro.make(
        game=GAME,
        state=level_str,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    obs, _ = env.reset()
    rewind_state(env, initial_emu_state)
    start_level = get_level(env.unwrapped.get_ram())

    ffmpeg_proc = None
    if want_video:
        import subprocess
        import cv2
        os.makedirs(os.path.dirname(want_video) or ".", exist_ok=True)
        h, w = obs.shape[:2]
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                "-r", str(FPS), "-i", "-",
                "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
                want_video,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _record(frame):
        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(frame.tobytes())

    _record(obs)
    pre_ram = env.unwrapped.get_ram().copy()
    ram_list     = [pre_ram.copy()]
    all_events:  list[dict]  = []
    rewards:     list[float] = []
    leveled_up  = False
    game_cleared = False
    info: dict = {}

    replay_acts = actions

    for step, act in enumerate(replay_acts):
        pre_ram = env.unwrapped.get_ram().copy()
        act_arr = np.asarray(act, dtype=np.uint8)
        for _ in range(SKIP):
            obs, _, _, _, info = env.step(act_arr.copy())
        curr_ram = env.unwrapped.get_ram()

        _record(obs)
        ram_list.append(curr_ram)
        rewards.append(compute_reward(pre_ram, curr_ram))
        all_events.extend(scan_events(pre_ram, curr_ram, step))
        if _levelup_ev.trigger(pre_ram, curr_ram):
            leveled_up = True
        if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
            game_cleared = True

    env.close()

    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if verbose:
            print(f"  MP4 saved → {want_video}")

    outcome     = "game_clear" if game_cleared else "level_up" if leveled_up else "lose"
    final_score = int(info.get("score", 0))
    total_steps = step + 1
    final_level = get_level(curr_ram)

    if verbose:
        for e in all_events:
            print(f"  step {e['step']:4d}  {e['tag']:<12}  {e['detail']}")

    return {"result": outcome, "steps": total_steps, "score": final_score,
            "start_level": start_level, "final_level": final_level,
            "events": all_events, "rewards": rewards,
            "ram": np.array(ram_list, dtype=np.uint8),
            "video": want_video}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Replay a Contra NPZ recording")
    parser.add_argument("npz",     help="Path to .npz file")
    parser.add_argument("--video", default=None, help="Save MP4 to this path")
    args = parser.parse_args()

    result = replay_npz(args.npz, want_video=args.video, verbose=True)
    print(f"\nlevel {result['start_level']} → {result['final_level']}"
          f"  score={result['score']}  steps={result['steps']}"
          f"  total_reward={sum(result['rewards']):.1f}"
          f"  ram={result['ram'].shape}")


if __name__ == "__main__":
    main()
