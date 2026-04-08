"""
gui_inspect_npz.py — Step through any Contra NPZ recording frame by frame.

Usage:
    python gui_inspect_npz.py <path/to/trace.npz> [--start STEP]

Controls:
    Left / Right arrow  : step 1 frame
    Shift + Left/Right  : step 10 frames
    Space               : play / pause
    Q / Escape          : quit
"""

import argparse
import os
import numpy as np
import pygame

import stable_retro as retro
from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from contra.events import compute_reward, scan_events

GAME      = "Contra-Nes"
SKIP      = 3
SCALE     = 3
NES_W, NES_H = 240, 224
INFO_H    = 56
FPS       = 20

_DPAD_NP   = np.array(DPAD_TABLE,   dtype=np.uint8)
_BUTTON_NP = np.array(BUTTON_TABLE, dtype=np.uint8)


def _nes_action(row, fmt):
    if fmt == "pair":
        return (_DPAD_NP[int(row[0])] | _BUTTON_NP[int(row[1])]).tolist()
    return row.tolist()


def load_frames(fpath: str) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[list], int]:
    """Replay the NPZ and return (frames, actions, rewards, events_per_step, start_level_idx)."""
    d       = np.load(fpath, allow_pickle=True)
    actions = d["actions"]
    fmt     = "pair" if actions.shape[1] == 2 else "nes"

    # infer level
    start_level = 0
    if "level" in d:
        s = str(d["level"])
        digits = "".join(filter(str.isdigit, s))
        start_level = int(digits) - 1 if digits else 0
    else:
        import re
        m = re.search(r'[Ll]evel(\d+)', os.path.basename(fpath))
        start_level = int(m.group(1)) - 1 if m else 0

    env = retro.make(
        game=GAME, state=f"Level{start_level + 1}",
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if "initial_state" in list(d.keys()):
        env.initial_state = bytes(d["initial_state"])
    obs, _ = env.reset()

    replay_acts = actions

    frames      = [obs.copy()]
    act_records = [np.zeros(9, dtype=np.uint8)]  # placeholder for frame 0
    rewards     = [0.0]                           # placeholder for frame 0
    events      = [[]]                            # placeholder for frame 0

    prev_ram = env.unwrapped.get_ram().copy()
    print(f"Pre-loading {len(replay_acts)} frames (this may take a moment)...")
    for i, row in enumerate(replay_acts):
        if i % 500 == 0:
            print(f"  {i}/{len(replay_acts)}")
        act = _nes_action(row, fmt)
        for _ in range(SKIP):
            obs, _, term, trunc, _ = env.step(act)
        curr_ram = env.unwrapped.get_ram().copy()
        frames.append(obs.copy())
        act_records.append(np.array(act, dtype=np.uint8))
        rewards.append(compute_reward(prev_ram, curr_ram))
        step_events = scan_events(prev_ram, curr_ram, i + 1)
        events.append([f"{e['tag']}({e['detail']})" if e['detail'] else e['tag']
                       for e in step_events])
        prev_ram = curr_ram

    env.close()
    print(f"Loaded {len(frames)} frames. Showing GUI — use arrows to scrub, Space to play.")
    return frames, act_records, rewards, events, start_level


def render_info(surface, step: int, total: int, act: np.ndarray,
                reward: float, cumul_reward: float, step_events: list, font):
    surface.fill((20, 20, 20))
    label_map = ["B", "-", "SEL", "STA", "UP", "DN", "LT", "RT", "A"]
    btns = [label_map[i] for i in range(9) if act[i]]
    btn_str = "+".join(btns) if btns else "NOOP"
    r_color = (100, 255, 100) if reward > 0 else (255, 80, 80) if reward < 0 else (160, 160, 160)
    line1 = f"Step {step:4d}/{total - 1}  r={reward:+.1f}  Σ={cumul_reward:+.1f}  {btn_str}"
    surface.blit(font.render(line1, True, r_color), (8, 4))
    if step_events:
        ev_str = "  ".join(step_events)
        surface.blit(font.render(ev_str, True, (255, 220, 60)), (8, 24))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to .npz trace file")
    parser.add_argument("--start", type=int, default=0, help="Starting step")
    args = parser.parse_args()

    frames, acts, rewards, events, level_idx = load_frames(args.npz)
    total = len(frames)
    cumul_rewards = np.cumsum(rewards).tolist()

    pygame.init()
    win_w = NES_W * SCALE
    win_h = NES_H * SCALE + INFO_H
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"NPZ Inspector — Level{level_idx + 1} — {os.path.basename(args.npz)}")
    font  = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    step    = max(0, min(args.start, total - 1))
    playing = False

    info_surf = pygame.Surface((win_w, INFO_H))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    mods = pygame.key.get_mods()
                    if (mods & pygame.KMOD_SHIFT) and (mods & pygame.KMOD_CTRL):
                        delta = 100
                    elif mods & pygame.KMOD_SHIFT:
                        delta = 10
                    else:
                        delta = 1
                    step  = min(step + delta, total - 1)
                    playing = False
                elif event.key == pygame.K_LEFT:
                    mods = pygame.key.get_mods()
                    if (mods & pygame.KMOD_SHIFT) and (mods & pygame.KMOD_CTRL):
                        delta = 100
                    elif mods & pygame.KMOD_SHIFT:
                        delta = 10
                    else:
                        delta = 1
                    step  = max(step - delta, 0)
                    playing = False

        if playing:
            step = (step + 1) % total

        # draw frame
        frame = frames[step]
        surf  = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (win_w, NES_H * SCALE)), (0, 0))

        render_info(info_surf, step, total, acts[step],
                    rewards[step], cumul_rewards[step], events[step], font)
        screen.blit(info_surf, (0, NES_H * SCALE))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
