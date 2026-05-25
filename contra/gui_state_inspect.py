"""
NES Contra state inspector — replay a trace .npz and browse frame-by-frame
with the decoded game state shown alongside each image.

Usage:
    python contra/gui_state_inspect.py <path/to/trace.npz>

Controls:
    Left / Right arrow  : step ±1 frame  (hold for continuous scrub)
    Space               : play / pause
    Q / Escape          : quit

The trace is fully replayed at startup; the emulator is closed before the GUI
opens (stable-retro allows only one emulator instance per process).
"""

import os
import sys
import time
import warnings

import numpy as np
import pygame

warnings.filterwarnings("ignore", message=".*Gym.*")

from contra.game_state import describe_ram, decode_ram
from contra.replay import GAME, SKIP, rewind_state
import stable_retro as retro


SCALE         = 3    # upscale 240×224 NES frame → 720×672
PANEL_HEIGHT  = 80   # bottom strip: frame counter + timeline
STATE_PANEL_W = 340  # right panel: decoded game state

_C_SECTION = (255, 215, 60)   # amber  — "Player:", "Enemies:", "Scene:"
_C_TBLHDR  = (100, 100, 100)  # dim    — column headers / separators
_C_DATA    = (200, 200, 200)  # light  — data lines

_C_PLAYER  = (60,  255,  80)  # green — player overlay
_C_ENEMY   = (255,  60,  60)  # red   — enemy overlay


# ── Replay ─────────────────────────────────────────────────────────────────────

def _replay(npz_path: str) -> tuple[list, list, str, str, int]:
    """Replay a trace npz, collecting one frame and one RAM snapshot per step.

    Returns
    -------
    frames  : list of (H, W, 3) uint8 — one entry per logical step + initial
    rams    : list of (2048,) uint8
    level   : str
    outcome : str
    fps     : int  logical fps from the npz
    """
    ckpt    = np.load(npz_path, allow_pickle=True)
    actions = ckpt["actions"]
    initial = bytes(ckpt["initial_state"])
    level   = str(ckpt.get("level",   "?"))
    outcome = str(ckpt.get("outcome", "?"))
    fps     = int(ckpt.get("fps", 20))
    n       = len(actions)

    print(f"Replaying {n} steps  level={level}  outcome={outcome}")

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial)

    frames = [env.em.get_screen().copy()]
    rams   = [env.unwrapped.get_ram().copy()]

    for i, act in enumerate(actions):
        if i % 200 == 0:
            print(f"  {i}/{n}\r", end="", flush=True)
        act_arr = np.asarray(act, dtype=np.uint8)
        for sub in range(SKIP):
            env.step(act_arr.copy())
            if sub == 0:
                frames.append(env.em.get_screen().copy())
                rams.append(env.unwrapped.get_ram().copy())

    env.close()
    print(f"  {n}/{n}  done")
    return frames, rams, level, outcome, fps


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def _draw_state(screen, font, ram: np.ndarray, x: int, max_y: int) -> None:
    y = 8
    for line in describe_ram(ram).split("\n"):
        if y + 13 > max_y:
            break
        stripped = line.strip()
        if not stripped:
            y += 6
            continue
        if not line.startswith(" "):
            color = _C_SECTION
        elif "----" in stripped or (stripped.startswith("Slot") and "Type" in stripped):
            color = _C_TBLHDR
        else:
            color = _C_DATA
        screen.blit(font.render(line, True, color), (x + 6, y))
        y += 13


def _draw_overlays(screen, font_label, ram: np.ndarray, scale: int) -> None:
    """Draw player (green) and enemy (red) position markers on the video surface."""
    state = decode_ram(ram)

    # Player — crosshair inside a circle
    px = state["player_x"] * scale
    py = state["player_y"] * scale
    r  = 8
    pygame.draw.circle(screen, _C_PLAYER, (px, py), r, 2)
    pygame.draw.line(screen, _C_PLAYER, (px - r - 3, py), (px + r + 3, py), 1)
    pygame.draw.line(screen, _C_PLAYER, (px, py - r - 3), (px, py + r + 3), 1)

    # Enemies — circle with slot number
    for e in state["enemies"]:
        ex = e["x"] * scale
        ey = e["y"] * scale
        pygame.draw.circle(screen, _C_ENEMY, (ex, ey), 6, 2)
        lbl = font_label.render(str(e["slot"]), True, (255, 180, 180))
        screen.blit(lbl, (ex + 7, ey - 5))


# ── Main GUI ───────────────────────────────────────────────────────────────────

def main(npz_path: str) -> None:
    frames, rams, level, outcome, fps = _replay(npz_path)

    total        = len(frames)
    h, w         = frames[0].shape[:2]
    disp_w       = w * SCALE
    disp_h       = h * SCALE
    total_w      = disp_w + STATE_PANEL_W

    pygame.init()
    screen = pygame.display.set_mode((total_w, disp_h + PANEL_HEIGHT))
    pygame.display.set_caption(
        f"State Inspector — {os.path.basename(npz_path)}  [{level}  {outcome}]"
    )
    clock = pygame.time.Clock()

    font_info  = pygame.font.SysFont("monospace", 13)
    font_state = pygame.font.SysFont("monospace", 11)
    font_label = pygame.font.SysFont("monospace",  9)

    cached_idx  = -1
    cached_surf = None

    def get_surf(idx: int) -> pygame.Surface:
        nonlocal cached_idx, cached_surf
        if idx == cached_idx:
            return cached_surf
        surf = pygame.surfarray.make_surface(frames[idx].swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (disp_w, disp_h))
        cached_idx  = idx
        cached_surf = surf
        return surf

    def bar_rect() -> pygame.Rect:
        return pygame.Rect(8, disp_h + PANEL_HEIGHT - 14, disp_w - 16, 8)

    def frame_from_x(mx: int) -> int:
        r = bar_rect()
        t = max(0.0, min(1.0, (mx - r.x) / r.width))
        return int(t * (total - 1))

    current   = 0
    paused    = True
    scrubbing = False
    held: dict[int, float] = {}
    HOLD_DELAY = 0.25

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if bar_rect().collidepoint(ev.pos):
                    scrubbing = True
                    current   = frame_from_x(ev.pos[0])
                    paused    = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                scrubbing = False
            elif ev.type == pygame.MOUSEMOTION and scrubbing:
                current = frame_from_x(ev.pos[0])
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    current = min(current + 1, total - 1)
                    paused  = True
                    held[pygame.K_RIGHT] = time.time()
                elif ev.key == pygame.K_LEFT:
                    current = max(current - 1, 0)
                    paused  = True
                    held[pygame.K_LEFT] = time.time()
            elif ev.type == pygame.KEYUP:
                held.pop(ev.key, None)

        now = time.time()
        for key, delta in ((pygame.K_RIGHT, 1), (pygame.K_LEFT, -1)):
            if key in held and now - held[key] > HOLD_DELAY:
                current = max(0, min(current + delta, total - 1))

        if not paused:
            current = min(current + 1, total - 1)
            if current == total - 1:
                paused = True

        # video
        screen.blit(get_surf(current), (0, 0))
        _draw_overlays(screen, font_label, rams[current], SCALE)

        # state panel
        pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(disp_w, 0, STATE_PANEL_W, disp_h))
        pygame.draw.line(screen, (50, 50, 50), (disp_w, 0), (disp_w, disp_h))
        _draw_state(screen, font_state, rams[current], disp_w, disp_h)

        # bottom panel
        pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(0, disp_h, total_w, PANEL_HEIGHT))
        seconds = current / fps
        mm, ss  = divmod(int(seconds), 60)
        info    = (
            f"frame {current}/{total - 1}  "
            f"({mm:02d}:{ss:02d})  "
            f"{'PAUSED' if paused else 'PLAYING'}"
        )
        screen.blit(font_info.render(info, True, (140, 140, 140)), (8, disp_h + 10))

        # timeline bar
        r  = bar_rect()
        pygame.draw.rect(screen, (50, 50, 50), r)
        cx = (int(current / (total - 1) * r.width) + r.x) if total > 1 else r.x
        pygame.draw.line(screen, (255, 80, 80), (cx, r.y - 2), (cx, r.y + 10), 2)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python contra/gui_state_inspect.py <trace.npz>")
        sys.exit(1)
    main(sys.argv[1])
