"""
gui_validate_aim_shot.py — browse detected AIM_SHOT maneuvers frame-by-frame.

For each HP-loss event the viewer shows:
  - the NES video frame
  - a panel with event details (enemy, weapon, HP delta, positions)
  - the timeline highlights: orange = t_aim (first fire in window),
    red = hit range (t_start … t_end)

Controls:
    Left / Right arrow  : step 1 frame  (hold for continuous scrub)
    N / P               : jump to Next / Previous maneuver
    Space               : play / pause
    Q / Escape          : quit

Usage:
    python -m maneuvers.gui_validate_aim_shot path/to/trace.npz
"""

import sys
import time

import numpy as np
import pygame

from contra.replay import replay_actions, SKIP
from maneuvers.detect_aim_shot import collect_hp_loss_events


SCALE        = 3
PANEL_HEIGHT = 160
FPS_PLAYBACK = 20   # logical fps (= 60 NES fps / SKIP=3)

# Colour palette
COL_BG       = (20,  20,  20)
COL_INFO     = (140, 140, 140)
COL_LABEL    = (100, 100, 100)
COL_AIM      = (255, 160,  40)   # orange  — fire / aim step
COL_HIT      = (220,  50,  50)   # red     — hit range
COL_NEUTRAL  = (50,   50,  50)   # timeline background
COL_CURSOR   = (255,  80,  80)
COL_DETAIL   = (80,  220, 160)   # event detail text


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trace(npz_path: str):
    """Replay the trace once for video, once for maneuver detection."""
    print("Replaying for video frames …")
    result = replay_actions(npz_path, want_video=True, verbose=False)
    frames = result["video"]   # (N+1, H, W, 3) uint8 — frame[0] is before first action

    print("Detecting AIM_SHOT maneuvers …")
    maneuvers = collect_hp_loss_events(npz_path)

    # Convert logical steps → video frame indices.
    # frame[0] = initial state, frame[step + 1] = after step.
    for m in maneuvers:
        m["_f_aim"]   = m["t_aim"] + 1 if m["t_aim"] is not None else None
        m["_f_start"] = m["t_start"] + 1
        m["_f_end"]   = m["t_end"]   + 1
        # Full span shown on timeline: fire frame → last hit frame
        m["_f_span_start"] = m["_f_aim"] if m["_f_aim"] is not None else m["_f_start"]
        m["_f_span_end"]   = m["_f_end"]

    print(f"  {len(frames)} video frames, {len(maneuvers)} maneuver(s).")
    return frames, maneuvers


# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    lines, line = [], []
    for word in words:
        if font.size(" ".join(line + [word]))[0] <= max_width:
            line.append(word)
        else:
            if line:
                lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))
    return lines


def maneuver_at_frame(maneuvers: list[dict], f: int) -> dict | None:
    """Return the first maneuver whose full span (fire → last hit) covers frame f."""
    for m in maneuvers:
        if m["_f_span_start"] <= f <= m["_f_span_end"]:
            return m
    return None


def maneuver_index_at_frame(maneuvers: list[dict], f: int) -> int | None:
    for i, m in enumerate(maneuvers):
        if m["_f_span_start"] <= f <= m["_f_span_end"]:
            return i
    return None


# ── Main viewer ───────────────────────────────────────────────────────────────

def main(npz_path: str):
    frames, maneuvers = load_trace(npz_path)

    total_frames = len(frames)
    h, w = frames[0].shape[:2]
    disp_w, disp_h = w * SCALE, h * SCALE

    pygame.init()
    screen = pygame.display.set_mode((disp_w, disp_h + PANEL_HEIGHT))
    pygame.display.set_caption(f"AIM_SHOT validator — {npz_path}")
    clock = pygame.time.Clock()

    font_info   = pygame.font.SysFont("monospace", 13)
    font_detail = pygame.font.SysFont("monospace", 14)

    current_frame   = 0
    paused          = True
    scrubbing       = False
    cached_idx      = -1
    cached_surf     = None
    key_held_since: dict[int, float] = {}
    HOLD_DELAY = 0.25

    # Pre-build pygame surfaces from numpy frames
    def get_surf(idx: int) -> pygame.Surface:
        nonlocal cached_idx, cached_surf
        if idx == cached_idx:
            return cached_surf
        rgb = frames[idx]
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (disp_w, disp_h))
        cached_idx  = idx
        cached_surf = surf
        return surf

    def bar_rect() -> pygame.Rect:
        return pygame.Rect(8, disp_h + PANEL_HEIGHT - 14, disp_w - 16, 8)

    def frame_from_mouse_x(mx: int) -> int:
        r = bar_rect()
        t = max(0.0, min(1.0, (mx - r.x) / r.width))
        return int(t * (total_frames - 1))

    def jump_to_maneuver(delta: int):
        nonlocal current_frame, paused
        idx = maneuver_index_at_frame(maneuvers, current_frame)
        if idx is None:
            nearest = min(range(len(maneuvers)),
                          key=lambda i: abs(maneuvers[i]["_f_span_start"] - current_frame),
                          default=None)
            idx = nearest if nearest is not None else 0
        else:
            idx = max(0, min(len(maneuvers) - 1, idx + delta))
        if maneuvers:
            current_frame = maneuvers[idx]["_f_aim"] or maneuvers[idx]["_f_start"]
            paused = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if bar_rect().collidepoint(event.pos):
                    scrubbing     = True
                    current_frame = frame_from_mouse_x(event.pos[0])
                    paused        = True

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                scrubbing = False

            elif event.type == pygame.MOUSEMOTION and scrubbing:
                if bar_rect().collidepoint(event.pos):
                    current_frame = frame_from_mouse_x(event.pos[0])

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RIGHT:
                    current_frame = min(current_frame + 1, total_frames - 1)
                    paused = True
                    key_held_since[k] = time.time()
                elif k == pygame.K_LEFT:
                    current_frame = max(current_frame - 1, 0)
                    paused = True
                    key_held_since[k] = time.time()
                elif k == pygame.K_n:
                    jump_to_maneuver(+1)
                elif k == pygame.K_p:
                    jump_to_maneuver(-1)

            elif event.type == pygame.KEYUP:
                key_held_since.pop(event.key, None)

        # Held arrow keys
        now = time.time()
        for key, step in ((pygame.K_RIGHT, +1), (pygame.K_LEFT, -1)):
            if key in key_held_since and now - key_held_since[key] > HOLD_DELAY:
                current_frame = max(0, min(current_frame + step, total_frames - 1))

        if not paused:
            current_frame = min(current_frame + 1, total_frames - 1)
            if current_frame == total_frames - 1:
                paused = True

        # ── Draw video ────────────────────────────────────────────────────────
        screen.blit(get_surf(current_frame), (0, 0))

        # ── Draw panel ────────────────────────────────────────────────────────
        pygame.draw.rect(screen, COL_BG, pygame.Rect(0, disp_h, disp_w, PANEL_HEIGHT))

        logical_step = max(0, current_frame - 1)   # frame 0 = before step 0
        info_str = (
            f"frame {current_frame}/{total_frames - 1}  "
            f"step {logical_step}  "
            f"{'PAUSED' if paused else 'PLAYING'}  "
            f"[N/P] next/prev maneuver"
        )
        screen.blit(font_info.render(info_str, True, COL_INFO), (8, disp_h + 6))

        # Current maneuver details
        ev = maneuver_at_frame(maneuvers, current_frame)
        y = disp_h + 28
        if ev:
            aim_str  = str(ev["t_aim"]) if ev["t_aim"] is not None else "?"
            nfire    = len(ev["fire_steps"])
            hit_str  = (f"{ev['t_start']}-{ev['t_end']}"
                        if ev["t_start"] != ev["t_end"] else str(ev["t_start"]))
            tag_col  = (100, 255, 120) if ev["destroyed"] else (255, 200, 80)

            lines = [
                (tag_col,    ev["tag"]),
                (COL_AIM,    f"aim step: {aim_str}  ({nfire} fire action(s) in window)"),
                (COL_HIT,    f"hit steps: {hit_str}  slot={ev['slot']}  -{ev['hp_delta']}HP"),
                (COL_DETAIL, f"enemy: {ev['enemy_name']} ({ev['enemy_type']})  "
                             f"pos=({ev['enemy_x']},{ev['enemy_y']})  "
                             f"player=({ev['player_x']},{ev['player_y']})"),
                (COL_DETAIL, f"weapon: {ev['weapon']}  level: {ev['level']}"),
            ]
            for col, text in lines:
                screen.blit(font_detail.render(text, True, col), (8, y))
                y += 20
        else:
            screen.blit(font_detail.render("— no maneuver at this frame —",
                                           True, (70, 70, 70)), (8, y))

        # ── Timeline ──────────────────────────────────────────────────────────
        r    = bar_rect()
        bw   = r.width
        pygame.draw.rect(screen, COL_NEUTRAL, r)

        for m in maneuvers:
            # full span: fire frame → last hit frame, single bar
            x1 = int(m["_f_span_start"] / total_frames * bw) + r.x
            x2 = max(int(m["_f_span_end"] / total_frames * bw) + r.x, x1 + 3)
            pygame.draw.rect(screen, COL_HIT, (x1, r.y, x2 - x1, r.height))

        # cursor
        cx = int(current_frame / total_frames * bw) + r.x
        pygame.draw.line(screen, COL_CURSOR, (cx, r.y - 2), (cx, r.y + r.height + 2), 2)

        pygame.display.flip()
        clock.tick(FPS_PLAYBACK)

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m maneuvers.gui_validate_aim_shot path/to/trace.npz")
        sys.exit(1)
    main(sys.argv[1])
