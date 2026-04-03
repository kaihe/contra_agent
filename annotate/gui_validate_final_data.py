"""
Validate Annotations
====================
Browse a final_data sample frame-by-frame with text annotations and input
actions shown below the video.

Usage:
    python validate_annotations.py <path/to/final_data/game/uuid/>
    python validate_annotations.py final_data/contra_nes/019cf137-.../

Controls:
    Left / Right arrow  : step 10 frames  (hold for continuous fast scrub)
    Space               : play / pause
    Q / Escape          : quit
"""

import os
import sys
import time

import cv2
import numpy as np
import pygame

from annotate.proto.video_annotation_pb2 import VideoAnnotation  # type: ignore


PANEL_HEIGHT = 200    # pixels below the video for annotation + action display
FPS_PLAYBACK = 30
SCALE        = 4      # upscale 192x192 frames


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample(sample_dir: str):
    """Load 192x192.mp4 and annotation.proto from a final_data sample dir."""
    video_path = os.path.join(sample_dir, "192x192.mp4")
    proto_path = os.path.join(sample_dir, "annotation.proto")

    if not os.path.isfile(video_path):
        sys.exit(f"ERROR: {video_path} not found")
    if not os.path.isfile(proto_path):
        sys.exit(f"ERROR: {proto_path} not found")

    va = VideoAnnotation()
    with open(proto_path, "rb") as f:
        va.ParseFromString(f.read())

    fps = va.metadata.frames_per_second or 20.0

    # Build frame → instruction mapping from proto
    # Annotations sit on start frames; duration gives the covered span.
    frame_annotations = list(va.frame_annotations)
    ann_ranges = []   # list of (start_frame, end_frame, instruction)
    for i, fa in enumerate(frame_annotations):
        for ta in fa.frame_text_annotation:
            end = i + round(ta.duration * fps) if ta.duration else i
            ann_ranges.append((i, end, ta.instruction))

    narrative = va.video_global_task.video_narrative

    return video_path, fps, narrative, ann_ranges, frame_annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def annotation_for_frame(ann_ranges: list, frame: int) -> str | None:
    for start, end, instr in ann_ranges:
        if start <= frame <= end:
            return instr
    return None


def action_label(fa) -> str:
    """Return a compact string describing the input action for a frame."""
    if fa is None:
        return ""
    ua = fa.user_action

    # GamePad
    gp = ua.game_pad
    b  = gp.buttons
    parts = []
    if b.south:      parts.append("B")
    if b.east:       parts.append("A")
    if b.north:      parts.append("X")
    if b.west:       parts.append("Y")
    if b.select:     parts.append("SELECT")
    if b.start:      parts.append("START")
    if b.dpad_up:    parts.append("UP")
    if b.dpad_down:  parts.append("DOWN")
    if b.dpad_left:  parts.append("LEFT")
    if b.dpad_right: parts.append("RIGHT")
    if b.left_bumper:  parts.append("LB")
    if b.right_bumper: parts.append("RB")
    ls = gp.left_stick
    if abs(ls.x) > 0.1 or abs(ls.y) > 0.1:
        parts.append(f"LS({ls.x:+.1f},{ls.y:+.1f})")

    # Keyboard
    keys = list(ua.keyboard.keys)
    if keys:
        parts += keys

    # Mouse buttons
    mouse_btns = list(ua.mouse.buttons_down)
    if mouse_btns:
        parts += [f"M:{btn}" for btn in mouse_btns]

    return "  ".join(parts) if parts else "—"


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    lines, line = [], []
    for word in words:
        test = " ".join(line + [word])
        if font.size(test)[0] <= max_width:
            line.append(word)
        else:
            if line:
                lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))
    return lines


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def main(sample_dir: str):
    video_path, fps, narrative, ann_ranges, frame_annotations = load_sample(sample_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        sys.exit(f"ERROR: could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    h, w = first.shape[:2]
    disp_w, disp_h = w * SCALE, h * SCALE

    pygame.init()
    screen = pygame.display.set_mode((disp_w, disp_h + PANEL_HEIGHT))
    sample_name = os.path.basename(sample_dir.rstrip("/\\"))
    pygame.display.set_caption(f"Validator — {sample_name}")
    clock = pygame.time.Clock()

    font_ann    = pygame.font.SysFont("monospace", 15)
    font_action = pygame.font.SysFont("monospace", 14)
    font_info   = pygame.font.SysFont("monospace", 13)

    current_frame = 0
    paused        = True
    scrubbing     = False
    cached_frame_idx = -1
    cached_surf      = None
    key_held_since: dict[int, float] = {}   # key -> time of initial press
    HOLD_DELAY = 0.3                         # seconds before repeat kicks in

    def bar_rect():
        bar_y = disp_h + PANEL_HEIGHT - 14
        return pygame.Rect(8, bar_y, disp_w - 16, 8)

    def frame_from_mouse_x(mx: int) -> int:
        r = bar_rect()
        t = max(0.0, min(1.0, (mx - r.x) / r.width))
        return int(t * (total_frames - 1))

    def get_surf(idx: int) -> pygame.Surface:
        nonlocal cached_frame_idx, cached_surf
        if idx == cached_frame_idx:
            return cached_surf
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            bgr = np.zeros((h, w, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (disp_w, disp_h))
        cached_frame_idx = idx
        cached_surf = surf
        return surf

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if bar_rect().collidepoint(event.pos):
                    scrubbing = True
                    current_frame = frame_from_mouse_x(event.pos[0])
                    paused = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                scrubbing = False
            elif event.type == pygame.MOUSEMOTION and scrubbing:
                if bar_rect().collidepoint(event.pos):
                    current_frame = frame_from_mouse_x(event.pos[0])
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(current_frame + 10, total_frames - 1)
                    paused = True
                    key_held_since[pygame.K_RIGHT] = time.time()
                elif event.key == pygame.K_LEFT:
                    current_frame = max(current_frame - 10, 0)
                    paused = True
                    key_held_since[pygame.K_LEFT] = time.time()
            elif event.type == pygame.KEYUP:
                key_held_since.pop(event.key, None)

        # Held arrow keys: continuous scrub after HOLD_DELAY
        now = time.time()
        for key, step in ((pygame.K_RIGHT, 10), (pygame.K_LEFT, -10)):
            if key in key_held_since and now - key_held_since[key] > HOLD_DELAY:
                current_frame = max(0, min(current_frame + step, total_frames - 1))

        if not paused:
            current_frame = min(current_frame + 1, total_frames - 1)
            if current_frame == total_frames - 1:
                paused = True

        # --- Draw video ---
        screen.blit(get_surf(current_frame), (0, 0))

        # --- Draw panel ---
        pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(0, disp_h, disp_w, PANEL_HEIGHT))

        seconds = current_frame / fps
        mm, ss  = divmod(int(seconds), 60)
        info = (
            f"frame {current_frame}/{total_frames - 1}  "
            f"({mm:02d}:{ss:02d})  "
            f"{'PAUSED' if paused else 'PLAYING'}"
        )
        screen.blit(font_info.render(info, True, (140, 140, 140)), (8, disp_h + 6))

        # Text annotation (yellow)
        ann = annotation_for_frame(ann_ranges, current_frame)
        if ann:
            lines = wrap_text(ann, font_ann, disp_w - 16)
            for i, line in enumerate(lines):
                screen.blit(font_ann.render(line, True, (255, 220, 80)), (8, disp_h + 26 + i * 20))
        else:
            screen.blit(font_ann.render("— no annotation —", True, (80, 80, 80)), (8, disp_h + 26))

        # Input action (cyan), two rows below annotation area
        fa = frame_annotations[current_frame] if current_frame < len(frame_annotations) else None
        action_str = action_label(fa)
        action_y   = disp_h + PANEL_HEIGHT - 38
        screen.blit(font_info.render("action:", True, (100, 100, 100)), (8, action_y))
        screen.blit(font_action.render(action_str, True, (80, 220, 220)), (72, action_y))

        # Timeline bar
        bar_y = disp_h + PANEL_HEIGHT - 14
        bar_w = disp_w - 16
        pygame.draw.rect(screen, (50, 50, 50), (8, bar_y, bar_w, 8))
        for start, end, _ in ann_ranges:
            x1 = int(start / total_frames * bar_w) + 8
            x2 = max(int(end   / total_frames * bar_w) + 8, x1 + 2)
            pygame.draw.rect(screen, (80, 160, 255), (x1, bar_y, x2 - x1, 8))
        cursor_x = int(current_frame / total_frames * bar_w) + 8
        pygame.draw.line(screen, (255, 80, 80), (cursor_x, bar_y - 2), (cursor_x, bar_y + 10), 2)

        pygame.display.flip()
        clock.tick(FPS_PLAYBACK)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_annotations.py <final_data/game/uuid/>")
        sys.exit(1)
    main(sys.argv[1])