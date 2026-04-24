"""
gui_inspect_dpo.py — Browse DPO pairs from a search graph side-by-side.

Replays each pair's good and bad action traces from the branch emu state
and displays the resulting frames side-by-side for visual comparison.

Controls:
    Left / Right arrow  : step one frame (hold for continuous scrub)
    Up / Down arrow     : previous / next DPO pair
    Space               : play / pause
    Q / Escape          : quit

Usage:
    python annotate/gui_inspect_dpo.py synthetic/mc_graph/graph_level1_<date>.npz
"""

import os
import sys
import time

import numpy as np
import pygame
import stable_retro as retro

from annotate.gen_dpo_data import collect_dpo_pairs, _infer_level_goal, _good_trace
from contra.replay import rewind_state
from synthetic.mc_search import SKIP
from synthetic.mc_search_dpo import load_graph

_GAME           = "Contra-Nes"
_STATE_BY_LEVEL = {i: f"Level{i}" for i in range(1, 9)}

SCALE        = 2
PANEL_HEIGHT = 90
FPS_PLAYBACK = 20
HOLD_DELAY   = 0.25


# ── emulator ──────────────────────────────────────────────────────────────────

def _make_env(level: int) -> retro.RetroEnv:
    state_label = _STATE_BY_LEVEL[level]
    use_spread  = level > 1
    env = retro.make(
        game=_GAME,
        state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        env.load_state(f"spread_gun_state/{state_label}", retro.data.Integrations.CUSTOM_ONLY)
    env.reset()
    return env


TAIL_FRAMES = 10  # extra NES frames to render after dead trace ends (shows death animation)


def _render_trace(env, init_emu: bytes, prefix: np.ndarray, actions: np.ndarray, tail: int = 0) -> list[np.ndarray]:
    """Rewind to init_emu, fast-forward through prefix, then capture one frame per action."""
    rewind_state(env, init_emu)
    for act in prefix:
        for _ in range(SKIP):
            env.step(act)
    frames = []
    for act in actions:
        obs = None
        for _ in range(SKIP):
            result = env.step(act)
            obs = result[0]
        if obs is not None:
            frames.append(obs.copy())
    noop = np.zeros(9, dtype=np.uint8)
    for _ in range(tail):
        obs = env.step(noop)[0]
        if obs is not None:
            frames.append(obs.copy())
    return frames


def _load_pair_frames(env, pair: dict, good_actions: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    init_emu = pair["start_emu"]
    prefix   = good_actions[0 : pair["prefix_len"]]
    tail     = TAIL_FRAMES if pair.get("kind") == "dead" else 0
    good = _render_trace(env, init_emu, prefix, pair["good_trace"], tail=0)
    bad  = _render_trace(env, init_emu, prefix, pair["bad_trace"],  tail=tail)
    return good, bad


# ── pygame helpers ────────────────────────────────────────────────────────────

def _to_surf(arr: np.ndarray, w: int, h: int) -> pygame.Surface:
    surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    return pygame.transform.scale(surf, (w, h))


# ── main ──────────────────────────────────────────────────────────────────────

def main(graph_path: str) -> None:
    level, goal    = _infer_level_goal(graph_path)
    root, init_emu = load_graph(graph_path)
    good_nodes     = _good_trace(root)
    good_actions   = np.array(
        [n.action for n in good_nodes[1:]], dtype=np.uint8
    ) if len(good_nodes) > 1 else np.empty((0, 9), dtype=np.uint8)
    good_total = len(good_nodes) - 1  # exclude root
    env        = _make_env(level)
    pairs      = collect_dpo_pairs(root, init_emu, env)

    if not pairs:
        print("No DPO pairs found in graph.")
        sys.exit(0)

    print(f"Loaded {len(pairs)} DPO pairs  (level={level}, goal={goal}, good_trace={good_total})")

    cache: dict[int, tuple[list, list]] = {}

    # ── window setup (sized from first pair) ─────────────────────────────────
    pygame.init()

    def _fetch(idx: int, screen=None, font=None) -> tuple[list, list]:
        if idx not in cache:
            if screen is not None:
                screen.fill((10, 10, 10))
                msg = font.render(f"Loading pair {idx + 1}/{len(pairs)} …", True, (200, 200, 200))
                screen.blit(msg, (20, 40))
                pygame.display.flip()
            cache[idx] = _load_pair_frames(env, pairs[idx], good_actions)
        return cache[idx]

    font_info  = pygame.font.SysFont("monospace", 13)
    font_label = pygame.font.SysFont("monospace", 15, bold=True)

    # Load first pair to get frame dimensions
    good_frames, bad_frames = _fetch(0)
    fh, fw = good_frames[0].shape[:2] if good_frames else (224, 256)
    cell_w = fw * SCALE
    cell_h = fh * SCALE
    win_w  = cell_w * 2
    win_h  = cell_h + PANEL_HEIGHT

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"DPO Inspector — {os.path.basename(graph_path)}")
    clock  = pygame.time.Clock()

    black_surf = pygame.Surface((cell_w, cell_h))
    black_surf.fill((15, 15, 15))

    pair_idx  = 0
    frame_idx = 0
    paused    = True
    key_held: dict[int, float] = {}

    # ── bar geometry ─────────────────────────────────────────────────────────
    M      = 8                          # margin
    bar_h  = 8
    # local bar (current pair frame position)
    local_y = win_h - M - bar_h
    local_x = M
    local_w = win_w - M * 2

    def n_frames() -> int:
        return max(len(good_frames), len(bad_frames), 1)

    def get_surf(frames: list, idx: int) -> pygame.Surface:
        if not frames or idx >= len(frames):
            return black_surf
        return _to_surf(frames[idx], cell_w, cell_h)

    def switch_pair(new_idx: int) -> None:
        nonlocal pair_idx, good_frames, bad_frames, frame_idx, paused
        pair_idx    = new_idx % len(pairs)
        good_frames, bad_frames = _fetch(pair_idx, screen, font_label)
        frame_idx   = 0
        paused      = True

    # Pre-compute branch markers for global bar
    branch_markers = [(p.get("branch_pos", -1), p.get("kind", "bad")) for p in pairs]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RIGHT:
                    frame_idx = min(frame_idx + 1, n_frames() - 1)
                    paused = True
                    key_held[k] = time.time()
                elif k == pygame.K_LEFT:
                    frame_idx = max(frame_idx - 1, 0)
                    paused = True
                    key_held[k] = time.time()
                elif k == pygame.K_DOWN:
                    switch_pair(pair_idx + 1)
                elif k == pygame.K_UP:
                    switch_pair(pair_idx - 1)
            elif event.type == pygame.KEYUP:
                key_held.pop(event.key, None)

        # Held arrow scrub
        now = time.time()
        for key, step in ((pygame.K_RIGHT, 1), (pygame.K_LEFT, -1)):
            if key in key_held and now - key_held[key] > HOLD_DELAY:
                frame_idx = max(0, min(frame_idx + step, n_frames() - 1))

        if not paused:
            frame_idx += 1
            if frame_idx >= n_frames():
                frame_idx = n_frames() - 1
                paused = True

        # ── draw video panels ─────────────────────────────────────────────────
        screen.blit(get_surf(good_frames, frame_idx), (0, 0))
        screen.blit(get_surf(bad_frames,  frame_idx), (cell_w, 0))

        # panel labels
        lbl_g = font_label.render("GOOD", True, (80, 220, 80))
        kind = pairs[pair_idx].get("kind", "bad")
        bad_color = (220, 80, 80) if kind == "dead" else (80, 160, 220)
        lbl_b = font_label.render(f"BAD ({kind.upper()})",  True, bad_color)
        screen.blit(lbl_g, (6, 4))
        screen.blit(lbl_b, (cell_w + 6, 4))


        # ── panel background ──────────────────────────────────────────────────
        pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(0, cell_h, win_w, PANEL_HEIGHT))

        # ── info text ─────────────────────────────────────────────────────────
        bp = branch_markers[pair_idx][0]
        pivot = pairs[pair_idx].get("pivot", -1)
        nf  = n_frames()
        gr  = pairs[pair_idx].get("good_reward")
        br  = pairs[pair_idx].get("bad_reward")
        good_score = f"({gr:+.0f})" if gr is not None else ""
        bad_score  = f"({br:+.0f})" if br is not None else ""
        info = (
            f"pair {pair_idx + 1}/{len(pairs)}  "
            f"branch@{bp}/{good_total} (pivot={pivot})  "
            f"frame {frame_idx}/{nf - 1}  "
            f"good={len(good_frames)}{good_score}  bad={len(bad_frames)}{bad_score}  "
            f"{'PAUSED' if paused else 'PLAYING'}"
        )
        screen.blit(font_info.render(info, True, (160, 160, 160)), (M, cell_h + 6))



        # ── local bar (frame position within current pair) ────────────────────
        pygame.draw.rect(screen, (50, 50, 50), (local_x, local_y, local_w, bar_h))

        # good trace extent (top half, green)
        if good_frames:
            gw = int(len(good_frames) / nf * local_w)
            pygame.draw.rect(screen, (30, 140, 30),
                             (local_x, local_y, gw, bar_h // 2))
        # bad trace extent (bottom half)
        if bad_frames:
            bw = int(len(bad_frames) / nf * local_w)
            bg_color = (140, 30, 30) if kind == "dead" else (30, 80, 140)
            pygame.draw.rect(screen, bg_color,
                             (local_x, local_y + bar_h // 2, bw, bar_h // 2))

        # pivot marker
        if pivot >= 0 and nf > 1:
            px = local_x + int(pivot / (nf - 1) * local_w)
            pygame.draw.line(screen, (200, 200, 50),
                             (px, local_y - 2), (px, local_y + bar_h + 2), 2)

        # frame cursor
        if nf > 1:
            cx = local_x + int(frame_idx / (nf - 1) * local_w)
            cursor_color = (255, 80, 80) if kind == "dead" else (80, 200, 255)
            pygame.draw.line(screen, cursor_color,
                             (cx, local_y - 2), (cx, local_y + bar_h + 2), 2)

        pygame.display.flip()
        clock.tick(FPS_PLAYBACK)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python annotate/gui_inspect_dpo.py <graph.pkl>")
        sys.exit(1)
    main(sys.argv[1])
