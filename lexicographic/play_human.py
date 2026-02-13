"""
play_human — Play Contra with keyboard and record RAM for learnfun
===================================================================

Controls:
    W / A / S / D  — UP / LEFT / DOWN / RIGHT
    K              — A (jump)
    J              — B (fire)
    ENTER          — START (pause/unpause)
    ESC            — Quit and save

Records RAM snapshots and input actions every frame. Saves raw data
to a .npz file for later objective discovery via learnfun.

Usage:
    python play_human.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pygame
import stable_retro as retro

# =============================================================================
# CONFIG
# =============================================================================

GAME = "Contra-Nes"
STATE = "Level1"
RECORDING_DIR = os.path.join(os.path.dirname(__file__), "recordings")


# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Indices:      0  1     2       3      4   5     6     7      8

# Keyboard -> NES button index
KEY_MAP = {
    pygame.K_w: 4,        # W -> UP
    pygame.K_a: 6,        # A -> LEFT
    pygame.K_s: 5,        # S -> DOWN
    pygame.K_d: 7,        # D -> RIGHT
    pygame.K_SPACE: 8,    # Space -> A (jump)
    pygame.K_j: 0,   # Left Shift -> B (fire)
    pygame.K_RETURN: 3,   # ENTER -> START
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RECORDING_DIR, exist_ok=True)

    env = retro.make(
        game=GAME,
        state=STATE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    obs, _ = env.reset()

    # NES native resolution, 3x scale
    h, w = obs.shape[:2]  # 224, 240
    scale = 3

    pygame.init()
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Contra — Human Play (ESC to quit & save)")
    clock = pygame.time.Clock()

    ram_snapshots: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    ram_snapshots.append(env.get_ram().copy())

    frame_count = 0
    running = True

    print(f"Playing {GAME} / {STATE}")
    print(f"  Controls: WASD=move, Space=jump, LShift=fire, ENTER=start, ESC=quit")
    print(f"  Recording to {RECORDING_DIR}/")

    while running:
        # Build action from held keys
        action = np.zeros(9, dtype=np.int8)
        keys = pygame.key.get_pressed()
        for key, nes_idx in KEY_MAP.items():
            if keys[key]:
                action[nes_idx] = 1

        # Step emulator
        obs, _, terminated, truncated, info = env.step(action)
        ram_snapshots.append(env.get_ram().copy())
        actions.append(action.copy())
        frame_count += 1

        # Render: obs is (H, W, 3) RGB, scale up with pygame
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()
            ram_snapshots.append(env.get_ram().copy())

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        clock.tick(60)  # NES runs at ~60fps

    pygame.quit()
    env.close()

    # Save raw data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"human_{timestamp}_{frame_count}f.npz"
    filepath = os.path.join(RECORDING_DIR, filename)

    np.savez_compressed(
        filepath,
        ram=np.array(ram_snapshots, dtype=np.uint8),
        actions=np.array(actions, dtype=np.int8),
    )
    print(f"\nSaved {frame_count} frames to {filepath}")
    print(f"  RAM: {len(ram_snapshots)} snapshots x {ram_snapshots[0].shape[0]} bytes")
    print(f"  Actions: {len(actions)} frames x 9 buttons")


if __name__ == "__main__":
    main()
