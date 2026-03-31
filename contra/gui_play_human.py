"""
play_human.py — Record Contra playthroughs for imitation learning
==================================================================

Human inputs are sampled at 20 Hz (one new action every 3 emulator frames).
Between ticks the last action is repeated, matching the 20 Hz agent constraint.

Usage:
    python play_human.py Level3

Keyboard:
    W/A/S/D  — up/left/down/right
    J        — jump (NES A)
    F        — fire (NES B)
    Enter    — START
    P        — set anchor (rewind point on next death)
    ESC      — quit
"""

from __future__ import annotations

import gzip
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pygame
import stable_retro as retro

# ── Config ────────────────────────────────────────────────────────────────────

GAME          = "Contra-Nes"
RECORDING_DIR    = os.path.join(os.path.dirname(__file__), "contra", "human_recordings")
SPREAD_STATES_DIR = os.path.join(os.path.dirname(__file__), "contra", "spread_gun_states")
SCALE         = 3
NES_W, NES_H  = 240, 224
AGENT_HZ      = 20
EMU_HZ        = 60
INPUT_PERIOD  = EMU_HZ // AGENT_HZ  # 3 — latch a new action every N emu frames

# NES button indices: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
KEY_MAP = {
    pygame.K_w:      4,  # UP
    pygame.K_s:      5,  # DOWN
    pygame.K_a:      6,  # LEFT
    pygame.K_d:      7,  # RIGHT
    pygame.K_j:      8,  # A (jump)
    pygame.K_f:      0,  # B (fire)
    pygame.K_RETURN: 3,  # START
}

# ── Recording ─────────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecording:
    initial_state: bytes            = b""
    start_time:    float            = field(default_factory=time.time)
    actions:       list[np.ndarray] = field(default_factory=list)

    def add_frame(self, action: np.ndarray) -> None:
        self.actions.append(action.copy())

    def truncate(self, length: int) -> None:
        self.actions = self.actions[:length]

    def save(self, label: str, outcome: str) -> None:
        if not self.actions:
            return
        out_dir  = os.path.join(RECORDING_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        ts       = time.strftime("%m%d%H%M", time.localtime(self.start_time))
        filename = f"{label}_{outcome}_{ts}.npz"
        np.savez_compressed(
            os.path.join(out_dir, filename),
            actions       = np.array(self.actions, dtype=np.int8),
            level         = np.array(label),
            outcome       = np.array(outcome),
            fps           = np.array(AGENT_HZ),
            game          = np.array(GAME),
            initial_state = np.frombuffer(self.initial_state, dtype=np.uint8),
        )
        duration = time.time() - self.start_time
        print(f"\n  [{outcome.upper()}] {filename}")
        print(f"  frames={len(self.actions)}  {duration:.0f}s\n")


# ── Env helper ────────────────────────────────────────────────────────────────

def _load_state_bytes(level: str) -> bytes | None:
    """Return state bytes for the given level, or None to use retro's default.

    Level 1 always uses the game's default state.
    Level 2+ uses the spread-gun state if available, otherwise falls back to
    the legacy STATES_DIR, then the game default.
    """
    if level == "Level1":
        return None

    candidates = [
        os.path.join(SPREAD_STATES_DIR, f"{level}.state")
    ]
    for path in candidates:
        try:
            try:
                with gzip.open(path, "rb") as f:
                    return f.read()
            except OSError:
                with open(path, "rb") as f:
                    return f.read()
        except OSError:
            continue
    return None


def make_env(level: str) -> tuple[retro.RetroEnv, np.ndarray, bytes]:
    env = retro.make(
        game=GAME, state=level,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    obs, _ = env.reset()

    state_bytes = _load_state_bytes(level)
    if state_bytes is not None:
        env.initial_state = state_bytes
        obs, _ = env.reset()
    else:
        state_bytes = env.initial_state

    return env, obs, state_bytes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "Level1"

    os.makedirs(RECORDING_DIR, exist_ok=True)

    pygame.init()
    pygame.display.set_caption(f"Contra — {level}  (20 Hz)")
    screen = pygame.display.set_mode((NES_W * SCALE, NES_H * SCALE))
    clock  = pygame.time.Clock()

    env, obs, level_initial_state = make_env(level)

    wins          = 0
    level_label   = level  # updated as the player progresses through levels
    current_ep    = EpisodeRecording(initial_state=level_initial_state)
    current_ep.add_frame(np.zeros(9, dtype=np.int8))
    start_level   = None
    frame_count   = INPUT_PERIOD - 1  # first iteration hits latch immediately
    latched_action = np.zeros(9, dtype=np.int8)
    prev_lives    = None
    anchor_state  = None  # (emu_state_bytes, rec_len)
    pending_anchor = False
    rewind_grace  = 0
    rewind_data   = None

    def start_new_episode(new_initial_state: bytes, new_label: str) -> None:
        nonlocal level_initial_state, level_label, current_ep, start_level
        nonlocal frame_count, latched_action, prev_lives, anchor_state, pending_anchor
        nonlocal rewind_grace, rewind_data
        level_initial_state = new_initial_state
        level_label         = new_label
        current_ep          = EpisodeRecording(initial_state=new_initial_state)
        current_ep.add_frame(np.zeros(9, dtype=np.int8))
        start_level         = None
        frame_count         = INPUT_PERIOD - 1
        latched_action      = np.zeros(9, dtype=np.int8)
        prev_lives          = None
        anchor_state        = None
        pending_anchor      = False
        rewind_grace        = 0
        rewind_data         = None
        pygame.display.set_caption(f"Contra — {new_label}  (20 Hz)")

    def do_rewind(emu_state: bytes, rec_len: int) -> None:
        nonlocal frame_count, latched_action, prev_lives
        env.em.set_state(emu_state)
        env.initial_state = level_initial_state
        current_ep.truncate(rec_len)
        frame_count    = INPUT_PERIOD - 1
        latched_action = np.zeros(9, dtype=np.int8)
        prev_lives     = None

    print(f"Playing {level}  |  P=set anchor  ESC=quit\n")

    running = True
    while running:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    pending_anchor = True
                    print("  [P] Anchor pending — latches at next period boundary")

        if not running:
            break

        # ── Input: sample at 20 Hz, repeat for the remaining 2 emu frames ─────
        frame_count += 1
        if frame_count % INPUT_PERIOD == 0:
            raw = np.zeros(9, dtype=np.int8)
            keys = pygame.key.get_pressed()
            for key, idx in KEY_MAP.items():
                if keys[key]:
                    raw[idx] = 1
            latched_action = raw

        # ── Latch anchor at period boundary, before the step ──────────────────
        if pending_anchor and frame_count % INPUT_PERIOD == 0:
            anchor_state   = (env.em.get_state(), len(current_ep.actions))
            pending_anchor = False
            print(f"  [P] Anchor set at frame {anchor_state[1]}")

        # ── Step ──────────────────────────────────────────────────────────────
        obs, _, terminated, truncated, info = env.step(latched_action)

        if frame_count % INPUT_PERIOD == 0:
            current_ep.add_frame(latched_action)

        if start_level is None:
            start_level = int(info.get("level", 0))

        # ── Render ────────────────────────────────────────────────────────────
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        pygame.transform.scale(surf, (NES_W * SCALE, NES_H * SCALE), screen)
        pygame.display.flip()

        # ── Episode logic ─────────────────────────────────────────────────────
        cur_level   = int(info.get("level", start_level))
        lives_val   = int(info.get("lives", 0))
        level_clear = cur_level > start_level
        player_died = prev_lives is not None and lives_val < prev_lives
        prev_lives  = lives_val

        if level_clear:
            wins += 1
            current_ep.save(level_label, "win")
            new_label = f"Level{cur_level + 1}"
            start_new_episode(env.em.get_state(), new_label)

        elif rewind_grace > 0:
            rewind_grace -= 1
            if rewind_grace == 0 and rewind_data is not None:
                emu_state, rec_len = rewind_data
                rewind_data = None
                do_rewind(emu_state, rec_len)
                print(f"  [Rewind] Restored to anchor (frame {rec_len})")

        elif player_died:
            if anchor_state is not None:
                rewind_grace = 10
                rewind_data  = anchor_state
            else:
                env.em.set_state(level_initial_state)
                env.initial_state = level_initial_state
                start_new_episode(level_initial_state, level_label)
                print("  [Restart] No anchor — restarted from level beginning")

        clock.tick(EMU_HZ)

    pygame.quit()
    env.close()
    print(f"Session done. {wins} win(s) saved.")


if __name__ == "__main__":
    main()
