"""
VLAEnv — single-process stable-retro wrapper that produces VLA-compatible observations.

Design constraints
------------------
- stable_retro allows only ONE emulator instance per process.  VLAEnv owns that
  instance and exposes snapshot()/restore() so the GRPO trainer can branch G group
  members sequentially from the same starting state.
- Observation: two consecutive frames (prev + curr) at IMG_SIZE × IMG_SIZE,
  ImageNet-normalised, plus a 118-dim structured state vector (proprio).
- Action: a list of combined action indices (0..35) to execute as a chunk.
  Each index is decoded to a 9-bit NES MultiBinary array and stepped EMU_SKIP
  times to simulate 20 Hz agent control on the 60 Hz emulator.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import torch
from PIL import Image

import contra  # registers Contra-Nes integration
import stable_retro as retro

from contra.events import compute_reward, EV_LEVELUP, EV_PLAYER_DIE
from contra.game_state import state_from_ram

# ── Constants ─────────────────────────────────────────────────────────────────

GAME     = "Contra-Nes"
EMU_SKIP = 3      # sub-frames per agent step  (60 Hz / 3 = 20 Hz)
IMG_SIZE = 192    # must match BC training

# ImageNet normalisation — same constants as vla/datasets/dataset.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# NES MultiBinary(9): [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
_DPAD_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1: L
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2: R
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: U
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: D
    [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 5: UL
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 6: UR
    [0, 0, 0, 0, 0, 1, 1, 0, 0],  # 7: DL
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 8: DR
], dtype=np.int8)

_BUTTON_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1: Jump (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: Fire (B)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: Fire+Jump
], dtype=np.int8)


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _resize(screen: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) uint8 → (IMG_SIZE, IMG_SIZE, 3) uint8."""
    return np.array(Image.fromarray(screen).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))


def _frames_to_tensor(prev: np.ndarray, curr: np.ndarray) -> torch.Tensor:
    """Two (IMG_SIZE, IMG_SIZE, 3) uint8 frames → [2, 3, H, W] float32 normalised."""
    frames = np.stack([prev, curr])                               # [2, H, W, 3]
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)  # [2, 3, H, W]
    return (t - _MEAN) / _STD


def _decode_nes(action: int) -> np.ndarray:
    """Combined action index 0..35 → 9-bit NES MultiBinary."""
    dpad, button = action // 4, action % 4
    return np.clip(_DPAD_TABLE[dpad] + _BUTTON_TABLE[button], 0, 1).astype(np.int8)


# ── VLAEnv ────────────────────────────────────────────────────────────────────

class VLAEnv:
    """
    Single stable-retro Contra env wrapped for VLA inference.

    Observation dict keys
    ---------------------
    images  : torch.Tensor [2, 3, IMG_SIZE, IMG_SIZE] float32 normalised
    proprio : torch.Tensor [118] float32
    """

    def __init__(self, level: str = "Level1") -> None:
        self._level        = level
        self._env          = None
        self._prev_frame   = None   # (IMG_SIZE, IMG_SIZE, 3) uint8
        self._curr_frame   = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """(Re-)open env and restore to the level's initial save state."""
        if self._env is not None:
            self._env.close()
        self._env = self._open_env()
        screen = self._env.em.get_screen().copy()
        f = _resize(screen)
        self._prev_frame = f
        self._curr_frame = f.copy()
        return self._obs()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── core env ops ──────────────────────────────────────────────────────────

    # Terminal status constants
    RUNNING = ""
    DEAD    = "DEAD"
    DONE    = "DONE"

    def step(self, action_chunk: list[int]) -> tuple[dict, float, str]:
        """
        Execute one chunk of combined action indices.

        Each action in the chunk is stepped EMU_SKIP times.
        Reward is accumulated (not discounted — caller handles discounting).

        Returns
        -------
        obs    : dict with "images" and "proprio"
        reward : float, sum of compute_reward across the chunk
        status : "" (running) | "DEAD" (player died) | "DONE" (level cleared)
        """
        total_reward = 0.0
        status       = self.RUNNING

        for action_idx in action_chunk:
            nes_action = _decode_nes(action_idx)
            pre_ram    = self._env.unwrapped.get_ram().copy()

            terminated = truncated = False
            for _ in range(EMU_SKIP):
                _, _, terminated, truncated, _ = self._env.step(nes_action.copy())
                if terminated or truncated:
                    break

            curr_ram      = self._env.unwrapped.get_ram()
            total_reward += compute_reward(pre_ram, curr_ram)

            if EV_PLAYER_DIE.trigger(pre_ram, curr_ram) or terminated or truncated:
                status = self.DEAD
                break
            if EV_LEVELUP.trigger(pre_ram, curr_ram):
                status = self.DONE
                break

        self._prev_frame = self._curr_frame
        self._curr_frame = _resize(self._env.em.get_screen().copy())
        return self._obs(), total_reward, status

    # ── state branching ───────────────────────────────────────────────────────

    def snapshot(self) -> bytes:
        """Return the current emulator save-state as raw bytes."""
        return self._env.em.get_state()

    def restore(self, state: bytes) -> dict:
        """
        Restore the emulator to a previously snapshotted state.

        Resets the 2-frame buffer to the restored screen so the model sees a
        consistent observation after branching.
        """
        self._env.em.set_state(state)
        self._env.data.update_ram()
        screen = self._env.em.get_screen().copy()
        f = _resize(screen)
        self._prev_frame = f
        self._curr_frame = f.copy()
        return self._obs()

    # ── internals ─────────────────────────────────────────────────────────────

    def _obs(self) -> dict:
        ram = self._env.unwrapped.get_ram()
        return {
            "images":  _frames_to_tensor(self._prev_frame, self._curr_frame),
            "proprio": torch.from_numpy(state_from_ram(ram)),
        }

    def _open_env(self):
        env = retro.make(
            game=GAME,
            state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        env.reset()

        state_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "contra",
            "integration", "Contra-Nes", f"{self._level}.state",
        )
        state_bytes = None
        for opener in (gzip.open, open):
            try:
                with opener(state_path, "rb") as fh:
                    state_bytes = fh.read()
                break
            except OSError:
                continue
        if state_bytes is None:
            raise FileNotFoundError(f"State file not found: {state_path}")

        env.em.set_state(state_bytes)
        env.data.update_ram()
        return env
