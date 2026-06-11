"""Single-emulator Contra wrapper used by VLA post-training."""

from __future__ import annotations

import warnings

import numpy as np
import stable_retro as retro
import torch
from PIL import Image

from contra.events import EV_LEVELUP, EV_PLAYER_DIE
from contra.game_state import state_from_ram
from vla.datasets.preprocess import decode_action36

from .rewards import RewardProfile, make_step_info, shaped_reward

warnings.filterwarnings("ignore", message=".*Gym.*")

GAME = "Contra-Nes"
SKIP = 3
IMG_SIZE = 192

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess_screen(screen: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(screen).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.array(image, copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
    return (tensor - _MEAN) / _STD


class VLAEnv:
    RUNNING = "RUNNING"
    DEAD = "DEAD"
    DONE = "DONE"

    def __init__(
        self,
        level: str = "Level1",
        reward_profile: RewardProfile | None = None,
        render_mode=None,
    ) -> None:
        self.level = level
        self.reward_profile = reward_profile or RewardProfile()
        self.render_mode = render_mode
        self.env = self._open_env()
        self.step_idx = 0

    def _open_env(self):
        return retro.make(
            game=GAME,
            state=self.level,
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.IMAGE,
            render_mode=self.render_mode,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )

    def _reopen_env(self) -> None:
        self.env.close()
        self.env = retro.make(
            game=GAME,
            state=self.level,
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.IMAGE,
            render_mode=self.render_mode,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )

    @property
    def em(self):
        return self.env.em

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, level: str | None = None) -> dict[str, torch.Tensor]:
        if level is not None and level != self.level:
            self.level = level
            self._reopen_env()
        self.env.reset()
        self.step_idx = 0
        return self.current_obs()

    def current_obs(self) -> dict[str, torch.Tensor]:
        ram = self.env.unwrapped.get_ram()
        return {
            "images": _preprocess_screen(self.env.em.get_screen().copy()).unsqueeze(0),
            "proprio": torch.from_numpy(state_from_ram(ram).astype(np.float32, copy=True)),
        }

    def step(self, action: int | list[int] | tuple[int, ...]) -> tuple[dict, float, str, dict]:
        if isinstance(action, (list, tuple)):
            obs, total, status, info = self.current_obs(), 0.0, self.RUNNING, {}
            for item in action:
                obs, reward, status, info = self.step(int(item))
                total += reward
                if status != self.RUNNING:
                    break
            return obs, total, status, info

        nes_action = decode_action36(int(action))
        status = self.RUNNING
        pre_ram = self.env.unwrapped.get_ram().copy()
        curr_ram = pre_ram

        for _ in range(SKIP):
            frame_pre_ram = self.env.unwrapped.get_ram().copy()
            self.env.step(nes_action.copy())
            curr_ram = self.env.unwrapped.get_ram().copy()

            if EV_PLAYER_DIE.trigger(frame_pre_ram, curr_ram):
                status = self.DEAD
                break
            if EV_LEVELUP.trigger(frame_pre_ram, curr_ram):
                status = self.DONE
                break

        reward, components = shaped_reward(pre_ram, curr_ram, self.reward_profile)
        if components["player_die"] != 0.0:
            status = self.DEAD
        elif components["levelup"] != 0.0:
            status = self.DONE
        info = make_step_info(pre_ram, curr_ram, self.step_idx, components)
        self.step_idx += 1
        return self.current_obs(), reward, status, info

    def snapshot(self) -> bytes:
        return self.env.em.get_state()

    def restore(self, state: bytes) -> dict[str, torch.Tensor]:
        self.env.em.set_state(state)
        self.env.data.update_ram()
        return self.current_obs()

    def close(self) -> None:
        self.env.close()
