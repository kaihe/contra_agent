"""Level-1 reward shaping for ContraVLA GRPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contra.events import (
    ADDR_LIVES,
    EV_ENEMY_HIT,
    EV_GUN_PICKUP,
    EV_GUN_POWERUP,
    EV_LEVELUP,
    EV_PLAYER_DIE,
    EV_PUSH_FORWARD,
    EV_SPREAD_LOST,
    get_level,
    scan_events,
)


@dataclass(frozen=True)
class RewardProfile:
    enemy_hp: float = 2.0
    progress_px: float = 1.0 / 60.0
    max_progress_px: float = 30.0
    gun_pickup: float = 20.0
    gun_powerup: float = 20.0
    spread_lost: float = -100.0
    levelup: float = 2000.0
    player_die: float = -1000.0


def xscroll(ram: np.ndarray) -> int:
    return int(ram[100]) << 8 | int(ram[101])


def reward_components(
    pre_ram: np.ndarray,
    curr_ram: np.ndarray,
    profile: RewardProfile,
) -> dict[str, float]:
    progress = min(max(EV_PUSH_FORWARD.trigger(pre_ram, curr_ram), 0.0), profile.max_progress_px)
    return {
        "enemy_hp": profile.enemy_hp * EV_ENEMY_HIT.trigger(pre_ram, curr_ram),
        "progress": profile.progress_px * progress,
        "gun_pickup": profile.gun_pickup * EV_GUN_PICKUP.trigger(pre_ram, curr_ram),
        "gun_powerup": profile.gun_powerup * EV_GUN_POWERUP.trigger(pre_ram, curr_ram),
        "spread_lost": profile.spread_lost * EV_SPREAD_LOST.trigger(pre_ram, curr_ram),
        "levelup": profile.levelup * EV_LEVELUP.trigger(pre_ram, curr_ram),
        "player_die": profile.player_die * EV_PLAYER_DIE.trigger(pre_ram, curr_ram),
    }


def shaped_reward(
    pre_ram: np.ndarray,
    curr_ram: np.ndarray,
    profile: RewardProfile,
) -> tuple[float, dict[str, float]]:
    components = reward_components(pre_ram, curr_ram, profile)
    return float(sum(components.values())), components


def make_step_info(
    pre_ram: np.ndarray,
    curr_ram: np.ndarray,
    step_idx: int,
    components: dict[str, float],
) -> dict:
    return {
        "xscroll": xscroll(curr_ram),
        "yscroll": int(curr_ram[101]),
        "level": get_level(curr_ram),
        "lives": int(curr_ram[ADDR_LIVES]),
        "events": scan_events(pre_ram, curr_ram, step_idx),
        "reward_components": components,
    }
