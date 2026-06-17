"""Shared reward config + computation for PPO and mc_search.

Single source of truth for the reward signal so the Monte-Carlo searcher
(``synthetic/mc_search.py``) and the trained policy (``ppo/contra_wrapper.py``)
optimise the *same* objective: a win path found by search is then meaningful
evidence that the reward shaping is learnable.

A reward config is a YAML file under ``contra/reward_configs/<name>.yaml``:

    reward_weights:        # merged onto defaults; may be partial
      <event>: <weight>
      ...

``reward_weights`` may be partial; missing keys fall back to
:data:`DEFAULT_REWARD_WEIGHTS`. The event keys match the ``EV_*`` triggers in
``contra/events.py`` (one wrapper supports every level via ``level_advance_style``).
"""

import os
from dataclasses import dataclass

import numpy as np
import yaml

from contra.events import (
    ADDR_LEVEL,
    ADDR_XSCROLL_HI,
    EV_BOSS_HIT,
    EV_CORE_BROKEN,
    EV_LEVELUP,
    EV_PLAYER_DIE,
    EV_PUSH_INSIDE,
    EV_PUSH_UP,
    EV_REGULAR_ENEMY_HIT,
    EV_ROOM_ENTER,
    EV_SPREAD_PICK,
    level_advance_style,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "reward_configs")

DEFAULT_REWARD_WEIGHTS = {
    "enemy_hp": 1.0,
    "boss_hp": 1.0,
    "progress": 1.0 / 60.0,    # "forward" levels: per xscroll pixel
    "core_broken": 10.0,       # "inside" levels: wall core destroyed (sparse)
    "push_inside": 0.5,        # "inside" levels: per step walking through the door
    "room_enter": 10.0,        # "inside" levels: entered the next indoor screen
    "push_up": 0.5,            # "up" levels: per vertical-scroll pixel
    "spread_pick": 20.0,
    "levelup": 100.0,
    "player_die": -15.0,
    "time_out": -10.0,
}


def xscroll(ram: np.ndarray) -> int:
    return int(ram[100]) << 8 | int(ram[101])


def reward_components(
    pre_ram: np.ndarray,
    curr_ram: np.ndarray,
    weights: dict[str, float],
    prev_xscroll: int,
    timed_out: bool = False,
) -> dict[str, float]:
    """Level-aware reward components.

    The combat / item / terminal components are level-agnostic. The *advancement*
    component is selected from the level read out of RAM (ADDR_LEVEL), using the
    same per-level advancement style as the mc_search event system:
      "forward" : horizontal scroll progress (side-scroll levels)
      "inside"  : core destroyed + walking through door + entering next room (indoor)
      "up"      : vertical scroll progress (climbing levels)
    So one wrapper supports every level — it just needs to start in the right state.
    """
    components = {
        "enemy_hp": weights["enemy_hp"] * EV_REGULAR_ENEMY_HIT.trigger(pre_ram, curr_ram),
        "boss_hp": weights["boss_hp"] * EV_BOSS_HIT.trigger(pre_ram, curr_ram),
        "spread_pick": weights["spread_pick"] * EV_SPREAD_PICK.trigger(pre_ram, curr_ram),
        "levelup": weights["levelup"] * EV_LEVELUP.trigger(pre_ram, curr_ram),
        "player_die": weights["player_die"] * EV_PLAYER_DIE.trigger(pre_ram, curr_ram),
        "time_out": weights["time_out"] * float(timed_out),
    }

    style = level_advance_style(int(pre_ram[ADDR_LEVEL]))
    if style == "inside":
        components["core_broken"] = weights["core_broken"] * EV_CORE_BROKEN.trigger(pre_ram, curr_ram)
        components["push_inside"] = weights["push_inside"] * EV_PUSH_INSIDE.trigger(pre_ram, curr_ram)
        components["room_enter"] = weights["room_enter"] * EV_ROOM_ENTER.trigger(pre_ram, curr_ram)
    elif style == "up":
        components["push_up"] = weights["push_up"] * EV_PUSH_UP.trigger(pre_ram, curr_ram)
    else:  # "forward"
        progress = float(xscroll(curr_ram) - prev_xscroll)
        components["progress"] = weights["progress"] * progress

    return components


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights loaded from a reward_configs YAML file."""

    name: str
    reward_weights: dict

    def to_dict(self) -> dict:
        return {
            "reward_weights": dict(self.reward_weights),
        }

    @classmethod
    def from_dict(cls, d: dict, name: str = "loaded") -> "RewardConfig":
        weights = DEFAULT_REWARD_WEIGHTS.copy()
        given = d.get("reward_weights", {})
        unknown = sorted(set(given) - set(weights))
        if unknown:
            raise ValueError(f"Unknown reward weight(s) in '{name}': {unknown}")
        weights.update(given)
        return cls(
            name=name,
            reward_weights=weights,
        )


def load(name: str) -> RewardConfig:
    """Load a reward config from ``contra/reward_configs/<name>.yaml``."""
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    with open(path) as f:
        return RewardConfig.from_dict(yaml.safe_load(f), name=name)


# Default config (full default weights) for callers that don't name one.
DEFAULT_CONFIG = RewardConfig(name="default", reward_weights=DEFAULT_REWARD_WEIGHTS.copy())


def compute_reward(pre_ram: np.ndarray, curr_ram: np.ndarray,
                   config: RewardConfig = DEFAULT_CONFIG) -> float:
    """Single-step total reward for mc_search.

    Stateless: `progress` is measured within this one step as
    ``xscroll(curr) - xscroll(pre)`` (a step spans `skip` frames).
    """
    components = reward_components(
        pre_ram, curr_ram,
        config.reward_weights,
        prev_xscroll=xscroll(pre_ram),
        timed_out=False,
    )
    return sum(components.values())
