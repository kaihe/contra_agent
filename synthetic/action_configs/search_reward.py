"""Search-only reward for mc_search (generation phase).

A deliberate *copy/fork* of ``contra/reward.py``. The reinforcement-learning
phase optimises a learnable shaping signal; the generation phase instead wants
the *cleanest winning trace, found efficiently*. So this module keeps the same
level-aware advancement/combat/terminal components (reused directly from
contra.reward to avoid drift) but adds per-button hold penalties the RL reward
must NOT have — one weight per button, keyed by the action-table nicknames
``F`` (fire/B), ``J`` (jump/A) and ``U`` / ``D`` / ``L`` / ``R`` (d-pad):

  * each is charged on every step that button is held (see ``BUTTON_BITS``).

Charging on every step a button is down (not just the 0->1 press edge) drives
traces toward minimal button use during search. With only ``F``/``J`` set (the
default) this reproduces the old fire/jump penalty that replaced the post-hoc
prune_actions pass. Giving every button but ``R`` a small cost additionally
breaks ties toward the simplest reward-equivalent action — Right (forward) stays
free, so a wasted aim-up/fire is shed. A step that lands a hit still pays for
itself via ``enemy_hp``/``boss_hp``; holding a button when it achieves nothing
goes net negative and the searcher learns to release it.

These penalties require the current action, so :func:`compute_reward` takes the
action vector (which RAM-only ``contra.reward.compute_reward`` does not).
"""

from dataclasses import dataclass

import numpy as np

# Reuse the level-aware component logic verbatim so search and RL agree on what
# "progress / combat / death" mean — only the cleanliness penalties are new.
from contra.reward import reward_components, xscroll

# Mirrors the values that have produced wins (contra/reward_configs/stable.yaml),
# plus the generation-only press penalties.
DEFAULT_REWARD_WEIGHTS = {
    # combat / items (level-agnostic)
    "enemy_hp": 1.0,
    "boss_hp": 1.0,
    "spread_pick": 20.0,
    "rapid_fire": 10.0,
    # terminal (level-agnostic)
    "levelup": 1.0,
    "player_die": -15.0,
    "time_out": 0.0,
    # advancement — only the level's own style term is applied
    "progress": 0.1,        # "forward": per xscroll pixel
    "push_inside": 1.0,     # "inside": dense progress through indoor rooms
    "room_enter": 1.0,      # "inside": per-room milestone
    "core_broken": 1.0,     # "inside": core-clear spike
    "push_up": 1.0,         # "up": dense vertical progress
    # generation-only per-button hold penalties (charged each step the bit is held),
    # keyed by the same button nicknames as the action table (F=fire/B, J=jump/A,
    # U/D/L/R = d-pad). fire/jump default to a real penalty; the d-pad and Right
    # default to 0 so they are inert unless a level opts in. See BUTTON_BITS.
    "F": -0.3,   # fire (B)
    "J": -0.3,   # jump (A)
    "U": 0.0,    # up
    "D": 0.0,    # down
    "L": 0.0,    # left
    "R": 0.0,    # right (canonical forward action; normally left at 0)
}

# Button nickname -> action-vector bit index. Lets a level place a small tie-break
# penalty on every button except Right, so that among reward-equivalent actions
# the search commits the simplest one and BC sees a consistent state->action label
# instead of an arbitrary R/UR coin flip.
BUTTON_BITS = {
    "F": 0,   # fire (B)
    "U": 4,   # up
    "D": 5,   # down
    "L": 6,   # left
    "R": 7,   # right
    "J": 8,   # jump (A)
}


@dataclass(frozen=True)
class RewardConfig:
    """Search reward weights (a superset of the RL weights with press penalties)."""

    name: str
    reward_weights: dict

    @classmethod
    def from_dict(cls, d: dict, name: str = "loaded") -> "RewardConfig":
        weights = DEFAULT_REWARD_WEIGHTS.copy()
        given = d.get("reward_weights", {})
        unknown = sorted(set(given) - set(weights))
        if unknown:
            raise ValueError(f"Unknown reward weight(s) in '{name}': {unknown}")
        weights.update(given)
        return cls(name=name, reward_weights=weights)

    def with_costs(self, **costs: float | None) -> "RewardConfig":
        """Return a copy with per-button hold penalties overridden (from level YAML).

        Each keyword is a button-nickname weight key (see ``BUTTON_BITS``); a
        ``None`` value is ignored, so unspecified buttons keep their default.
        """
        w = dict(self.reward_weights)
        for key, val in costs.items():
            if val is None:
                continue
            if key not in w:
                raise ValueError(f"Unknown cost weight: {key!r}")
            w[key] = val
        return RewardConfig(name=self.name, reward_weights=w)


DEFAULT_CONFIG = RewardConfig(name="clean", reward_weights=DEFAULT_REWARD_WEIGHTS.copy())


def press_penalty(action: np.ndarray, weights: dict) -> float:
    """Per-button hold penalty: sum the cost of every pressed bit in `action`.

    Each search step is one decision (the action is held for `skip` frames), so a
    step pays the cost of each button it holds (``BUTTON_BITS``) regardless of the
    previous step. With every button but Right given a small cost, this biases the
    search toward the simplest reward-equivalent action; with only fire/jump set
    (the default) it reduces to the old fire/jump press penalty.
    """
    cost = 0.0
    for nick, bit in BUTTON_BITS.items():
        if action[bit]:
            cost += weights[nick]
    return cost


def compute_reward(pre_ram: np.ndarray, curr_ram: np.ndarray,
                   config: RewardConfig = DEFAULT_CONFIG,
                   action: np.ndarray | None = None) -> float:
    """Single-step search reward: RL components + fire/jump hold penalties.

    When ``action`` is given, the fire/jump hold penalty is added. With
    ``action=None`` this reduces to the plain RL reward, so callers that don't
    track actions still work.
    """
    components = reward_components(
        pre_ram, curr_ram,
        config.reward_weights,
        prev_xscroll=xscroll(pre_ram),
        timed_out=False,
    )
    total = sum(components.values())
    if action is not None:
        total += press_penalty(action, config.reward_weights)
    return total
