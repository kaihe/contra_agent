"""Shared action-space + frame-skip config for mc_search and PPO.

Single source of truth for the two things that must agree between the
Monte-Carlo searcher (``synthetic/mc_search.py``) and the trained policy
(``ppo/contra_wrapper.py``):

  * the discrete set of NES actions (a flat list of button vectors), and
  * the frame ``skip`` (how many NES frames one decision is held for).

If these diverge, a win path discovered by search is not reproducible by the
policy (different reachable states), which is exactly the bug this module
exists to prevent. Both consumers import :data:`DEFAULT` from here.

Each action is a length-9 NES button vector in bit order
``[B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]``. The action space is a
plain ``Discrete(num_actions)``: the agent picks an index, and the vector at
that index is the buttons held for ``skip`` frames.

Config values live in JSON under ``contra/action_configs/<name>.json``; this
module only holds the dataclass and the loader. :data:`DEFAULT` is loaded from
``baseline.json``.

Note: ``contra/inputs.py`` keeps a separate, legacy two-head encoding still
used by the bigram builder / pixel2play / annotator. This module is the
canonical config for the mc_search<->PPO loop only.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "action_configs")


@dataclass(frozen=True)
class ActionSpace:
    """A flat discrete action space (named NES button vectors) + frame skip.

    ``names[i]`` is the human-readable label (e.g. "RF" = Right+Fire) for the
    button vector ``actions[i]``. Index order is the JSON insertion order, so it
    is stable as long as baseline.json keeps its key order.
    """

    names: tuple    # tuple of action names, parallel to `actions`
    actions: tuple  # tuple of length-9 NES button vectors
    skip: int = 8   # NES frames one decision is held for

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def actions_np(self, dtype=np.uint8) -> np.ndarray:
        """All actions as an (num_actions, 9) array."""
        return np.array(self.actions, dtype=dtype)

    def nes_action(self, idx: int, dtype=np.uint8) -> np.ndarray:
        """The NES button vector for action index ``idx``."""
        return self.actions_np(dtype)[idx]

    def to_dict(self) -> dict:
        """JSON-serialisable form (used when embedding config into a model)."""
        return {"skip": self.skip,
                "actions": {n: list(a) for n, a in zip(self.names, self.actions)}}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpace":
        actions = d["actions"]  # {name: vector}, insertion-ordered
        return cls(
            names=tuple(actions.keys()),
            actions=tuple(tuple(v) for v in actions.values()),
            skip=d.get("skip", 8),
        )


def load(name: str = "baseline") -> ActionSpace:
    """Load an action config from ``contra/action_configs/<name>.json``."""
    path = os.path.join(CONFIG_DIR, f"{name}.json")
    with open(path) as f:
        return ActionSpace.from_dict(json.load(f))


# Canonical config, shared with mc_search so a searched win path is
# policy-reproducible (see baseline.json).
DEFAULT = load("baseline")
