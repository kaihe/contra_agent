"""Search-only action space for mc_search (generation phase).

A deliberate *copy/fork* of the RL action space: the goal here is to generate
clean winning traces as efficiently as possible, which is a different objective
from the reinforcement-learning phase. Keeping this separate from
``contra/action_space.py`` lets the searcher use small, level-tailored action
tables without touching the canonical space PPO trains against.

Traces store raw 9-bit NES button vectors, so a trace produced with a trimmed
search table is still replayable / learnable by any downstream consumer — the
table only constrains what the *search* explores.

Per-level tables live in ``synthetic/action_configs/level<N>.yaml`` and are
curated from winning-trace action histograms (see analyze_action_usage.py).
Levels without a dedicated file fall back to the RL baseline table.

Bit order: ``[B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]`` (B=fire, A=jump).
"""

import os
from dataclasses import dataclass, field

import yaml

from contra.action_space import ActionSpace, DEFAULT as BASELINE
from synthetic.action_configs.search_reward import BUTTON_BITS

# Per-level YAML tables live alongside this module in synthetic/action_configs/.
CONFIG_DIR = os.path.dirname(__file__)

# Per-button hold-penalty keys a level's ``costs:`` block may set: the action-table
# button nicknames F/J/U/D/L/R (one per button). Right is the canonical forward
# action — ``R`` is allowed but normally left at 0.
ALLOWED_COST_KEYS = set(BUTTON_BITS)
ALLOWED_PRIOR_KEYS = {"traces", "mode", "smooth"}


@dataclass(frozen=True)
class SearchLevelConfig:
    """Everything that tunes generation for one level, from a single YAML file.

    Bundles the level's action table with its per-button press penalties and its
    action-prior source, so per-level tuning lives in one place (``level<N>.yaml``).
    ``costs`` holds only the ``*_cost`` weights the file overrides; ``prior`` holds
    the on-the-fly prior config (``traces`` glob, ``mode``, ``smooth``). Either may
    be empty, in which case search_reward defaults / a uniform prior apply.
    """

    action_space: ActionSpace
    costs: dict = field(default_factory=dict)
    prior: dict = field(default_factory=dict)


def load_for_level(level: int) -> SearchLevelConfig:
    """Load the per-level search config, falling back to the RL baseline if absent.

    The fallback keeps every level runnable: levels without a tuned file just
    search the full baseline action set with default penalties, exactly as
    before this fork existed.

    The level YAML carries both the action table (``skip`` + ``actions``, parsed
    by :class:`ActionSpace`) and an optional ``costs:`` block of per-button hold
    penalties, keyed by the action-table button nicknames (every button but
    ``R`` may be charged a small tie-break cost)::

        costs:
          F: -0.02   # fire
          J: -0.02   # jump
          U: -0.02   # up
          D: -0.02   # down
          L: -0.02   # left

    and an optional ``prior:`` block telling the sampler which traces to build the
    action prior from, on the fly (``ActionSampler.for_level``)::

        prior:
          traces: "contra/human_recordings/Level1/*.npz"  # glob (relative to CWD)
          mode: bigram      # bigram = P(next|prev) ; unigram = marginal P(action)
          smooth: 0.0       # blend toward uniform in [0, 1]
    """
    path = os.path.join(CONFIG_DIR, f"level{level}.yaml")
    if not os.path.exists(path):
        return SearchLevelConfig(action_space=BASELINE)
    with open(path) as f:
        raw = yaml.safe_load(f)
    costs = raw.get("costs", {}) or {}
    unknown = sorted(set(costs) - ALLOWED_COST_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in costs: of level{level}.yaml: {unknown}; "
            f"allowed: {sorted(ALLOWED_COST_KEYS)}"
        )
    prior = raw.get("prior", {}) or {}
    unknown_p = sorted(set(prior) - ALLOWED_PRIOR_KEYS)
    if unknown_p:
        raise ValueError(
            f"Unknown key(s) in prior: of level{level}.yaml: {unknown_p}; "
            f"allowed: {sorted(ALLOWED_PRIOR_KEYS)}"
        )
    return SearchLevelConfig(action_space=ActionSpace.from_dict(raw), costs=costs, prior=prior)
