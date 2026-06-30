"""ActionSampler — per-level action prior + state-masked random rollouts.

Owns everything ``mc_search`` needs to *propose* actions during one search:

  * the level's action table (and a fast prev-action → row lookup),
  * the action **prior** for that level (or a uniform fallback) as a row-stochastic
    PMF, *built on the fly* from the trace set named in the level YAML's ``prior:``
    block (``build_action_bigram.build_prior``) — bigram P(next|prev) or unigram
    P(action) — so it is always fresh against the current action table,
  * masked sampling from that prior against the stateful legal-action mask
    (``action_mask.legal_mask``), and
  * the random **rollout** the Monte-Carlo lookahead scores.

It also carries the ``search_reward.RewardConfig`` used to score a rollout, so a
single object crosses into each worker process. Built once per search by
:meth:`ActionSampler.for_level` and rebuilt identically inside every worker (it
derives purely from on-disk config), so it never has to be pickled across the
process boundary.
"""

import glob

import numpy as np

from contra.replay import rewind_state, step_env, SKIP as REPLAY_SKIP
from contra.events import EV_PLAYER_DIE
from synthetic.action_mask import legal_mask
from synthetic.action_configs.search_action_space import load_for_level
from synthetic.action_configs.search_reward import compute_reward, DEFAULT_CONFIG
from synthetic.build_action_bigram import build_prior


def _uniform_pmf(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / n, dtype=np.float32)


# ── Sampler ───────────────────────────────────────────────────────────────────

class ActionSampler:
    """Action prior + state-masked random rollout generator for one level."""

    def __init__(self, level: int, actions: np.ndarray, names: tuple,
                 reward_config: object, prior_pmf: np.ndarray, uniform_pmf: np.ndarray):
        self.level = level                  # the level this prior/action set is for
        self.actions = actions              # (N, 9) uint8 button vectors
        self.names = names                  # action labels, parallel to `actions`
        self.reward_config = reward_config  # search_reward.RewardConfig (scores rollouts)
        self.prior_pmf = prior_pmf          # (N, N) action prior for `level` (uniform if none)
        self.uniform_pmf = uniform_pmf      # used for any other level (game_clear crossing)
        self._index_by_bytes = {a.tobytes(): i for i, a in enumerate(actions)}

    @staticmethod
    def _level_config(level: int):
        """Return (SearchLevelConfig, actions, names, reward) for `level`."""
        cfg = load_for_level(level)
        actions = cfg.action_space.actions_np()
        if cfg.action_space.skip != REPLAY_SKIP:
            raise ValueError(
                f"action-space skip ({cfg.action_space.skip}) != replay.SKIP "
                f"({REPLAY_SKIP}); search frame-skip must match replay/step_env or "
                "traces won't reproduce."
            )
        reward = DEFAULT_CONFIG.with_costs(**cfg.costs)  # absent buttons → defaults
        return cfg, actions, tuple(cfg.action_space.names), reward

    @classmethod
    def _from_files(cls, level, actions, names, reward, files, mode, smooth):
        """Construct a sampler whose prior is counted from `files` (uniform if empty)."""
        uniform = _uniform_pmf(len(actions))
        if not files:
            return cls(level, actions, names, reward, uniform, uniform)
        _, _, prior = build_prior(level, files, mode=mode, smooth=smooth)
        return cls(level, actions, names, reward, prior, uniform)

    @classmethod
    def for_level(cls, level: int) -> "ActionSampler":
        """Build the prior on the fly from the level YAML's ``prior:`` block.

        The sampler counts action transitions in the configured ``traces`` glob
        every time (no prebuilt npz), so the prior is always fresh against the
        current action table — a table edit needs no rebuild. Omit the block (or
        match no files) to sample uniformly. ``mode`` (bigram/unigram) and
        ``smooth`` come from the same block.
        """
        cfg, actions, names, reward = cls._level_config(level)
        p = cfg.prior
        files = sorted(glob.glob(p["traces"])) if p.get("traces") else []
        if p.get("traces") and not files:
            print(f"WARNING: Level{level} prior traces {p['traces']!r} matched no "
                  "files; using uniform.")
        return cls._from_files(level, actions, names, reward, files,
                               p.get("mode", "bigram"), float(p.get("smooth", 0.0)))

    @classmethod
    def from_traces(cls, level: int, trace_glob: str, *, mode: str = "bigram",
                    smooth: float = 0.0) -> "ActionSampler":
        """Build the prior from an explicit trace glob — ad-hoc A/B of source/mode.

          * ``mode="bigram"``  — P(next | prev), sampling depends on the last action.
          * ``mode="unigram"`` — marginal P(action), previous action ignored (the
            prior matrix has identical rows, so the rollout is unchanged).

        Bypasses the level YAML's ``prior:`` block; ``for_level`` is the configured
        path.
        """
        _, actions, names, reward = cls._level_config(level)
        files = sorted(glob.glob(trace_glob))
        if not files:
            raise ValueError(f"no traces match {trace_glob!r}")
        return cls._from_files(level, actions, names, reward, files, mode, smooth)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def pmf(self, level: int) -> np.ndarray:
        # The prior is in this level's own action ordering, so only this level has
        # a matching prior; a game_clear run that crosses into another level
        # samples uniformly there.
        return self.prior_pmf if level == self.level else self.uniform_pmf

    def row_for(self, action: np.ndarray) -> int:
        """Bigram row index for a previously committed action (0 if unknown)."""
        return self._index_by_bytes.get(np.asarray(action, dtype=np.uint8).tobytes(), 0)

    @staticmethod
    def sample(pmf_row: np.ndarray, mask: np.ndarray, r: float) -> int:
        """Sample an action index from `pmf_row` restricted to `mask`, using r∈[0,1).

        The prior weight of illegal actions is zeroed and the remainder
        renormalised (the stateful step: ``action_mask.legal_mask`` decides what
        is legal). If no legal action carries prior mass, fall back to uniform
        over the legal set.
        """
        w = pmf_row * mask
        s = w.sum()
        if s <= 0.0:
            legal = np.flatnonzero(mask)
            return int(legal[min(int(r * len(legal)), len(legal) - 1)])
        return min(int(np.searchsorted(np.cumsum(w), r * s)), len(pmf_row) - 1)

    def rollout(self, env, start_state: bytes, length: int, level: int,
                prev_action: np.ndarray) -> tuple[list, float, bool]:
        """Sample one prior-guided, state-masked random rollout from `start_state`.

        At each step the current RAM + previous action decide which presses are
        meaningful (``legal_mask``); the prior row is restricted to that legal set
        before sampling, so structurally inert fire/jump presses are never
        emitted. `prev_action` is the action committed just before this rollout,
        needed for the fire/jump press-edge.

        Returns ``(actions, cumulative_reward, died)``. Stops early on death (that
        action is included but its reward is not).
        """
        rewind_state(env, start_state)
        pmf = self.pmf(level)
        actions = self.actions
        prev = self.row_for(prev_action)
        # Pre-sample all randoms at once (cheaper than per-step draws).
        rands = np.random.random(length).astype(np.float32)

        seq, total = [], 0.0
        for i in range(length):
            pre = env.unwrapped.get_ram().copy()
            mask = legal_mask(actions, pre, prev_action)
            prev = self.sample(pmf[prev], mask, rands[i])
            act = actions[prev].copy()
            step_env(env, act)
            cur = env.unwrapped.get_ram()
            seq.append(act)
            prev_action = act
            if EV_PLAYER_DIE.trigger(pre, cur):
                return seq, total, True
            total += compute_reward(pre, cur, self.reward_config, action=act)
        return seq, total, False
