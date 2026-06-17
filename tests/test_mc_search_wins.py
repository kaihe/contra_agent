"""Acceptance tests: mc_search must find a win path for every level at the
shipped default budget, using the canonical action space (baseline.yaml) and the
shared `stable` reward config.

These are slow, stochastic, opt-in tests — they actually run the Monte Carlo
search. Run them explicitly:

    pytest -m slow tests/test_mc_search_wins.py
    pytest -m slow tests/test_mc_search_wins.py -k level1   # one level

Each level is searched with `goal=level_up` (stop at the level transition), which
keeps per-level runtime to roughly a minute on a many-core machine while still
exercising the full search → win-trace pipeline. Trace files are redirected to a
temp dir so the run does not litter `synthetic/mc_trace/`.
"""

import os

import pytest

from synthetic import mc_search

# Mirror the argparse defaults in mc_search.main() — this is the "default budget".
DEFAULT_BUDGET = dict(
    rollouts=64,
    rollout_len=48,
    max_time=600,
    max_rewind=30,
    max_actions=6000,
    goal="level_up",
    workers=os.cpu_count(),
)


@pytest.mark.slow
@pytest.mark.parametrize("level", range(1, 9))
def test_mc_search_wins_level(level, tmp_path, monkeypatch, capsys):
    # Keep win traces out of the repo (CLAUDE.md: artifacts → tmp/).
    monkeypatch.setattr(mc_search, "TRACE_DIR", str(tmp_path))

    # The search can run for up to max_time seconds per level, so stream its
    # per-step progress log live to the terminal instead of leaving the run
    # silent. capsys.disabled() bypasses pytest's stdout capture without needing
    # the -s flag. verbose=True turns on mc_search's progress table.
    with capsys.disabled():
        print(
            f"\n[level {level}] searching with default budget "
            f"(max_time={DEFAULT_BUDGET['max_time']}s, "
            f"rollouts={DEFAULT_BUDGET['rollouts']}, workers={DEFAULT_BUDGET['workers']})...",
            flush=True,
        )
        # reward_config=None → the shared "stable" config (_resolve_reward_config).
        trace_path = mc_search._run_one_search(
            level=level,
            reward_config=None,
            verbose=True,
            **DEFAULT_BUDGET,
        )

    assert trace_path is not None and os.path.exists(trace_path), (
        f"mc_search found no win for level {level} within "
        f"{DEFAULT_BUDGET['max_time']}s at the default budget"
    )
