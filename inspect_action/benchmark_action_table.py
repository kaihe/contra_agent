"""
Action Table Benchmark
========================

Compares action tables × skip values using Monte Carlo search across two experiments:

  Exp 1 – Navigation : from Level1 start, how far does the search reach
           within a fixed rollout action budget?

  Exp 2 – Boss Fight : from the boss state, how much boss HP is removed
           within a fixed rollout action budget?

Budget is measured as total rollout actions executed (rollouts × rollout_len
per MC iteration) — the actual computational cost of the search, independent
of how many steps get committed.

Usage:
    python benchmark_action_table.py
    python benchmark_action_table.py --trials 5 --nav-budget 100000 --boss-budget 400000
"""

import argparse
import gzip
import os
import sys
import time

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import stable_retro as retro

sys.path.insert(0, os.path.dirname(__file__))
from monte_carlo_playfun import State, get_boss_hit_sum, search_and_play

GAME       = "Contra-Nes"
STATES_DIR = os.path.join(os.path.dirname(__file__), "..", "main", "states")

NAV_STATE_FILE  = os.path.join(STATES_DIR, "Level1_x0_step1.state")
BOSS_STATE_FILE = os.path.join(STATES_DIR, "Level1_x3048_step921.state")


# =============================================================================
# ACTION TABLES
# =============================================================================

OLD_TABLE = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # F
    [1, 0, 0, 0, 0, 0, 1, 0, 0],  # LF
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # RF
    [1, 0, 0, 0, 1, 0, 0, 0, 0],  # UF
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # DF
    [1, 0, 0, 0, 0, 0, 1, 0, 1],  # LJF
    [1, 0, 0, 0, 0, 0, 0, 1, 1],  # RJF
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # JF
]
OLD_NAMES = ["F", "LF", "RF", "UF", "DF", "LJF", "RJF", "JF"]

NEW_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # R     (31.6% human freq)
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # RF    (15.6%)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # D     (7.4%)
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # RJ    (6.1%)
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # DF    (5.6%)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # F     (5.2%)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # L     (3.2%)
]
NEW_NAMES = ["R", "RF", "D", "RJ", "DF", "F", "L"]

# Each config: (label, action_table, action_names, skip)
CONFIGS = [
    ("Old skip=4", OLD_TABLE, OLD_NAMES, 4),
    ("New skip=4", NEW_TABLE, NEW_NAMES, 4),
    ("Old skip=8", OLD_TABLE, OLD_NAMES, 8),
    ("New skip=8", NEW_TABLE, NEW_NAMES, 8),
]


# =============================================================================
# STATE LOADING
# =============================================================================

def load_state_file(path: str, env) -> tuple[bytes, dict]:
    with gzip.open(path, "rb") as f:
        data = f.read()
    env.em.set_state(data)
    env.data.update_ram()
    env.step([0] * 9)
    return env.em.get_state(), env.data.lookup_all()


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_trial(env, state_file: str, action_table: list, action_names: list,
              skip: int, rollouts: int, rollout_len: int, commit_steps: int,
              patience: int, rollout_budget: int) -> dict:
    """Run one MC search trial budgeted by total rollout actions. Returns metrics."""
    t0 = time.time()
    emu_state, info = load_state_file(state_file, env)
    initial_boss_hp = get_boss_hit_sum(env)

    committed_actions, final_state, total_rollout_evals = search_and_play(
        env, emu_state, info,
        rollouts=rollouts,
        rollout_len=rollout_len,
        commit_steps=commit_steps,
        patience=patience,
        max_steps=999_999,      # no committed-step cap; rollout_budget controls stopping
        max_time=999_999,       # no wall-clock cap
        action_table=action_table,
        action_names=action_names,
        verbose=False,
        rollout_budget=rollout_budget,
        skip=skip,
    )

    win = final_state.done and final_state.lives > 0
    return {
        "committed_steps":    len(committed_actions),
        "rollout_evals":      total_rollout_evals,
        "steps_to_win":       len(committed_actions) if win else None,
        "win":                win,
        "max_xscroll":        final_state.max_x_reached,
        "reward":             final_state.cumulative_reward,
        "boss_hp_removed":    max(0, initial_boss_hp - final_state.boss_hit_sum),
        "initial_boss_hp":    initial_boss_hp,
        "elapsed":            time.time() - t0,
    }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(label: str, state_file: str, env,
                   rollouts: int, rollout_len: int,
                   commit_steps: int, patience: int, rollout_budget: int,
                   n_trials: int) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  rollouts={rollouts}  rollout_len={rollout_len}  "
          f"commit={commit_steps}  patience={patience}  "
          f"rollout_budget={rollout_budget:,}  trials={n_trials}")
    print(f"{'=' * 70}")

    all_results = {}
    for cfg_label, table, names, skip in CONFIGS:
        print(f"\n  ── {cfg_label}  {names} ──")
        trials = []
        for trial_idx in range(n_trials):
            np.random.seed(trial_idx * 17 + 3)
            res = run_trial(
                env, state_file, table, names, skip,
                rollouts=rollouts, rollout_len=rollout_len,
                commit_steps=commit_steps, patience=patience,
                rollout_budget=rollout_budget,
            )
            trials.append(res)
            win_str = "WIN" if res["win"] else "---"
            print(f"    trial {trial_idx+1}  {win_str}  "
                  f"committed={res['committed_steps']:4d}  "
                  f"evals={res['rollout_evals']:7,}  "
                  f"xscroll={res['max_xscroll']:4d}  "
                  f"boss_hp={res['boss_hp_removed']:3d}/{res['initial_boss_hp']}  "
                  f"reward={res['reward']:7.1f}  "
                  f"t={res['elapsed']:.1f}s")
        all_results[cfg_label] = trials
    return all_results


def print_summary(label: str, results: dict, primary_metric: str):
    print(f"\n{'─' * 70}")
    print(f"  SUMMARY — {label}   (primary: {primary_metric})")
    print(f"{'─' * 70}")
    header = f"  {'Config':<16} {'metric (mean±std)':>18}  {'wins':>5}  {'reward mean':>12}"
    print(header)
    print("  " + "-" * 56)
    for name, trials in results.items():
        vals = [t[primary_metric] for t in trials]
        wins = sum(t["win"] for t in trials)
        rwds = [t["reward"] for t in trials]
        m, s = np.mean(vals), np.std(vals)
        print(f"  {name:<16}  {m:>8.1f} ± {s:<6.1f}  "
              f"{wins:>3}/{len(trials)}  {np.mean(rwds):>10.1f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",      type=int, default=5)
    parser.add_argument("--nav-budget",  type=int, default=100_000,
                        help="Total rollout actions for navigation experiment")
    parser.add_argument("--boss-budget", type=int, default=100_000,
                        help="Total rollout actions for boss experiment")
    args = parser.parse_args()

    env = retro.make(
        game=GAME, state="Level1",
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env.reset()

    # ── Experiment 1: Navigation ───────────────────────────────────────────
    nav_results = run_experiment(
        label          = "Exp 1 — Navigation: max xscroll from Level1 start",
        state_file     = NAV_STATE_FILE,
        env            = env,
        rollouts       = 64,
        rollout_len    = 16,
        commit_steps   = 4,
        patience       = 8,
        rollout_budget = args.nav_budget,
        n_trials       = args.trials,
    )
    print_summary("Navigation", nav_results, primary_metric="max_xscroll")

    # # ── Experiment 2: Boss Fight ───────────────────────────────────────────
    # boss_results = run_experiment(
    #     label          = "Exp 2 — Boss Fight: boss HP removed within budget",
    #     state_file     = BOSS_STATE_FILE,
    #     env            = env,
    #     rollouts       = 256,
    #     rollout_len    = 16,
    #     commit_steps   = 8,
    #     patience       = 8,
    #     rollout_budget = args.boss_budget,
    #     n_trials       = args.trials,
    # )
    # print_summary("Boss Fight", boss_results, primary_metric="boss_hp_removed")

    env.close()


if __name__ == "__main__":
    main()
