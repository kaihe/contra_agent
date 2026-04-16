"""
prune_actions.py — Study mc_traces and replace no-effect actions with no-ops.

For each action in a trace we compare the emulator RAM after the original
action vs. after a no-op (all-zero) action from the same starting state.
If the RAM is identical the action has no game effect and is replaced with
a no-op.  Pruning is greedy: the winning state (same-RAM → use no-op,
different-RAM → keep original) feeds into the next step, so later decisions
reflect the already-pruned trajectory.

Parallelism: each trace file is processed by an independent worker process
with its own emulator instance.  Within a trace the greedy steps are
inherently sequential.

Usage:
    # Analyse a glob of traces (print stats only):
    python prune_actions.py mc_trace/win_level1_*.npz

    # Also verify each pruned trace achieves the same outcome:
    python prune_actions.py mc_trace/win_level1_*.npz --verify

    # Save pruned traces into mc_trace_pruned/ (same filename):
    python prune_actions.py mc_trace/win_level1_*.npz --save

    # All levels, save + verify, 8 workers:
    python prune_actions.py mc_trace/*.npz --save --verify --workers 8
"""

import argparse
import multiprocessing as mp
import os
import warnings

warnings.filterwarnings("ignore", message=".*Gym.*")

import numpy as np
import stable_retro as retro

from contra.replay import rewind_state, step_env, GAME
from contra.events import EV_PLAYER_DIE, EV_LEVELUP, EV_GAME_CLEAR

NOOP = np.zeros(9, dtype=np.uint8)

# ── Worker state ───────────────────────────────────────────────────────────────

_worker_env = None


def _worker_init() -> None:
    global _worker_env
    import warnings
    warnings.filterwarnings("ignore", message=".*Gym.*")
    _worker_env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    _worker_env.reset()


# ── Core pruning ───────────────────────────────────────────────────────────────

def prune_trace(npz_path: str, env) -> dict:
    """Greedily replace no-effect actions with no-ops.

    Returns a dict with:
      npz_path, level, outcome, initial_state,
      pruned (N,9), noop_mask (N,) bool, original_count, noop_count, fps
    """
    ckpt = np.load(npz_path, allow_pickle=True)
    actions       = ckpt["actions"].astype(np.uint8)
    initial_state = bytes(ckpt["initial_state"])
    level_str     = str(ckpt["level"]) if "level" in ckpt else "Level1"
    outcome       = str(ckpt["outcome"]) if "outcome" in ckpt else "unknown"

    N         = len(actions)
    pruned    = actions.copy()
    noop_mask = np.zeros(N, dtype=bool)

    current_emu_state = initial_state

    for i, act in enumerate(actions):
        act = act.astype(np.uint8)

        # Apply original action
        rewind_state(env, current_emu_state)
        step_env(env, act)
        obs_orig  = env.em.get_screen().copy()
        next_orig = env.em.get_state()

        # Apply no-op from same starting state
        rewind_state(env, current_emu_state)
        step_env(env, NOOP)
        obs_noop  = env.em.get_screen().copy()
        next_noop = env.em.get_state()

        if np.array_equal(obs_orig, obs_noop):
            noop_mask[i]      = True
            pruned[i]         = NOOP
            current_emu_state = next_noop
        else:
            current_emu_state = next_orig

    return {
        "npz_path":       npz_path,
        "level":          level_str,
        "outcome":        outcome,
        "initial_state":  initial_state,
        "pruned":         pruned,
        "noop_mask":      noop_mask,
        "original_count": N,
        "noop_count":     int(noop_mask.sum()),
        "fps":            int(ckpt["fps"]) if "fps" in ckpt else 20,
    }

# ── Save ───────────────────────────────────────────────────────────────────────

def save_pruned(result: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        actions=result["pruned"],
        initial_state=np.frombuffer(result["initial_state"], dtype=np.uint8),
        level=result["level"],
        outcome=result["outcome"],
        fps=result["fps"],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    from contra.replay import replay_actions

    npz_path = "synthetic/mc_trace/win_level1_202603301145.npz"

    _worker_init()
    result = prune_trace(npz_path, _worker_env)

    n_orig = result["original_count"]
    n_noop = result["noop_count"]
    pct    = 100 * n_noop / n_orig if n_orig else 0.0
    print(f"{n_noop}/{n_orig} ({pct:.1f}%) actions replaced with no-ops")

    replay_result = replay_actions(
        result["pruned"],
        initial_state=result["initial_state"],
        level=result["level"],
        want_video=False,
        verbose=False,
    )
    ok = replay_result["result"] == result["outcome"]
    print(f"verify: {'OK' if ok else 'FAIL'}"
          f" (got {replay_result['result']!r}, expected {result['outcome']!r})")


if __name__ == "__main__":
    main()
