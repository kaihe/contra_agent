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
        ram_orig  = env.unwrapped.get_ram().copy()
        next_orig = env.em.get_state()

        # Apply no-op from same starting state
        rewind_state(env, current_emu_state)
        step_env(env, NOOP)
        ram_noop  = env.unwrapped.get_ram().copy()
        next_noop = env.em.get_state()

        if np.array_equal(ram_orig, ram_noop):
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


# ── Verification ───────────────────────────────────────────────────────────────

def verify_trace(result: dict, env) -> str:
    """Replay the pruned trace and return 'game_clear' | 'level_up' | 'lose'."""
    rewind_state(env, result["initial_state"])
    leveled_up   = False
    game_cleared = False
    for act in result["pruned"]:
        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, act)
        curr_ram = env.unwrapped.get_ram()
        if EV_PLAYER_DIE.trigger(pre_ram, curr_ram):
            return "lose"
        if EV_LEVELUP.trigger(pre_ram, curr_ram):
            leveled_up = True
        if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
            game_cleared = True
    if game_cleared:
        return "game_clear"
    if leveled_up:
        return "level_up"
    return "lose"


def _outcomes_match(orig: str, pruned_result: str) -> bool:
    WIN = {"win", "level_up", "game_clear"}
    if orig in WIN and pruned_result in WIN:
        return True
    return orig == pruned_result


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


# ── Worker task (runs in a subprocess) ────────────────────────────────────────

def _worker_task(args: tuple) -> dict:
    """Prune one trace (and optionally verify + save).  Returns a stats dict."""
    npz_path, do_verify, do_save, out_dir = args
    env = _worker_env

    try:
        result = prune_trace(npz_path, env)
    except Exception as exc:
        return {"npz_path": npz_path, "error": str(exc)}

    verify_status = None
    if do_verify:
        pruned_outcome = verify_trace(result, env)
        verify_status  = _outcomes_match(result["outcome"], pruned_outcome)
        if not verify_status:
            verify_status = f"FAIL(orig={result['outcome']} pruned={pruned_outcome})"

    if do_save:
        fname    = os.path.basename(npz_path)
        out_path = os.path.join(out_dir, fname)
        save_pruned(result, out_path)

    return {
        "npz_path":       npz_path,
        "original_count": result["original_count"],
        "noop_count":     result["noop_count"],
        "verify_status":  verify_status,
        "saved_to":       os.path.join(out_dir, os.path.basename(npz_path)) if do_save else None,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune no-effect actions from mc_traces by replacing them with no-ops"
    )
    parser.add_argument("traces",    nargs="+", help="Paths to .npz trace files")
    parser.add_argument("--save",    action="store_true",
                        help="Save pruned traces to mc_trace_pruned/ with the same filename")
    parser.add_argument("--verify",  action="store_true",
                        help="Replay each pruned trace to confirm the original outcome is preserved")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"Number of parallel workers (default: {os.cpu_count()})")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress per-trace output")
    args = parser.parse_args()

    paths = sorted(p for p in args.traces if os.path.exists(p))
    skipped = len(args.traces) - len(paths)
    if skipped:
        print(f"  SKIP {skipped} missing file(s)")

    if not paths:
        print("No traces to process.")
        return

    # Determine output directory relative to the first trace's parent
    first_dir = os.path.dirname(os.path.abspath(paths[0]))
    out_dir   = os.path.normpath(os.path.join(first_dir, "..", "mc_trace_pruned"))

    tasks = [(p, args.verify, args.save, out_dir) for p in paths]

    total_orig   = 0
    total_noop   = 0
    verify_ok    = 0
    verify_fail  = 0

    workers = min(args.workers, len(paths))
    pool = mp.Pool(workers, initializer=_worker_init) if workers > 1 else None

    if pool is not None:
        it = pool.imap_unordered(_worker_task, tasks)
    else:
        _worker_init()
        it = (_worker_task(t) for t in tasks)

    for r in it:
        if "error" in r:
            print(f"  ERROR {os.path.basename(r['npz_path'])}: {r['error']}")
            continue

        N    = r["original_count"]
        noop = r["noop_count"]
        pct  = 100.0 * noop / N if N else 0.0
        total_orig += N
        total_noop += noop

        if not args.quiet:
            line = (f"  {os.path.basename(r['npz_path']):50s}  "
                    f"{N:4d} actions  {noop:4d} no-op ({pct:5.1f}%)  "
                    f"{N - noop:4d} effective")
            if r["verify_status"] is not None:
                v = r["verify_status"]
                line += f"  verify: {'OK' if v is True else v}"
            if r["saved_to"]:
                line += f"  → {os.path.basename(r['saved_to'])}"
            print(line)

        if r["verify_status"] is not None:
            if r["verify_status"] is True:
                verify_ok += 1
            else:
                verify_fail += 1

    if pool is not None:
        pool.close()
        pool.join()

    print()
    if total_orig:
        pct = 100.0 * total_noop / total_orig
        print(f"Summary: {total_orig} total actions  →  "
              f"{total_noop} no-ops ({pct:.1f}% pruned)  "
              f"{total_orig - total_noop} effective")
    if args.verify:
        print(f"Verify:  {verify_ok} OK  {verify_fail} FAILED")
    if args.save:
        print(f"Output:  {out_dir}/")


if __name__ == "__main__":
    main()
