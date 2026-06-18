"""
Monte Carlo Search with Backtracking
=================================================

Algorithm:
1. From the current committed state, generate N random rollouts.
2. Commit all actions from the best (highest reward) rollout.
3. If the best rollout still ends in death, force a rewind.
4. If reward stops improving for `patience` steps, rewind by a random
   amount between 1 and `max_rewind` steps.

Data layout:
  committed_actions[i] : action committed at step i
  states[i]            : emu_state after applying committed_actions[i]
  rewards[i]           : cumulative_reward after applying committed_actions[i]

Usage:
    python mc_search.py --level 1
"""

import argparse
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro
from contra.replay import rewind_state, step_env
from contra.action_space import DEFAULT as ACTION_SPACE
from contra.events import scan_events, get_level, EV_PLAYER_DIE, EV_GAME_CLEAR, ADDR_LEVEL_ROUTINE
from contra.reward import compute_reward, load as load_reward_config, DEFAULT_CONFIG

# Pruning is applied to every winning trace before it is saved. Support both
# invocation styles: `python -m synthetic.mc_search` / pytest (repo root on
# path) and `python synthetic/mc_search.py` (only synthetic/ on path).
try:
    from synthetic.prune_actions import prune_actions, verify_level_up
except ImportError:
    from prune_actions import prune_actions, verify_level_up

# Canonical action space shared with PPO (contra/action_space.py): a win path
# found here must be reproducible by the trained policy, so both use the same
# flat action-vector list and frame skip.
ACTION_NAMES = list(ACTION_SPACE.names)

_death_ev      = EV_PLAYER_DIE
_game_clear_ev = EV_GAME_CLEAR
_NOOP_ACTION   = np.zeros(9, dtype=np.uint8)

GAME = "Contra-Nes"
DEFAULT_STATE_BY_LEVEL = {i: f"Level{i}" for i in range(1, 9)}
STATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'contra', 'integration', 'Contra-Nes')
SKIP = ACTION_SPACE.skip  # frames per decision; shared with replay.step_env and PPO
TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp", "mc_trace")
VIDEO_DIR = os.path.join("tmp", "replay_videos")

# Reward config shared with PPO (contra/reward_configs/<name>.yaml). Set per
# search in _run_one_search and explicitly loaded by worker processes.
REWARD_CONFIG = DEFAULT_CONFIG


def _resolve_reward_config(level: int, name: str | None):
    """Load the named reward config, or default to the shared 'stable' config."""
    name = name or "stable"
    try:
        return load_reward_config(name)
    except FileNotFoundError:
        print(f"WARNING: reward config '{name}' not found, using default weights.")
        return DEFAULT_CONFIG


@dataclass
class State:
    emu_state: bytes
    done: bool = False

    def clone(self):
        return State(emu_state=self.emu_state, done=self.done)


_ACTIONS_NP = ACTION_SPACE.actions_np()        # (NUM_ACTIONS, 9) button vectors
NUM_ACTIONS = ACTION_SPACE.num_actions

# The canonical action space (contra/action_space.py) is already the curated
# set, so search enumerates every action. The TRIMMED_* plumbing is kept (it
# drives the fast bigram CDF sampler) but now spans all actions.
TRIMMED_ACTION_INDICES = np.arange(NUM_ACTIONS, dtype=np.int32)
NUM_TRIMMED = len(TRIMMED_ACTION_INDICES)  # == NUM_ACTIONS

_UNIFORM_PRIOR = np.full((NUM_ACTIONS, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
_ACTION_PRIORS: dict[int, np.ndarray] = {}  # level (1-indexed) → bigram prior

# Precomputed per-row trimmed sub-priors: shape (NUM_ACTIONS, NUM_TRIMMED).
# Row i gives the probability of each trimmed action given previous action i.
_TRIMMED_PRIORS: dict[int, np.ndarray] = {}  # level → (NUM_ACTIONS, NUM_TRIMMED)

# CDF of trimmed priors for fast sampling via searchsorted (avoids np.random.choice overhead).
_TRIMMED_CDFS: dict[int, np.ndarray] = {}   # level → (NUM_ACTIONS, NUM_TRIMMED) cumulative sums


def _build_trimmed_prior(full_prior: np.ndarray) -> np.ndarray:
    """Extract and renormalise searchable action columns from a prior matrix."""
    sub = full_prior[:, TRIMMED_ACTION_INDICES].astype(np.float64)
    row_sums = sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (sub / row_sums).astype(np.float32)


def _get_prior(level: int) -> np.ndarray:
    return _ACTION_PRIORS.get(level, _UNIFORM_PRIOR)


def _get_trimmed_prior(level: int) -> np.ndarray:
    if level not in _TRIMMED_PRIORS:
        _TRIMMED_PRIORS[level] = _build_trimmed_prior(_get_prior(level))
    return _TRIMMED_PRIORS[level]


def _get_trimmed_cdf(level: int) -> np.ndarray:
    if level not in _TRIMMED_CDFS:
        _TRIMMED_CDFS[level] = np.cumsum(_get_trimmed_prior(level), axis=1).astype(np.float32)
    return _TRIMMED_CDFS[level]


def _old_bigram_index(nes: np.ndarray) -> int:
    """Map a 9-button NES vector to the old 36-action bigram index.

    ``synthetic/action_bigram.npz`` was built before the action-space config was
    flattened. Its rows/columns are ordered as 9 d-pad states x 4 button states:
    [_, L, R, U, D, UL, UR, DL, DR] x [_, J, F, FJ].
    """
    up, down, left, right = bool(nes[4]), bool(nes[5]), bool(nes[6]), bool(nes[7])
    fire, jump = bool(nes[0]), bool(nes[8])

    if up and left:
        dpad = 5
    elif up and right:
        dpad = 6
    elif down and left:
        dpad = 7
    elif down and right:
        dpad = 8
    elif left:
        dpad = 1
    elif right:
        dpad = 2
    elif up:
        dpad = 3
    elif down:
        dpad = 4
    else:
        dpad = 0

    if fire and jump:
        buttons = 3
    elif jump:
        buttons = 1
    elif fire:
        buttons = 2
    else:
        buttons = 0

    return dpad * 4 + buttons


_OLD_BIGRAM_INDICES = np.array([_old_bigram_index(a) for a in _ACTIONS_NP], dtype=np.int32)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True).astype(np.float64)
    fallback = np.full_like(matrix, 1.0 / matrix.shape[1], dtype=np.float64)
    return np.where(row_sums > 0, matrix / row_sums, fallback).astype(np.float32)


def _map_bigram_to_action_space(prior: np.ndarray) -> np.ndarray | None:
    """Convert an old 36x36 bigram or current NxN prior into current action order."""
    if prior.shape == (NUM_ACTIONS, NUM_ACTIONS):
        return _normalise_rows(prior.astype(np.float64))
    if prior.shape == (36, 36):
        mapped = prior[np.ix_(_OLD_BIGRAM_INDICES, _OLD_BIGRAM_INDICES)]
        return _normalise_rows(mapped.astype(np.float64))
    return None


def _load_bigram(start_level: int) -> None:
    """Load level bigram priors, mapping legacy 36x36 matrices to current actions."""
    path = os.path.join(os.path.dirname(__file__), "action_bigram.npz")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, using uniform random actions.")
        return

    data = np.load(path)
    for level in range(start_level, 9):
        key = f"Level{level}"
        if key not in data:
            print(f"WARNING: key '{key}' not in {path}, using uniform for level {level}.")
            continue
        prior = _map_bigram_to_action_space(data[key])
        if prior is None:
            print(
                f"WARNING: bigram shape {data[key].shape} cannot map to "
                f"({NUM_ACTIONS},{NUM_ACTIONS}) for {key}, using uniform."
            )
            continue
        _ACTION_PRIORS[level] = prior
        _TRIMMED_PRIORS.pop(level, None)
        _TRIMMED_CDFS.pop(level, None)


# ── Parallel rollout worker ────────────────────────────────────────────────────

_worker_env = None

def _worker_init(game: str, state_label: str, use_spread: bool,
                 reward_config_name: str, start_level: int) -> None:
    global _worker_env, REWARD_CONFIG
    import warnings
    warnings.filterwarnings("ignore", message=".*Gym.*")
    REWARD_CONFIG = (
        DEFAULT_CONFIG
        if reward_config_name == DEFAULT_CONFIG.name
        else load_reward_config(reward_config_name)
    )
    _load_bigram(start_level)
    np.random.seed(os.getpid() % (2 ** 32))
    _worker_env = retro.make(
        game=game, state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,   # workers never use pixel obs; skip frame decode
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        _worker_env.load_state(f"spread_gun_state/{state_label}",
                               retro.data.Integrations.CUSTOM_ONLY)
    _worker_env.reset()


def _worker_rollout(args: tuple) -> tuple:
    emu_state, length, level = args
    return run_random_rollout(_worker_env, emu_state, length, level)


def run_random_rollout(env, start_emu_state: bytes, length: int, level: int = 1) -> tuple:
    """Returns (seq, cumulative_reward, died)."""
    rewind_state(env, start_emu_state)
    trimmed_cdf = _get_trimmed_cdf(level)
    # Pre-sample all random numbers for this rollout in one call (faster than per-step choice)
    rands = np.random.random(length + 1).astype(np.float32)
    prev_trimmed = int(np.searchsorted(trimmed_cdf[np.random.randint(NUM_ACTIONS)], rands[0]))
    prev_idx = TRIMMED_ACTION_INDICES[min(prev_trimmed, NUM_TRIMMED - 1)]
    cumulative_reward = 0.0
    seq = []
    for i in range(length):
        t = int(np.searchsorted(trimmed_cdf[prev_idx], rands[i + 1]))
        prev_idx = TRIMMED_ACTION_INDICES[min(t, NUM_TRIMMED - 1)]
        act = _ACTIONS_NP[prev_idx].copy()

        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, act)
        curr_ram = env.unwrapped.get_ram()

        if _death_ev.trigger(pre_ram, curr_ram):
            seq.append(act)
            return seq, cumulative_reward, True

        cumulative_reward += compute_reward(pre_ram, curr_ram, REWARD_CONFIG)
        seq.append(act)

    return seq, cumulative_reward, False


# ── Main search ────────────────────────────────────────────────────────────────

def search_and_play(env, initial_emu_state: bytes,
                    rollouts: int, rollout_len: int,
                    max_time: int,
                    level: int = 1,
                    max_rewind: int = 30,
                    max_actions: int = 4000,
                    goal: str = "level_up",
                    verbose=True,
                    pool=None):

    committed = State(emu_state=initial_emu_state)
    committed_actions = []   # [i]: action committed at step i
    states            = []   # [i]: emu_state after committed_actions[i]
    rewards           = []   # [i]: cumulative_reward after committed_actions[i]
    current_level     = level

    rollouts_high     = rollouts * 2
    current_rollouts  = rollouts
    t_start           = time.time()
    pending_events: list[str] = []
    total_sampled_actions = 0   # total actions sampled across all rollouts

    if verbose:
        print(f"\n  {'step':>4}  {'reward':>7}  {'death':>5}  {'rolls':>5}  {'time':>7}  event")
        print("  " + "-" * 58)

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            if verbose:
                print(f"\n  ⏱ Time budget exhausted ({max_time:.0f}s)")
            break
        if len(committed_actions) >= max_actions:
            if verbose:
                print(f"\n  ✂ Action limit reached ({max_actions}), abandoning trace")
            break
        if committed.done:
            if verbose:
                print(f"\n  🏆 WIN!  time={elapsed:.1f}s  steps={len(committed_actions)}")
            break

        # ── 0. No-op wait during level transition (routine 0x08/0x09) ───────
        rewind_state(env, committed.emu_state)
        ram_now = env.unwrapped.get_ram()
        if int(ram_now[ADDR_LEVEL_ROUTINE]) in (0x08, 0x09):
            pre_ram = ram_now.copy()
            step_env(env, _NOOP_ACTION)
            curr_ram = env.unwrapped.get_ram()
            committed.emu_state = env.em.get_state()
            new_level = get_level(curr_ram)
            if new_level != current_level:
                current_level = new_level
            if goal == "game_clear" and _game_clear_ev.trigger(pre_ram, curr_ram):
                committed.done = True
            elif goal == "level_up" and new_level != level:
                committed.done = True
            for ev in scan_events(pre_ram, curr_ram, len(committed_actions)):
                tag = ev['tag'] + (f"({ev['detail']})" if ev['detail'] else "")
                pending_events.append(tag)
            committed_actions.append(_NOOP_ACTION.copy())
            states.append(committed.emu_state)
            rewards.append(rewards[-1] if rewards else 0.0)
            continue

        # ── 1. Monte Carlo lookahead ──────────────────────────────────────────
        task = (committed.emu_state, rollout_len, current_level)
        if pool is not None:
            rollout_results = pool.map(_worker_rollout, [task] * current_rollouts)
        else:
            rollout_results = [run_random_rollout(env, committed.emu_state, rollout_len, current_level)
                               for _ in range(current_rollouts)]
            rewind_state(env, committed.emu_state)

        total_sampled_actions += sum(len(seq) for seq, _, _ in rollout_results)

        best_seq, best_rollout_reward, best_died = None, -float('inf'), True
        died_count = 0
        for seq, reward, died in rollout_results:
            if died:
                died_count += 1
            if reward > best_rollout_reward:
                best_rollout_reward, best_seq, best_died = reward, seq, died

        death_rate = died_count / current_rollouts

        # ── 2. Commit or rewind ───────────────────────────────────────────────
        if best_died:
            # All rollouts died — rewind and scale up rollouts
            n = len(committed_actions)
            if n > 0:
                rewind_back = np.random.randint(1, min(max_rewind, n) + 1)
                rewind_to   = n - rewind_back
            else:
                rewind_to = 0

            current_reward = rewards[-1] if rewards else 0.0
            current_rollouts = rollouts_high
            if verbose:
                ev_col = " ".join(pending_events) if pending_events else ""
                print(f"  {n:4d}  {current_reward:7.1f}  {death_rate:5.2f}  {current_rollouts:5d}  {elapsed:6.1f}s  {ev_col}⏪ →{rewind_to}")
                pending_events.clear()

            if rewind_to <= 0:
                committed.emu_state = initial_emu_state
                rewind_to = 0
            else:
                committed.emu_state = states[rewind_to - 1]

            rewind_state(env, committed.emu_state)
            committed_actions = committed_actions[:rewind_to]
            states            = states[:rewind_to]
            rewards           = rewards[:rewind_to]
            continue

        # Death rate recovered — reset rollouts to base
        if death_rate < 0.5 and current_rollouts == rollouts_high:
            current_rollouts = rollouts

        n = len(best_seq)
        commit_n = np.random.randint(n // 2, n + 1) if n >= 2 else n
        actions_to_commit = best_seq[:commit_n]

        rewind_state(env, committed.emu_state)
        for act in actions_to_commit:
            pre_ram = env.unwrapped.get_ram().copy()
            step_env(env, act)
            curr_ram = env.unwrapped.get_ram()

            if _death_ev.trigger(pre_ram, curr_ram):
                raise RuntimeError(f"Death during commit at step {len(committed_actions)}")

            committed.emu_state = env.em.get_state()
            new_level = get_level(curr_ram)
            if new_level != current_level:
                current_level = new_level
                if verbose:
                    pending_events.append(f"prior→Level{current_level}")
            if goal == "game_clear" and _game_clear_ev.trigger(pre_ram, curr_ram):
                committed.done = True
            elif goal == "level_up" and new_level != level:
                committed.done = True

            for ev in scan_events(pre_ram, curr_ram, len(committed_actions)):
                tag = ev['tag'] + (f"({ev['detail']})" if ev['detail'] else "")
                pending_events.append(tag)

            committed_actions.append(act)
            states.append(committed.emu_state)
            rewards.append((rewards[-1] if rewards else 0.0) + compute_reward(pre_ram, curr_ram, REWARD_CONFIG))

            if committed.done:
                break

        # ── 3. Progress log ───────────────────────────────────────────────────
        step_num      = len(committed_actions)
        prev_step_num = step_num - len(actions_to_commit)
        current_reward = rewards[-1] if rewards else 0.0
        if verbose and ((step_num // 10) > (prev_step_num // 10) or committed.done or pending_events):
            ev_col = " ".join(pending_events) if pending_events else ""
            print(f"  {step_num:4d}  {current_reward:7.1f}  {death_rate:5.2f}  {current_rollouts:5d}  {elapsed:6.1f}s  {ev_col}")
            pending_events.clear()

    return committed_actions, committed, rewards, total_sampled_actions


FPS = round(60 / SKIP)  # logical fps = 60 NES fps / SKIP


def save_trace(initial_state_for_npz: bytes, actions: list, trace_path: str,
               level: int = 1) -> None:
    """Save a winning trace to NPZ."""

    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    np.savez_compressed(trace_path,
        actions=np.array(actions, dtype=np.uint8),
        initial_state=np.frombuffer(initial_state_for_npz, dtype=np.uint8),
        level=f"Level{level}", outcome="win", fps=FPS
    )

    print(f"Trace saved to: {trace_path}")


def _run_one_search(level, rollouts, rollout_len, max_time, max_rewind, max_actions,
                    goal, workers, verbose=False, instance_id=None, reward_config=None):
    """Set up env+pool, run one full search, save trace if won. Returns trace path or None."""
    prefix = f"[i{instance_id}] " if instance_id is not None else ""

    # Resolve reward config in the parent and pass its name to workers so this
    # is robust under both fork and spawn multiprocessing modes.
    global REWARD_CONFIG
    REWARD_CONFIG = _resolve_reward_config(level, reward_config)
    _load_bigram(level)
    if instance_id is not None:
        np.random.seed((os.getpid() + instance_id * 1337) % (2**32))

    state_label = DEFAULT_STATE_BY_LEVEL[level]
    use_spread  = level > 1

    pool = mp.Pool(workers, initializer=_worker_init,
                   initargs=(GAME, state_label, use_spread, REWARD_CONFIG.name, level)) if workers > 1 else None
    env = retro.make(
        game=GAME, state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        env.load_state(f"spread_gun_state/{state_label}", retro.data.Integrations.CUSTOM_ONLY)
        if verbose:
            print(f"  Spread state: spread_gun_state/{state_label}.state")
    env.reset()
    initial_state_for_npz = env.em.get_state()
    initial_emu_state      = env.em.get_state()

    if prefix:
        print(f"{prefix}start  level={level}  workers={workers}", flush=True)

    actions, final_state, rewards, total_sampled_actions = search_and_play(
        env, initial_emu_state,
        rollouts=rollouts, rollout_len=rollout_len, max_time=max_time,
        level=level, max_rewind=max_rewind, max_actions=max_actions,
        goal=goal, verbose=verbose, pool=pool,
    )

    env.close()
    if pool:
        pool.close()
        pool.join()

    if verbose:
        print(f"\n{'=' * 70}\nRESULT\n{'=' * 70}")
        print(f"  Actions: {len(actions)}")
        print(f"  Reward:  {rewards[-1] if rewards else 0.0:.2f}")
        print(f"  Sampled: {total_sampled_actions}")

    if not final_state.done:
        if prefix:
            reward_str = f"{rewards[-1]:.1f}" if rewards else "0.0"
            print(f"{prefix}no win  steps={len(actions)}  reward={reward_str}", flush=True)
        return None

    # Prune ineffective button presses, then verify the pruned trace still
    # clears the level. The search env is already closed above, so prune/verify
    # each open and close their own env (one emulator per process).
    pruned = prune_actions(actions, initial_state_for_npz, verbose=verbose)
    if verify_level_up(pruned, initial_state_for_npz, label=f"{prefix}pruned", verbose=verbose):
        actions = pruned
    else:
        print(f"{prefix}prune verification failed — saving unpruned trace", flush=True)

    suffix     = f"_i{instance_id}" if instance_id is not None else ""
    date_str   = time.strftime("%Y%m%d%H%M%S" if instance_id is not None else "%Y%m%d%H%M")
    level_tag  = "game" if goal == "game_clear" else f"level{level}"
    trace_path = os.path.join(TRACE_DIR, f"level{level}", f"win_{level_tag}_{date_str}{suffix}.npz")
    save_trace(initial_state_for_npz, actions, trace_path, level=level)
    if prefix:
        reward_str = f"{rewards[-1]:.1f}" if rewards else "0.0"
        print(f"{prefix}WIN   steps={len(actions)}  reward={reward_str}  sampled={total_sampled_actions}  → {trace_path}", flush=True)
    return trace_path


def generate_traces(level, n, *, rollouts=64, rollout_len=48, max_time=600,
                    max_rewind=30, max_actions=6000, goal="level_up",
                    workers=None, reward_config=None, max_attempts=None):
    """Generate `n` winning traces for `level`, looping the search in one process.

    Each search opens and closes its own env+pool before the next starts (one
    emulator per process), and saves with a second-resolution, instance-suffixed
    filename so same-minute wins never overwrite. Stops after `n` wins or
    `max_attempts` searches (default 3*n). Returns the list of saved trace paths.
    """
    if workers is None:
        workers = os.cpu_count()
    if max_attempts is None:
        max_attempts = n * 3

    paths = []
    attempts = 0
    while len(paths) < n and attempts < max_attempts:
        path = _run_one_search(
            level=level, rollouts=rollouts, rollout_len=rollout_len,
            max_time=max_time, max_rewind=max_rewind, max_actions=max_actions,
            goal=goal, workers=workers, verbose=False,
            instance_id=attempts, reward_config=reward_config,
        )
        attempts += 1
        if path:
            paths.append(path)
            print(f"progress: {len(paths)}/{n} wins  (attempt {attempts})", flush=True)

    print(f"\nDone: {len(paths)}/{n} winning traces in {attempts} attempts.", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Playfun Monte Carlo Search")
    parser.add_argument("--level",       type=int, default=1, choices=list(range(1, 9)))
    parser.add_argument("--rollouts",    type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=48)
    parser.add_argument("--max-rewind",  type=int, default=30,
                        help="Max steps to rewind on backtrack (default: 30)")
    parser.add_argument("--max-time",    type=int, default=600)
    parser.add_argument("--workers",     type=int, default=os.cpu_count())
    parser.add_argument("--goal",        type=str, default="level_up",
                        choices=["level_up", "game_clear"],
                        help="level_up: stop on level-up (default); game_clear: stop on game clear")
    parser.add_argument("--max-actions", type=int, default=6000,
                        help="Abandon trace if committed actions exceed this limit (default: 6000)")
    parser.add_argument("--no-verbose", action="store_true", default=False,
                        help="Suppress per-step search output")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of winning traces to generate; loops the "
                             "search until this many wins are collected (default: 1)")
    parser.add_argument("--reward-config", type=str, default=None,
                        help="Reward config name under contra/reward_configs/ "
                             "(default: stable)")
    args = parser.parse_args()

    _load_bigram(args.level)
    np.random.seed(int(time.time() * 1000) % (2**32))

    verbose = not args.no_verbose
    if verbose:
        state_label = DEFAULT_STATE_BY_LEVEL[args.level]
        print("=" * 70)
        print("Playfun — Monte Carlo Search with Backtracking")
        print("=" * 70)
        print(f"  Game:           {GAME}")
        print(f"  Level:          {args.level}  ({state_label})")
        print(f"  Reward Config:  {args.reward_config or 'stable'}")
        print(f"  Skip:           {SKIP}")
        print(f"  Rollouts/Step:  {args.rollouts}")
        print(f"  Rollout Length: {args.rollout_len} actions ({args.rollout_len * SKIP} frames)")
        print(f"  Max Rewind:     {args.max_rewind} steps")
        print(f"  Workers:        {args.workers if args.workers > 1 else 1}")
        print(f"  Goal:           {args.goal}")
        print(f"  Max Actions:    {args.max_actions}")
        print(f"  Time Budget:    {args.max_time}s")
        print("=" * 70)

    if args.runs <= 1:
        _run_one_search(
            level=args.level, rollouts=args.rollouts, rollout_len=args.rollout_len,
            max_time=args.max_time, max_rewind=args.max_rewind, max_actions=args.max_actions,
            goal=args.goal, workers=args.workers, verbose=verbose,
            reward_config=args.reward_config,
        )
        return

    # Multi-run: each search logs concise "[iN] ..." lines, so per-step output
    # is suppressed inside generate_traces.
    generate_traces(
        args.level, args.runs,
        rollouts=args.rollouts, rollout_len=args.rollout_len, max_time=args.max_time,
        max_rewind=args.max_rewind, max_actions=args.max_actions, goal=args.goal,
        workers=args.workers, reward_config=args.reward_config,
    )


if __name__ == "__main__":
    main()
