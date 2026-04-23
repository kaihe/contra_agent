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
from contra.inputs import DPAD_TABLE, BUTTON_TABLE, NUM_DPAD, NUM_BUTTONS
from contra.events import compute_reward, scan_events, get_level, EV_PLAYER_DIE, EV_GAME_CLEAR, ADDR_LEVEL_ROUTINE

_death_ev      = EV_PLAYER_DIE
_game_clear_ev = EV_GAME_CLEAR
_NOOP_ACTION   = np.zeros(9, dtype=np.uint8)

GAME = "Contra-Nes"
DEFAULT_STATE_BY_LEVEL = {i: f"Level{i}" for i in range(1, 9)}
STATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'contra', 'integration', 'Contra-Nes')
SKIP = 3
TRACE_DIR = os.path.join(os.path.dirname(__file__), "mc_trace")
VIDEO_DIR = os.path.join("tmp", "replay_videos")


@dataclass
class State:
    emu_state: bytes
    done: bool = False

    def clone(self):
        return State(emu_state=self.emu_state, done=self.done)


_DPAD_NP   = np.array(DPAD_TABLE,   dtype=np.uint8)
_BUTTON_NP = np.array(BUTTON_TABLE, dtype=np.uint8)
NUM_ACTIONS = NUM_DPAD * NUM_BUTTONS  # 28

# Trimmed action space: drop diagonal d-pad (DL, DR) and Fire+Jump combo.
# Dpad  : _, L, R, U, D, UL, UR  (indices 0-6; drop DL=7, DR=8)
# Button: _, J, F                 (indices 0-2; drop FJ=3)
# Total : 7 × 3 = 21 actions
_TRIMMED_DPAD    = [0, 1, 2, 3, 4, 5, 6]
_TRIMMED_BUTTONS = [0, 1, 2]
TRIMMED_ACTION_INDICES = np.array(
    [d * NUM_BUTTONS + b for d in _TRIMMED_DPAD for b in _TRIMMED_BUTTONS],
    dtype=np.int32,
)
NUM_TRIMMED = len(TRIMMED_ACTION_INDICES)  # 21

_UNIFORM_PRIOR = np.full((NUM_ACTIONS, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
_ACTION_PRIORS: dict[int, np.ndarray] = {}  # level (1-indexed) → bigram prior

# Precomputed per-row trimmed sub-priors: shape (NUM_ACTIONS, NUM_TRIMMED).
# Row i gives the probability of each trimmed action given previous action i.
_TRIMMED_PRIORS: dict[int, np.ndarray] = {}  # level → (NUM_ACTIONS, NUM_TRIMMED)

# CDF of trimmed priors for fast sampling via searchsorted (avoids np.random.choice overhead).
_TRIMMED_CDFS: dict[int, np.ndarray] = {}   # level → (NUM_ACTIONS, NUM_TRIMMED) cumulative sums


def _build_trimmed_prior(full_prior: np.ndarray) -> np.ndarray:
    """Extract and renormalise trimmed columns from a full (28,28) prior."""
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


# ── Parallel rollout worker ────────────────────────────────────────────────────

_worker_env = None

def _worker_init(game: str, state_label: str, use_spread: bool) -> None:
    global _worker_env
    import warnings
    warnings.filterwarnings("ignore", message=".*Gym.*")
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


def _load_bigram(start_level: int) -> None:
    """Load bigram priors for all levels >= start_level into _ACTION_PRIORS."""
    path = os.path.join(os.path.dirname(__file__), "action_bigram.npz")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, using uniform random actions.")
        return
    data = np.load(path)
    for level in range(start_level, 9):
        key = f"Level{level}"
        if key in data:
            prior = data[key]
            if prior.shape == (NUM_ACTIONS, NUM_ACTIONS):
                _ACTION_PRIORS[level] = prior
                _TRIMMED_PRIORS.pop(level, None)  # invalidate cached trimmed prior + CDF
                _TRIMMED_CDFS.pop(level, None)
                # print(f"Loaded action bigram prior for {key}")
            else:
                print(f"WARNING: bigram shape {prior.shape} != ({NUM_ACTIONS},{NUM_ACTIONS}) for {key}, using uniform.")
        else:
            print(f"WARNING: key '{key}' not in {path}, using uniform for level {level}.")


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
        act = (_DPAD_NP[prev_idx // NUM_BUTTONS] | _BUTTON_NP[prev_idx % NUM_BUTTONS]).copy()

        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, act)
        curr_ram = env.unwrapped.get_ram()

        if _death_ev.trigger(pre_ram, curr_ram):
            seq.append(act)
            return seq, cumulative_reward, True

        cumulative_reward += compute_reward(pre_ram, curr_ram)
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

    rollouts_high     = rollouts * 4
    current_rollouts  = rollouts
    t_start           = time.time()
    pending_events: list[str] = []

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
            rewards.append((rewards[-1] if rewards else 0.0) + compute_reward(pre_ram, curr_ram))

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

    return committed_actions, committed, rewards


FPS = 20  # logical fps = 60 NES fps / SKIP


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
                    goal, workers, verbose=False, instance_id=None):
    """Set up env+pool, run one full search, save trace if won. Returns trace path or None."""
    prefix = f"[i{instance_id}] " if instance_id is not None else ""

    _load_bigram(level)
    if instance_id is not None:
        np.random.seed((os.getpid() + instance_id * 1337) % (2**32))

    state_label = DEFAULT_STATE_BY_LEVEL[level]
    use_spread  = level > 1

    pool = mp.Pool(workers, initializer=_worker_init,
                   initargs=(GAME, state_label, use_spread)) if workers > 1 else None
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

    actions, final_state, rewards = search_and_play(
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

    if not final_state.done:
        if prefix:
            reward_str = f"{rewards[-1]:.1f}" if rewards else "0.0"
            print(f"{prefix}no win  steps={len(actions)}  reward={reward_str}", flush=True)
        return None

    suffix     = f"_i{instance_id}" if instance_id is not None else ""
    date_str   = time.strftime("%Y%m%d%H%M%S" if instance_id is not None else "%Y%m%d%H%M")
    level_tag  = "game" if goal == "game_clear" else f"level{level}"
    trace_path = os.path.join(TRACE_DIR, f"win_{level_tag}_{date_str}{suffix}.npz")
    save_trace(initial_state_for_npz, actions, trace_path, level=level)
    if prefix:
        reward_str = f"{rewards[-1]:.1f}" if rewards else "0.0"
        print(f"{prefix}WIN   steps={len(actions)}  reward={reward_str}  → {trace_path}", flush=True)
    return trace_path


def main():
    parser = argparse.ArgumentParser(description="Playfun Monte Carlo Search")
    parser.add_argument("--level",       type=int, default=1, choices=list(range(1, 9)))
    parser.add_argument("--rollouts",    type=int, default=512)
    parser.add_argument("--rollout-len", type=int, default=48)
    parser.add_argument("--max-rewind",  type=int, default=30,
                        help="Max steps to rewind on backtrack (default: 30)")
    parser.add_argument("--max-time",    type=int, default=600)
    parser.add_argument("--workers",     type=int, default=os.cpu_count())
    parser.add_argument("--goal",        type=str, default="level_up",
                        choices=["level_up", "game_clear"],
                        help="level_up: stop on level-up (default); game_clear: stop on game clear")
    parser.add_argument("--max-actions", type=int, default=4000,
                        help="Abandon trace if committed actions exceed this limit (default: 4000)")
    parser.add_argument("--no-verbose", action="store_true", default=False,
                        help="Suppress per-step search output")
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
        print(f"  Skip:           {SKIP}")
        print(f"  Rollouts/Step:  {args.rollouts}")
        print(f"  Rollout Length: {args.rollout_len} actions ({args.rollout_len * SKIP} frames)")
        print(f"  Max Rewind:     {args.max_rewind} steps")
        print(f"  Workers:        {args.workers if args.workers > 1 else 1}")
        print(f"  Goal:           {args.goal}")
        print(f"  Max Actions:    {args.max_actions}")
        print(f"  Time Budget:    {args.max_time}s")
        print("=" * 70)

    _run_one_search(
        level=args.level, rollouts=args.rollouts, rollout_len=args.rollout_len,
        max_time=args.max_time, max_rewind=args.max_rewind, max_actions=args.max_actions,
        goal=args.goal, workers=args.workers, verbose=verbose,
    )


if __name__ == "__main__":
    main()
