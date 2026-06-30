"""
Monte Carlo search with backtracking (trace generation phase)
=============================================================

From a committed game state, sample many short random rollouts (biased by a
per-level action bigram), commit a prefix of the best-scoring one, and rewind
when every rollout dies. Repeat until the level is cleared.

This is the *generation* phase: it uses the forked search action space + reward
(``synthetic/action_configs/search_action_space.py``,
``synthetic/action_configs/search_reward.py``), which optimise for the cleanest
winning trace found efficiently — distinct from the RL phase's ``contra/*`` objective.

Two design choices keep the trace clean and the search honest:
  * fire/jump press penalties in the reward (no post-hoc pruning), and
  * a commit ``settle_margin``: never commit an action whose delayed reward
    (bullet → enemy-HP hit) falls outside the rollout window that scored it.

Layout of a search's committed history (parallel lists):
  actions[i] : action committed at step i
  states[i]  : emulator savestate after actions[i]
  rewards[i] : cumulative reward after actions[i]

Usage:
    python synthetic/mc_search.py --level 1
"""

import argparse
import multiprocessing as mp
import os
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro

from contra.replay import rewind_state, step_env, SKIP as REPLAY_SKIP
from contra.events import (scan_events, get_level, ADDR_LEVEL_ROUTINE,
                           EV_PLAYER_DIE, EV_GAME_CLEAR, EV_LEVEL_TRANSITION)
from synthetic.action_sampler import ActionSampler
from synthetic.action_configs.search_reward import compute_reward

# ── Constants ───────────────────────────────────────────────────────────────────

GAME = "Contra-Nes"
DEFAULT_STATE_BY_LEVEL = {i: f"Level{i}" for i in range(1, 9)}
TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp", "mc_trace")

SKIP = REPLAY_SKIP          # NES frames per decision; must match replay.step_env
_NOOP = np.zeros(9, dtype=np.uint8)


# Worker process state: an env + the (rebuilt) sampler, set once in _worker_init.
_worker_env = None
_worker_sampler: ActionSampler | None = None


def _worker_init(game: str, state_label: str, use_spread: bool, level: int) -> None:
    global _worker_env, _worker_sampler
    warnings.filterwarnings("ignore", message=".*Gym.*")
    _worker_sampler = ActionSampler.for_level(level)
    np.random.seed(os.getpid() % (2 ** 32))
    _worker_env = retro.make(
        game=game, state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,   # workers never decode frames
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        _worker_env.load_state(f"spread_gun_state/{state_label}",
                               retro.data.Integrations.CUSTOM_ONLY)
    _worker_env.reset()


def _worker_rollout(task: tuple) -> tuple:
    return _worker_sampler.rollout(_worker_env, *task)


# ── Search ──────────────────────────────────────────────────────────────────────

@dataclass
class State:
    emu_state: bytes
    done: bool = False


@dataclass
class SearchEffort:
    sampled_actions: int = 0
    search_wall_s: float = 0.0
    search_steps: int = 0
    final_reward: float = 0.0


class _Search:
    """One Monte-Carlo-with-backtracking search over a single committed history."""

    def __init__(self, env, sampler: ActionSampler, initial_state: bytes, *,
                 level: int, goal: str, rollouts: int, rollout_len: int,
                 max_time: int, max_rewind: int, max_actions: int,
                 settle_margin: int, pool, verbose: bool):
        self.env, self.sampler, self.initial = env, sampler, initial_state
        self.level, self.goal = level, goal
        self.base_rollouts, self.high_rollouts = rollouts, rollouts * 2
        self.rollouts = rollouts
        self.rollout_len, self.max_time = rollout_len, max_time
        self.max_rewind, self.max_actions = max_rewind, max_actions
        self.settle_margin, self.pool, self.verbose = settle_margin, pool, verbose

        self.state = State(emu_state=initial_state)
        self.cur_level = level
        self.actions: list = []     # committed actions
        self.states: list = []      # savestate after each committed action
        self.rewards: list = []     # cumulative reward after each committed action
        self.events: list = []      # pending event tags for the next log line
        self.sampled = 0            # total actions sampled across all rollouts
        self.t0 = time.time()

    # -- public entry --------------------------------------------------------------

    def run(self):
        self._header()
        while True:
            elapsed = time.time() - self.t0
            if elapsed > self.max_time:
                self._say(f"\n  ⏱ Time budget exhausted ({self.max_time:.0f}s)"); break
            if len(self.actions) >= self.max_actions:
                self._say(f"\n  ✂ Action limit reached ({self.max_actions}), abandoning trace"); break
            if self.state.done:
                self._say(f"\n  🏆 WIN!  time={elapsed:.1f}s  steps={len(self.actions)}"); break

            if self._advance_transition():
                continue
            best_seq, death_rate, died = self._lookahead()
            if died:
                self._rewind(death_rate, elapsed)
            else:
                self._commit(best_seq, death_rate, elapsed)

        effort = SearchEffort(self.sampled, time.time() - self.t0,
                              len(self.actions), self._reward)
        return self.actions, self.state, self.rewards, effort

    # -- phases --------------------------------------------------------------------

    def _advance_transition(self) -> bool:
        """Handle sitting in the post-boss routine; return True if it was handled.

        For level_up the level is already cleared, so we stop here — before the
        transition's xscroll reset injects a spurious negative reward. game_clear
        must no-op through the end-of-level sequence into the next level.
        """
        rewind_state(self.env, self.state.emu_state)
        ram = self.env.unwrapped.get_ram()
        # Post-boss end-of-level sequence (the routines EV_LEVEL_TRANSITION keys on).
        if int(ram[ADDR_LEVEL_ROUTINE]) not in (0x08, 0x09):
            return False
        if self.goal == "level_up":
            self.state.done = True
            return True
        pre = ram.copy()
        step_env(self.env, _NOOP)
        cur = self.env.unwrapped.get_ram()
        self.state.emu_state = self.env.em.get_state()
        self.cur_level = get_level(cur)
        if EV_GAME_CLEAR.trigger(pre, cur):
            self.state.done = True
        self._collect_events(pre, cur)
        self._append(_NOOP.copy(), self.state.emu_state, self._reward)
        return True

    def _lookahead(self) -> tuple[list, float, bool]:
        """Sample `rollouts` rollouts; return (best_seq, death_rate, best_died)."""
        prev_action = self.actions[-1] if self.actions else _NOOP
        task = (self.state.emu_state, self.rollout_len, self.cur_level, prev_action)
        if self.pool is not None:
            results = self.pool.map(_worker_rollout, [task] * self.rollouts)
        else:
            results = [self.sampler.rollout(self.env, *task) for _ in range(self.rollouts)]
            rewind_state(self.env, self.state.emu_state)

        self.sampled += sum(len(seq) for seq, _, _ in results)
        best_seq, best_reward, best_died, deaths = None, -float("inf"), True, 0
        for seq, reward, died in results:
            deaths += died
            if reward > best_reward:
                best_reward, best_seq, best_died = reward, seq, died
        return best_seq, deaths / self.rollouts, best_died

    def _rewind(self, death_rate: float, elapsed: float) -> None:
        """Every rollout died: roll back a random amount and widen the search."""
        n = len(self.actions)
        rewind_to = n - np.random.randint(1, min(self.max_rewind, n) + 1) if n > 0 else 0
        self.rollouts = self.high_rollouts
        self._log(n, self._reward, death_rate, elapsed, extra=f"⏪ →{rewind_to}")

        if rewind_to <= 0:
            self.state.emu_state, rewind_to = self.initial, 0
        else:
            self.state.emu_state = self.states[rewind_to - 1]
        rewind_state(self.env, self.state.emu_state)
        del self.actions[rewind_to:]
        del self.states[rewind_to:]
        del self.rewards[rewind_to:]

    def _commit(self, best_seq: list, death_rate: float, elapsed: float) -> None:
        """Replay and commit a prefix of the best rollout, stopping the prefix
        `settle_margin` steps short so every committed action's delayed reward was
        already scored by the rollout that selected it."""
        if death_rate < 0.5 and self.rollouts == self.high_rollouts:
            self.rollouts = self.base_rollouts

        cap = max(1, len(best_seq) - self.settle_margin)
        commit_n = np.random.randint((cap + 1) // 2, cap + 1) if cap >= 2 else cap

        rewind_state(self.env, self.state.emu_state)
        prev_steps = len(self.actions)
        for act in best_seq[:commit_n]:
            pre = self.env.unwrapped.get_ram().copy()
            step_env(self.env, act)
            cur = self.env.unwrapped.get_ram()
            if EV_PLAYER_DIE.trigger(pre, cur):
                raise RuntimeError(f"Death during commit at step {len(self.actions)}")

            self.state.emu_state = self.env.em.get_state()
            new_level = get_level(cur)
            if new_level != self.cur_level:
                self.cur_level = new_level
                if self.verbose:
                    self.events.append(f"prior→Level{new_level}")
            if self._reached_goal(pre, cur, new_level):
                self.state.done = True
            self._collect_events(pre, cur)
            self._append(act, self.state.emu_state,
                         self._reward + compute_reward(pre, cur, self.sampler.reward_config, action=act))
            if self.state.done:
                break

        step_num = len(self.actions)
        if self.verbose and ((step_num // 10) > (prev_steps // 10)
                             or self.state.done or self.events):
            self._log(step_num, self._reward, death_rate, elapsed)

    def _reached_goal(self, pre, cur, new_level: int) -> bool:
        """Win condition. level_up fires on the post-boss transition edge (the
        level is cleared there); `new_level != level` is a fallback for levels
        that increment without that routine."""
        if self.goal == "game_clear":
            return bool(EV_GAME_CLEAR.trigger(pre, cur))
        return bool(EV_LEVEL_TRANSITION.trigger(pre, cur)) or new_level != self.level

    # -- helpers -------------------------------------------------------------------

    @property
    def _reward(self) -> float:
        return self.rewards[-1] if self.rewards else 0.0

    def _append(self, action, emu_state, reward) -> None:
        self.actions.append(action)
        self.states.append(emu_state)
        self.rewards.append(reward)

    def _collect_events(self, pre, cur) -> None:
        if not self.verbose:
            return
        for ev in scan_events(pre, cur, len(self.actions)):
            self.events.append(ev['tag'] + (f"({ev['detail']})" if ev['detail'] else ""))

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _header(self) -> None:
        if self.verbose:
            print(f"\n  {'step':>4}  {'reward':>7}  {'death':>5}  {'rolls':>5}  {'time':>7}  event")
            print("  " + "-" * 58)

    def _log(self, n, reward, death_rate, elapsed, extra="") -> None:
        if not self.verbose:
            return
        ev_col = " ".join(self.events)
        print(f"  {n:4d}  {reward:7.1f}  {death_rate:5.2f}  {self.rollouts:5d}  {elapsed:6.1f}s  {ev_col}{extra}")
        self.events.clear()


def search_and_play(env, initial_emu_state: bytes, rollouts: int, rollout_len: int,
                    max_time: int, level: int = 1, max_rewind: int = 30,
                    max_actions: int = 4000, goal: str = "level_up",
                    settle_margin: int = 16, verbose: bool = True, pool=None,
                    sampler: ActionSampler | None = None):
    """Run one search to completion. Returns (actions, final_state, rewards, effort)."""
    return _Search(
        env, sampler or ActionSampler.for_level(level), initial_emu_state,
        level=level, goal=goal, rollouts=rollouts, rollout_len=rollout_len,
        max_time=max_time, max_rewind=max_rewind, max_actions=max_actions,
        settle_margin=settle_margin, pool=pool, verbose=verbose,
    ).run()


# ── Trace I/O ─────────────────────────────────────────────────────────────────────

def save_trace(initial_state_for_npz: bytes, actions: list, trace_path: str,
               level: int = 1, effort: SearchEffort | None = None,
               *, rollouts: int | None = None, rollout_len: int | None = None,
               max_time: int | None = None, max_rewind: int | None = None,
               max_actions: int | None = None, goal: str | None = None,
               workers: int | None = None, reward_config: str | None = None) -> None:
    """Save a winning trace (actions + start state + search metadata) to NPZ."""
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    effort = effort or SearchEffort(search_steps=len(actions))

    def _i32(v):
        return np.array(-1 if v is None else v, dtype=np.int32)

    np.savez_compressed(trace_path,
        actions=np.array(actions, dtype=np.uint8),
        initial_state=np.frombuffer(initial_state_for_npz, dtype=np.uint8),
        level=f"Level{level}", outcome="win", fps=round(60 / SKIP),
        skip=np.array(SKIP, dtype=np.int32),
        sampled_actions=np.array(effort.sampled_actions, dtype=np.int64),
        search_wall_s=np.array(effort.search_wall_s, dtype=np.float64),
        search_steps=np.array(effort.search_steps, dtype=np.int64),
        trace_steps=np.array(len(actions), dtype=np.int64),
        final_reward=np.array(effort.final_reward, dtype=np.float32),
        rollouts=_i32(rollouts), rollout_len=_i32(rollout_len),
        max_time=_i32(max_time), max_rewind=_i32(max_rewind), max_actions=_i32(max_actions),
        goal=np.array("" if goal is None else goal),
        workers=_i32(workers),
        reward_config=np.array("" if reward_config is None else reward_config),
    )
    print(f"Trace saved to: {trace_path}")


# ── Orchestration ─────────────────────────────────────────────────────────────────

def _run_one_search(level, rollouts, rollout_len, max_time, max_rewind, max_actions,
                    goal, workers, settle_margin=16, verbose=False, instance_id=None):
    """Set up env + pool, run one full search, save the trace if it wins.

    Returns the saved trace path, or None if no win. Each worker rebuilds the
    ActionSampler from the level, so the action space + reward stay consistent
    under both fork and spawn multiprocessing.
    """
    prefix = f"[i{instance_id}] " if instance_id is not None else ""
    sampler = ActionSampler.for_level(level)
    if instance_id is not None:
        np.random.seed((os.getpid() + instance_id * 1337) % (2 ** 32))

    state_label = DEFAULT_STATE_BY_LEVEL[level]
    use_spread = level > 1
    pool = (mp.Pool(workers, initializer=_worker_init,
                    initargs=(GAME, state_label, use_spread, level))
            if workers > 1 else None)
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
    initial_state = env.em.get_state()

    if prefix:
        print(f"{prefix}start  level={level}  workers={workers}", flush=True)

    actions, final_state, rewards, effort = search_and_play(
        env, initial_state, rollouts=rollouts, rollout_len=rollout_len,
        max_time=max_time, level=level, max_rewind=max_rewind, max_actions=max_actions,
        goal=goal, settle_margin=settle_margin, verbose=verbose, pool=pool, sampler=sampler,
    )

    env.close()
    if pool:
        pool.close()
        pool.join()

    reward_str = f"{rewards[-1]:.1f}" if rewards else "0.0"
    if verbose:
        print(f"\n{'=' * 70}\nRESULT\n{'=' * 70}")
        print(f"  Actions: {len(actions)}")
        print(f"  Reward:  {reward_str}")
        print(f"  Sampled: {effort.sampled_actions}")
        print(f"  Search:  {effort.search_wall_s:.1f}s")

    if not final_state.done:
        if prefix:
            print(f"{prefix}no win  steps={len(actions)}  reward={reward_str}", flush=True)
        return None

    # No post-hoc pruning: the fire/jump penalty in search_reward.py keeps the
    # trace clean during generation (prune_actions.py is now CLI-only).
    suffix = f"_i{instance_id}" if instance_id is not None else ""
    date_str = time.strftime("%Y%m%d%H%M%S" if instance_id is not None else "%Y%m%d%H%M")
    level_tag = "game" if goal == "game_clear" else f"level{level}"
    trace_path = os.path.join(TRACE_DIR, f"level{level}", f"win_{level_tag}_{date_str}{suffix}.npz")
    save_trace(
        initial_state, actions, trace_path, level=level, effort=effort,
        rollouts=rollouts, rollout_len=rollout_len, max_time=max_time,
        max_rewind=max_rewind, max_actions=max_actions, goal=goal,
        workers=workers, reward_config=sampler.reward_config.name,
    )
    if prefix:
        print(f"{prefix}WIN   steps={len(actions)}  reward={reward_str}  "
              f"sampled={effort.sampled_actions}  → {trace_path}", flush=True)
    return trace_path


def generate_traces(level, n, *, rollouts=64, rollout_len=48, max_time=600,
                    max_rewind=30, max_actions=6000, goal="level_up",
                    workers=None, settle_margin=16, max_attempts=None):
    """Loop the search in one process until `n` winning traces are collected.

    Each search opens/closes its own env+pool (one emulator per process) and
    saves with a second-resolution, instance-suffixed filename so same-minute
    wins never overwrite. Stops after `n` wins or `max_attempts` searches
    (default 3*n). Returns the saved trace paths.
    """
    workers = workers or os.cpu_count()
    max_attempts = max_attempts or n * 3

    paths, attempts = [], 0
    t_start = time.time()
    while len(paths) < n and attempts < max_attempts:
        t0 = time.time()
        path = _run_one_search(
            level=level, rollouts=rollouts, rollout_len=rollout_len,
            max_time=max_time, max_rewind=max_rewind, max_actions=max_actions,
            goal=goal, workers=workers, settle_margin=settle_margin,
            verbose=False, instance_id=attempts,
        )
        attempts += 1
        if path:
            paths.append(path)
        print(f"attempt {attempts}: {'WIN ' if path else 'no win'} in "
              f"{time.time() - t0:.1f}s  ({len(paths)}/{n} wins)", flush=True)

    print(f"\nDone: {len(paths)}/{n} winning traces in {attempts} attempts, "
          f"{time.time() - t_start:.1f}s total.", flush=True)
    return paths


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo search with backtracking")
    p.add_argument("--level", type=int, default=1, choices=list(range(1, 9)))
    p.add_argument("--rollouts", type=int, default=64)
    p.add_argument("--rollout-len", type=int, default=48)
    p.add_argument("--settle-margin", type=int, default=16,
                   help="Evaluated rollout steps reserved at the tail and never committed, "
                        "so a committed action's delayed reward is scored before commit. "
                        "Must be < rollout-len; rollout-len/2 commits only the first half; "
                        "0 commits up to the whole rollout (default: 16)")
    p.add_argument("--max-rewind", type=int, default=30,
                   help="Max steps to rewind on backtrack (default: 30)")
    p.add_argument("--max-actions", type=int, default=6000,
                   help="Abandon a trace exceeding this many committed actions (default: 6000)")
    p.add_argument("--max-time", type=int, default=600, help="Per-search time budget (s)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--goal", type=str, default="level_up", choices=["level_up", "game_clear"],
                   help="level_up: stop on level transition (default); game_clear: full clear")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of winning traces to collect; loops the search (default: 1)")
    p.add_argument("--no-verbose", action="store_true", help="Suppress per-step search output")
    return p.parse_args()


def _print_config(args, sampler: ActionSampler) -> None:
    w = sampler.reward_config.reward_weights
    commit_cap = max(1, args.rollout_len - args.settle_margin)
    print("=" * 70)
    print("Monte Carlo Search with Backtracking")
    print("=" * 70)
    print(f"  Game:           {GAME}")
    print(f"  Level:          {args.level}  ({DEFAULT_STATE_BY_LEVEL[args.level]})")
    print(f"  Action Space:   {sampler.num_actions} actions")
    btn_costs = " ".join(f"{n}={w[n]:g}" for n in ("F", "J", "U", "D", "L", "R"))
    print(f"  Button Costs:   {btn_costs}")
    print(f"  Skip:           {SKIP}")
    print(f"  Rollouts/Step:  {args.rollouts}")
    print(f"  Rollout Length: {args.rollout_len} actions ({args.rollout_len * SKIP} frames)")
    print(f"  Settle Margin:  {args.settle_margin} steps (commit ≤ {commit_cap})")
    print(f"  Max Rewind:     {args.max_rewind} steps")
    print(f"  Workers:        {max(1, args.workers)}")
    print(f"  Goal:           {args.goal}")
    print(f"  Max Actions:    {args.max_actions}")
    print(f"  Time Budget:    {args.max_time}s")
    print("=" * 70)


def main():
    args = _parse_args()
    np.random.seed(int(time.time() * 1000) % (2 ** 32))
    verbose = not args.no_verbose
    if verbose:
        _print_config(args, ActionSampler.for_level(args.level))

    common = dict(
        rollouts=args.rollouts, rollout_len=args.rollout_len, max_time=args.max_time,
        max_rewind=args.max_rewind, max_actions=args.max_actions, goal=args.goal,
        workers=args.workers, settle_margin=args.settle_margin,
    )
    if args.runs <= 1:
        _run_one_search(level=args.level, verbose=verbose, **common)
    else:
        # Multi-run logs concise "[iN] ..." lines, so per-step output is suppressed.
        generate_traces(args.level, args.runs, **common)


if __name__ == "__main__":
    main()
