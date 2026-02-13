"""
playfun — Play NES Games via Beam Search with Time Travel
==========================================================

Uses objectives discovered by learnfun to play games. Maintains a pool of
candidate "futures" (input sequences), scores them by simulating forward
from the current state, and commits the best one. Periodically backtracks
to earlier checkpoints to escape local optima.

Scoring is parallelized across N_WORKERS processes, each with its own
emulator instance.

Based on Tom Murphy VII's playfun algorithm.

Usage:
    python playfun.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from dataclasses import dataclass, field

import numpy as np
import stable_retro as retro

from objectives import WeightedObjectives

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GAME = "Contra-Nes"
STATE = "Level1"
OBJECTIVES_PATH = os.path.join(os.path.dirname(__file__), "contra.objectives")
MAX_FRAMES = 10000
RECORD_PATH = os.path.join(os.path.dirname(__file__), "contra_lex.gif")
N_WORKERS = 32

# ---------------------------------------------------------------------------
# Hyperparameters (from original C++)
# ---------------------------------------------------------------------------
N_FUTURES = 40
DROP_FUTURES = 5
MUTATE_FUTURES = 7
LOOKAHEAD = 10           # frames per input chunk (motif size)
MAX_FUTURE_LENGTH = 800  # max frames in a future sequence
CHECKPOINT_EVERY = 10    # save state every N rounds
TRY_BACKTRACK_EVERY = 18
MIN_BACKTRACK_DISTANCE = 300


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Future:
    """A candidate input sequence with its cached score."""
    inputs: list[np.ndarray] = field(default_factory=list)
    score: float = 0.0


@dataclass
class Checkpoint:
    """A saved emulator state at a given frame."""
    frame: int = 0
    state: bytes = b""
    ram: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Input generation helpers
# ---------------------------------------------------------------------------
def random_inputs(n: int, action_space) -> list[np.ndarray]:
    """Generate n random input frames."""
    return [action_space.sample() for _ in range(n)]


def mutate_inputs(inputs: list[np.ndarray], action_space) -> list[np.ndarray]:
    """Create a mutated copy of an input sequence."""
    result = [inp.copy() for inp in inputs]
    if not result:
        return random_inputs(LOOKAHEAD, action_space)

    mutation = random.choice(["flip", "insert", "delete", "replace"])
    idx = random.randint(0, max(0, len(result) - 1))

    if mutation == "flip" and len(result) > 0:
        btn = random.randint(0, result[idx].shape[0] - 1)
        result[idx][btn] ^= 1
    elif mutation == "insert":
        result.insert(idx, action_space.sample())
    elif mutation == "delete" and len(result) > 1:
        result.pop(idx)
    elif mutation == "replace":
        result[idx] = action_space.sample()

    if len(result) > MAX_FUTURE_LENGTH:
        result = result[:MAX_FUTURE_LENGTH]

    return result


def dualize(inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Mirror directional inputs: LEFT<->RIGHT, UP<->DOWN, A<->B.

    NES FILTERED layout: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
    Indices:              0  1     2       3      4   5     6     7      8
    """
    result = []
    for inp in inputs:
        d = inp.copy()
        d[6], d[7] = inp[7], inp[6]
        d[4], d[5] = inp[5], inp[4]
        d[0], d[8] = inp[8], inp[0]
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Worker process: each has its own emulator
# ---------------------------------------------------------------------------
_worker_env: retro.RetroEnv | None = None
_worker_objectives: WeightedObjectives | None = None


def _worker_init(game: str, state: str, objectives_path: str) -> None:
    """Initialize a worker process with its own env and objectives."""
    global _worker_env, _worker_objectives
    _worker_env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    _worker_env.reset()
    _worker_objectives = WeightedObjectives.load(objectives_path)


def _worker_score(args: tuple) -> float:
    """Score a single candidate in a worker process.

    Args: (state_bytes, ram_before, candidate_inputs, futures_inputs)
    """
    state_bytes, ram_before, candidate, futures_inputs = args
    env = _worker_env
    objectives = _worker_objectives

    # Restore to the shared state
    env.em.set_state(state_bytes)
    env.data.update_ram()

    # Execute candidate
    for inp in candidate:
        env.step(inp)
    ram_after = env.get_ram().copy()

    # Immediate score
    score = objectives.evaluate(ram_before, ram_after)

    # Future scores
    for fut_inputs in futures_inputs:
        if not fut_inputs:
            continue
        save = env.em.get_state()
        for inp in fut_inputs[:MAX_FUTURE_LENGTH]:
            env.step(inp)
        ram_future = env.get_ram().copy()
        score += objectives.evaluate(ram_after, ram_future)
        env.em.set_state(save)
        env.data.update_ram()

    return score


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def generate_candidates(
    futures: list[Future], action_space, n: int
) -> list[list[np.ndarray]]:
    """Generate n candidate input chunks from existing futures + random."""
    candidates: list[list[np.ndarray]] = []

    for fut in futures:
        if fut.inputs:
            candidates.append([inp.copy() for inp in fut.inputs[:LOOKAHEAD]])

    while len(candidates) < n:
        candidates.append(random_inputs(LOOKAHEAD, action_space))

    return candidates[:n]


# ---------------------------------------------------------------------------
# Future management
# ---------------------------------------------------------------------------
def trim_futures(futures: list[Future], n_committed: int) -> None:
    """Remove the first n_committed frames from each future's input sequence."""
    for fut in futures:
        fut.inputs = fut.inputs[n_committed:]


def cull_and_repopulate(futures: list[Future], action_space) -> None:
    """Drop worst futures, add mutations of the best."""
    if len(futures) < 2:
        return

    futures.sort(key=lambda f: f.score, reverse=True)

    n_drop = min(DROP_FUTURES, len(futures) - 1)
    for _ in range(n_drop):
        futures.pop()

    if futures:
        best = futures[0]
        for _ in range(MUTATE_FUTURES):
            new_inputs = mutate_inputs(best.inputs, action_space)
            futures.append(Future(inputs=new_inputs, score=0.0))

    if futures and len(futures[0].inputs) > 0:
        futures.append(Future(inputs=dualize(futures[0].inputs), score=0.0))

    while len(futures) > N_FUTURES:
        futures.pop()


# ---------------------------------------------------------------------------
# Backtracking (runs on main env only)
# ---------------------------------------------------------------------------
def try_backtrack(
    env: retro.RetroEnv,
    checkpoints: list[Checkpoint],
    committed_inputs: list[np.ndarray],
    objectives: WeightedObjectives,
    current_frame: int,
    recorder: Recorder,
) -> tuple[bool, int]:
    """Try to find a better path from a past checkpoint.

    Returns (did_backtrack, new_current_frame).
    """
    if not checkpoints:
        return False, current_frame

    target_frame = current_frame - MIN_BACKTRACK_DISTANCE
    candidate_cp = None
    for cp in reversed(checkpoints):
        if cp.frame <= target_frame:
            candidate_cp = cp
            break

    if candidate_cp is None:
        return False, current_frame

    ram_now = env.get_ram().copy()

    env.em.set_state(candidate_cp.state)
    env.data.update_ram()
    ram_at_cp = candidate_cp.ram.copy()

    inputs_segment = committed_inputs[candidate_cp.frame:current_frame]
    if not inputs_segment:
        for inp in committed_inputs[candidate_cp.frame:]:
            env.step(inp)
        return False, current_frame

    original_score = objectives.evaluate(ram_at_cp, ram_now)

    mutations = [
        ("dualize", dualize(inputs_segment)),
        ("random", random_inputs(len(inputs_segment), env.action_space)),
    ]
    if len(inputs_segment) > 2:
        half = inputs_segment[: len(inputs_segment) // 2]
        mutations.append(("chop", half + half))

    best_mutation = None
    best_score = original_score

    for name, mutated in mutations:
        env.em.set_state(candidate_cp.state)
        env.data.update_ram()
        for inp in mutated:
            env.step(inp)
        ram_mutated = env.get_ram().copy()
        score = objectives.evaluate(ram_at_cp, ram_mutated)
        if score > best_score:
            best_score = score
            best_mutation = (name, mutated)

    if best_mutation is not None:
        name, mutated = best_mutation
        print(f"  Backtrack: {name} mutation improved score {original_score:.2f} -> {best_score:.2f}")

        recorder.truncate_to_frame(candidate_cp.frame)
        env.em.set_state(candidate_cp.state)
        env.data.update_ram()
        for inp in mutated:
            obs, _, _, _, _ = env.step(inp)
            recorder.add(obs)

        committed_inputs[candidate_cp.frame:current_frame] = mutated
        return True, candidate_cp.frame + len(mutated)

    env.em.set_state(candidate_cp.state)
    env.data.update_ram()
    for inp in committed_inputs[candidate_cp.frame:current_frame]:
        env.step(inp)
    return False, current_frame


# ---------------------------------------------------------------------------
# Recording helper
# ---------------------------------------------------------------------------
class Recorder:
    """Collect raw RGB frames and save as GIF.

    Tracks frame count so backtracking can truncate to a past point.
    """

    def __init__(self, path: str | None, skip: int = 4):
        self.path = path
        self.skip = skip
        self.frames: list[np.ndarray] = []
        self.frame_count = 0

    def add(self, frame: np.ndarray) -> None:
        if self.path is None:
            return
        self.frame_count += 1
        if self.frame_count % self.skip == 0:
            self.frames.append(frame.copy())

    def truncate_to_frame(self, target_frame: int) -> None:
        """Discard recorded frames beyond target_frame."""
        if self.path is None:
            return
        keep = target_frame // self.skip
        self.frames = self.frames[:keep]
        self.frame_count = target_frame

    def save(self) -> None:
        if self.path and self.frames:
            import imageio
            imageio.mimsave(self.path, self.frames, duration=20)
            print(f"Saved recording to {self.path} ({len(self.frames)} frames)")


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------
def play() -> None:
    print(f"playfun: {GAME} / {STATE}")
    print(f"  Objectives:  {OBJECTIVES_PATH}")
    print(f"  Max frames:  {MAX_FRAMES}")
    print(f"  Workers:     {N_WORKERS}")
    print(f"  Record:      {RECORD_PATH}")

    # Main env — used for committing moves and recording
    env = retro.make(
        game=GAME,
        state=STATE,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    obs, _ = env.reset()

    objectives = WeightedObjectives.load(OBJECTIVES_PATH)
    print(f"Loaded {len(objectives.objectives)} objectives "
          f"({objectives.active_count()} active)")

    objectives.observe(env.get_ram())

    recorder = Recorder(RECORD_PATH)
    recorder.add(obs)

    # Worker pool — each process gets its own emulator
    # Use 'spawn' context so workers don't inherit the main env
    # (retro only allows one emulator per process)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(GAME, STATE, OBJECTIVES_PATH),
    )

    futures: list[Future] = []
    checkpoints: list[Checkpoint] = []
    committed_inputs: list[np.ndarray] = []
    current_frame = 0
    round_num = 0

    for _ in range(N_FUTURES):
        futures.append(Future(inputs=random_inputs(LOOKAHEAD * 4, env.action_space)))

    print(f"\n--- Playing ({MAX_FRAMES} frames max) ---")

    try:
        while current_frame < MAX_FRAMES:
            # 1. Generate candidate input chunks
            candidates = generate_candidates(futures, env.action_space, n=N_FUTURES)

            # 2. Score all candidates in parallel
            state_bytes = env.em.get_state()
            ram_before = env.get_ram().copy()
            futures_inputs = [f.inputs for f in futures if f.inputs]

            work_items = [
                (state_bytes, ram_before, cand, futures_inputs)
                for cand in candidates
            ]
            scores = pool.map(_worker_score, work_items)

            # Pick best
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            best_candidate = candidates[best_idx]

            # 3. Commit best candidate on main env
            for inp in best_candidate:
                obs, _, terminated, truncated, info = env.step(inp)
                recorder.add(obs)
                objectives.observe(env.get_ram())
                committed_inputs.append(inp)
                current_frame += 1
                if terminated or truncated:
                    print(f"  Episode ended at frame {current_frame} (round {round_num})")
                    env.reset()
                    objectives.observe(env.get_ram())

            # 4. Trim committed frames from futures
            trim_futures(futures, len(best_candidate))

            # 5. Re-score futures in parallel (no further lookahead)
            state_bytes = env.em.get_state()
            ram_before = env.get_ram().copy()
            active_futures = [(i, f) for i, f in enumerate(futures) if f.inputs]
            if active_futures:
                fut_work = [
                    (state_bytes, ram_before, f.inputs[:LOOKAHEAD], [])
                    for _, f in active_futures
                ]
                fut_scores = pool.map(_worker_score, fut_work)
                for (i, f), s in zip(active_futures, fut_scores):
                    f.score = s

            # 6. Cull worst, repopulate with mutations
            cull_and_repopulate(futures, env.action_space)

            # 7. Checkpoint periodically
            if round_num % CHECKPOINT_EVERY == 0:
                cp = Checkpoint(
                    frame=current_frame,
                    state=env.em.get_state(),
                    ram=env.get_ram().copy(),
                )
                checkpoints.append(cp)

            # 8. Backtrack if stuck
            if round_num % TRY_BACKTRACK_EVERY == 0 and round_num > 0:
                did_bt, new_frame = try_backtrack(
                    env, checkpoints, committed_inputs, objectives, current_frame, recorder
                )
                if did_bt:
                    current_frame = new_frame

            round_num += 1

            if round_num % 10 == 0:
                active_futs = sum(1 for f in futures if f.inputs)
                print(f"  Round {round_num}: frame {current_frame}/{MAX_FRAMES}, "
                      f"best_score={best_score:.3f}, futures={active_futs}, "
                      f"checkpoints={len(checkpoints)}")

    finally:
        pool.terminate()
        pool.join()

    env.close()
    recorder.save()
    print(f"\nDone. Played {current_frame} frames in {round_num} rounds.")


if __name__ == "__main__":
    play()
