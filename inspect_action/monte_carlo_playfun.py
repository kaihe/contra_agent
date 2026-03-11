"""
Playfun — Monte Carlo Search with Backtracking
=================================================

This algorithm combines the "far sight" of Monte Carlo trace rollouts
with the "time travel" of backtracking to escape traps.

Algorithm:
1. From the current COMMITTED state, generate N random rollouts
   (e.g., sequences of 16 actions).
2. Find the rollout with the highest cumulative reward.
3. If the BEST rollout still results in death, the COMMITTED state is
   likely a trap! Force a rewind.
4. Otherwise, commit the first K actions from the best rollout.
5. If the committed state stops improving for `PATIENCE` steps, REWIND
   to a previous checkpoint and inject a random action to diverge.

Usage:
    python monte_carlo_playfun.py --gif
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
from contra_wrapper import ACTION_TABLE, ACTION_NAMES, Monitor

GAME = "Contra-Nes"
SKIP = 8
GIFS_DIR = os.path.join(os.path.dirname(__file__), "gifs")
TRACE_DIR = os.path.join(os.path.dirname(__file__), "mc_trace")

# =========================================================================
# HYPERPARAMETERS (Default values if not provided via CLI)
# =========================================================================
DEFAULT_STATE = "main/states/Level1_x3048_step921.state"


@dataclass
class State:
    emu_state: bytes
    cumulative_reward: float = 0.0
    xscroll: int = 0
    score: int = 0
    lives: int = 0
    max_x_reached: int = 0
    idle_steps: int = 0
    enemy_hp_sum: int = 0
    cores: int = 0
    score_reward: float = 0.0
    enemy_hp_reward: float = 0.0
    cores_reward: float = 0.0
    done: bool = False

    def clone(self):
        return State(
            emu_state=self.emu_state,
            cumulative_reward=self.cumulative_reward,
            xscroll=self.xscroll,
            score=self.score,
            lives=self.lives,
            max_x_reached=self.max_x_reached,
            idle_steps=self.idle_steps,
            enemy_hp_sum=self.enemy_hp_sum,
            cores=self.cores,
            score_reward=self.score_reward,
            enemy_hp_reward=self.enemy_hp_reward,
            cores_reward=self.cores_reward,
            done=self.done,
        )


def get_enemy_hp_sum(env):
    """Sum of all 16 enemy HP slots at RAM[0x578..0x587].
    Works for any level — enemies on screen, boss components, all included."""
    ram = env.get_ram()
    return sum(int(ram[1400 + i]) for i in range(16))

def get_cores(env):
    """Wall cores remaining at RAM[0x0086]."""
    return int(env.get_ram()[0x0086])

def get_score(env):
    """Read score directly from RAM[0x07E2..0x07E3] (little-endian 2 bytes)."""
    ram = env.get_ram()
    return int(ram[0x07E2]) + (int(ram[0x07E3]) << 8)


SCORE_SCALE = 0.05
CORE_PENALTY = -0.01

def compute_reward(state: State, info: dict, env, level: int = 1) -> float:
    reward = 0.0
    curr_score = get_score(env)
    curr_lives = info.get("lives", 0)
    curr_enemy_hp_sum = get_enemy_hp_sum(env)
    curr_cores = get_cores(env)

    # Score reward: delta * scale (all levels)
    score_delta = curr_score - state.score
    sr = score_delta * SCORE_SCALE
    state.score_reward += sr
    reward += sr

    # Cores reward: delta reward + continuous gentle penalty for cores remaining
    core_diff = state.cores - curr_cores
    cr = core_diff * 5.0 + curr_cores * CORE_PENALTY
    state.cores_reward += cr
    reward += cr

    if curr_lives < state.lives:
        reward -= 10000  # Massive penalty for death

    state.score = curr_score
    state.lives = curr_lives
    state.enemy_hp_sum = curr_enemy_hp_sum
    state.cores = curr_cores
    return reward


def step_env(env, action_idx: int, action_table=None, skip=None):
    if action_table is None:
        action_table = ACTION_TABLE
    if skip is None:
        skip = SKIP
    nes_action = action_table[action_idx]
    done = False
    info = {}
    for i in range(skip):
        act = list(nes_action)
        if i == skip - 1:
            act[0] = 0
        _, _, term, trunc, info = env.step(act)
        if term or trunc:
            done = True
            break
    return info, done


def run_random_rollout(env, start_state: State, length: int,
                       action_table=None, skip=None, level: int = 1) -> tuple:
    """Runs a random action sequence of `length`. Returns (sequence, final_reward, died, is_win)"""
    if action_table is None:
        action_table = ACTION_TABLE
    num_actions = len(action_table)
    seq = np.random.randint(0, num_actions, size=length)

    env.em.set_state(start_state.emu_state)
    env.data.update_ram()

    child = start_state.clone()
    died = False
    is_win = False

    for act in seq:
        info, done = step_env(env, act, action_table, skip)
        reward = compute_reward(child, info, env, level=level)
        child.cumulative_reward += reward

        if info.get("lives", child.lives) < start_state.lives:
            died = True
            break

        if done:
            if info.get("lives", 0) > 0:
                child.cumulative_reward += 100
                is_win = True
            else:
                died = True
            break

    return seq, child.cumulative_reward, died, is_win


def search_and_play(env, initial_emu_state: bytes, initial_info: dict,
                    rollouts: int, rollout_len: int, commit_steps: int,
                    patience: int, max_steps: int, max_time: int,
                    action_table=None, action_names=None, verbose=True,
                    rollout_budget: int = None, skip: int = None,
                    level: int = 1):
    if action_table is None:
        action_table = ACTION_TABLE
    if action_names is None:
        action_names = ACTION_NAMES if action_table is ACTION_TABLE \
                       else [str(i) for i in range(len(action_table))]
    num_actions = len(action_table)

    committed = State(
        emu_state=initial_emu_state,
        xscroll=initial_info.get("xscroll", 0),
        score=initial_info.get("score", 0),
        lives=initial_info.get("lives", 0),
        max_x_reached=initial_info.get("xscroll", 0),
        enemy_hp_sum=get_enemy_hp_sum(env),
        cores=get_cores(env),
    )
    committed_actions = []

    best_checkpoint = committed.clone()
    best_checkpoint_idx = 0
    best_reward = committed.cumulative_reward

    stale_count = 0
    rewind_count = 0
    max_trap_idx = 0
    total_rollout_evals = 0
    t_start = time.time()

    if verbose:
        print(f"\n{'Step':>5} {'Action':>6} {'Reward':>8} {'ScRew':>6} {'HPRew':>6} {'CoreRew':>8} {'Time':>6}")
        print("-" * 55)

    while len(committed_actions) < max_steps:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            if verbose:
                print(f"\n  ⏱ Time budget exhausted ({max_time:.0f}s)")
            break
        if rollout_budget is not None and total_rollout_evals >= rollout_budget:
            if verbose:
                print(f"\n  ⏱ Rollout budget exhausted ({total_rollout_evals} evals)")
            break

        if committed.done:
            if verbose:
                if committed.lives > 0:
                    print(f"\n  🏆 WIN!")
                else:
                    print(f"\n  💀 Game Over")
            break

        # ==========================================
        # 1. MONTE CARLO LOOKAHEAD
        # ==========================================
        best_seq = None
        best_future_reward = -float('inf')
        best_died = True
        found_win = False
        winning_seq = None
        rollouts_done = 0

        for _ in range(rollouts):
            seq, final_rwrd, died, is_win = run_random_rollout(
                env, committed, rollout_len, action_table, skip, level=level)
            rollouts_done += 1

            if is_win:
                found_win = True
                winning_seq = seq
                break

            if final_rwrd > best_future_reward:
                best_future_reward = final_rwrd
                best_seq = seq
                best_died = died

        total_rollout_evals += rollouts_done * rollout_len

        # ==========================================
        # 2. COMMIT OR REWIND
        # ==========================================
        if found_win:
            if verbose:
                print(f"\n  🎯 Rollout found a WIN! Committing full sequence...")
            actions_to_commit = winning_seq
        elif best_died:
            if verbose:
                print(f"  ☠️  All {rollouts} futures end in death! Trap detected! Forcing backtrack...")
            stale_count = patience # Force rewind
            actions_to_commit = []
        else:
            # We survived the future! Commit the first chunk of the plan
            actions_to_commit = best_seq[:commit_steps]

        # Execute chosen actions
        first_action_name = action_names[actions_to_commit[0]] if len(actions_to_commit) > 0 else "?"

        for act in actions_to_commit:
            env.em.set_state(committed.emu_state)
            env.data.update_ram()
            info, done = step_env(env, act, action_table, skip)

            new_committed = committed.clone()
            new_committed.emu_state = env.em.get_state()
            new_committed.done = done
            reward = compute_reward(new_committed, info, env, level=level)

            if done and info.get("lives", 0) > 0:
                reward += 100
            new_committed.cumulative_reward += reward

            committed = new_committed
            committed_actions.append(act)

            if committed.done:
                break

        # ==========================================
        # 3. CHECK PROGRESS & BACKTRACK
        # ==========================================
        if len(actions_to_commit) > 0:
            if committed.cumulative_reward > best_reward:
                best_reward = committed.cumulative_reward
                best_checkpoint = committed.clone()
                best_checkpoint_idx = len(committed_actions)
                stale_count = 0
            else:
                stale_count += 1

            if len(committed_actions) > max_trap_idx:
                rewind_count = 0

            step_num = len(committed_actions)
            prev_step_num = step_num - len(actions_to_commit)
            if verbose and ((step_num // 10) > (prev_step_num // 10) or committed.done or found_win):
                print(f"{step_num:5d} {first_action_name:>6} "
                      f"{committed.cumulative_reward:8.2f} "
                      f"{committed.score_reward:6.1f} {committed.enemy_hp_reward:6.1f} {committed.cores_reward:8.1f} "
                      f"{elapsed:5.1f}s")

        # BACKTRACK!
        if stale_count >= patience:
            rewind_count += 1
            max_trap_idx = max(max_trap_idx, len(committed_actions))

            # Rewind backwards dynamically - rewind further the more we fail consecutively!
            rewind_amount = (patience // 2) * rewind_count
            rewind_to = max(0, best_checkpoint_idx - rewind_amount)

            if verbose:
                print(f"\n  ⏪ REWIND #{rewind_count}: Stuck/Trap! "
                      f"Rewinding from step {len(committed_actions)} back to step {rewind_to} "
                      f"(best was {best_checkpoint_idx})")

            # Replay to rewind point
            env.em.set_state(initial_emu_state)
            env.data.update_ram()
            env.step([0] * 9)

            replay_state = State(
                emu_state=initial_emu_state,
                xscroll=initial_info.get("xscroll", 0),
                score=initial_info.get("score", 0),
                lives=initial_info.get("lives", 0),
                max_x_reached=initial_info.get("xscroll", 0),
                enemy_hp_sum=get_enemy_hp_sum(env),
        cores=get_cores(env),
            )

            for act in committed_actions[:rewind_to]:
                env.em.set_state(replay_state.emu_state)
                env.data.update_ram()
                info, done = step_env(env, act, action_table, skip)
                replay_state.emu_state = env.em.get_state()
                replay_state.done = done
                reward = compute_reward(replay_state, info, env, level=level)
                if done and info.get("lives", 0) > 0:
                    reward += 100
                replay_state.cumulative_reward += reward

            committed = replay_state
            committed_actions = committed_actions[:rewind_to]

            # Inject random action to diverge timeline
            random_action = np.random.randint(num_actions)
            env.em.set_state(committed.emu_state)
            env.data.update_ram()
            info, done = step_env(env, random_action, action_table, skip)
            committed.emu_state = env.em.get_state()
            committed.done = done
            reward = compute_reward(committed, info, env, level=level)
            committed.cumulative_reward += reward
            committed_actions.append(random_action)

            stale_count = 0

            # CRITICAL: We changed the timeline! We must forget the old high score
            # and set the current state as the new checkpoint, otherwise we will
            # instantly get "stale" because we can't beat the future we just erased!
            best_reward = committed.cumulative_reward
            best_checkpoint = committed.clone()
            best_checkpoint_idx = len(committed_actions)

            if verbose:
                print(f"  → Injected random action {action_names[random_action]}, "
                      f"now at step {len(committed_actions)} (max_trap_idx: {max_trap_idx})\n")

    return committed_actions, committed, total_rollout_evals


# =========================================================================
# REPLAY BEST SEQUENCE AS GIF
# =========================================================================
def replay_and_record(env, initial_emu_state: bytes, initial_info: dict, actions: list, trace_path: str, gif_path: str = None, action_table=None):
    if action_table is None:
        action_table = ACTION_TABLE
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    monitor = None
    if gif_path:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        monitor = Monitor(240, 224, saved_path=gif_path)

    env.em.set_state(initial_emu_state)
    env.data.update_ram()

    if monitor:
        monitor.record(env.get_screen())

    ram_snapshots = []
    actions_history = []
    xscroll_history = []
    score_history = []
    lives_history = []

    # Record initial frame exactly like play_human.py
    ram_snapshots.append(env.get_ram().copy())
    actions_history.append(np.zeros(9, dtype=np.int8))
    xscroll_history.append(int(initial_info.get('xscroll', 0)))
    score_history.append(int(initial_info.get('score', 0)))
    lives_history.append(int(initial_info.get('lives', 0)))

    for action_idx in actions:
        nes_action = action_table[action_idx]
        for i in range(SKIP):
            act = list(nes_action)
            if i == SKIP - 1:
                act[0] = 0

            obs, _, term, trunc, info = env.step(act)

            ram_snapshots.append(env.get_ram().copy())
            actions_history.append(np.array(act, dtype=np.int8))
            xscroll_history.append(int(info.get('xscroll', 0)))
            score_history.append(int(info.get('score', 0)))
            lives_history.append(int(info.get('lives', 0)))

            if monitor:
                monitor.record(obs)
            if term or trunc:
                break
        if term or trunc:
            break

    if monitor:
        monitor.close()
        size_kb = os.path.getsize(gif_path) / 1024
        print(f"\nGIF saved: {gif_path} ({size_kb:.1f} KB, {len(monitor.frames)} frames)")

    np.savez_compressed(
        trace_path,
        ram=np.array(ram_snapshots, dtype=np.uint8),
        actions=np.array(actions_history, dtype=np.int8),
        xscroll=np.array(xscroll_history, dtype=np.int32),
        score=np.array(score_history, dtype=np.int32),
        lives=np.array(lives_history, dtype=np.int8),
    )
    print(f"Recorded trace (RAM + Actions) saved to: {trace_path}")


def main():
    parser = argparse.ArgumentParser(description="Playfun Monte Carlo Search")
    parser.add_argument("--state", type=str, default="main/states/Level1_x3022_step5543_boss_spread.state",
                        help="State name (e.g. Level2) or path to .state file")
    parser.add_argument("--rollouts", type=int, default=512)
    parser.add_argument("--rollout-len", type=int, default=16)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    rollouts = args.rollouts
    rollout_len = args.rollout_len
    commit_steps = args.commit_steps
    patience = args.patience
    max_steps = args.max_steps
    max_time = args.max_time
    save_gif = not args.no_gif

    # Make absolutely sure we get different numpy random streams each run
    np.random.seed(int(time.time() * 1000) % (2**32))

    import gzip

    import re
    state_arg = args.state
    if state_arg.endswith(".state"):
        # File path
        with gzip.open(state_arg, "rb") as f:
            custom_state_data = f.read()
        state_label = os.path.basename(state_arg)[:-6]
        m = re.search(r"Level(\d+)", state_label, re.IGNORECASE)
        init_state = m.group(0) if m else "Level1"
    else:
        # Named retro state (e.g. "Level2")
        custom_state_data = None
        init_state = state_arg
        state_label = state_arg

    m = re.search(r"Level(\d+)", state_label, re.IGNORECASE)
    level = int(m.group(1)) if m else 1

    env = retro.make(
        game=GAME, state=init_state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env.reset()

    if custom_state_data:
        env.em.set_state(custom_state_data)
        env.data.update_ram()
    env.step([0] * 9)

    initial_emu_state = env.em.get_state()
    initial_info = env.data.lookup_all()

    print("=" * 70)
    print("Playfun — Monte Carlo Search with Backtracking")
    print("=" * 70)
    print(f"  State:              {state_arg}")
    print(f"  Level:              {level} (inferred)")
    print(f"  Skip:               {SKIP}")
    print(f"  Rollouts/Step:      {rollouts} (random sequences evaluated)")
    print(f"  Rollout Length:     {rollout_len} actions ({rollout_len * SKIP} frames)")
    print(f"  Commit Steps:       {commit_steps} actions at a time")
    print(f"  Patience:           {patience} stale commits before rewind")
    print(f"  Time Budget:        {max_time}s")
    print("=" * 70)

    actions, final_state, _ = search_and_play(
        env, initial_emu_state, initial_info,
        rollouts=rollouts,
        rollout_len=rollout_len,
        commit_steps=commit_steps,
        patience=patience,
        max_steps=max_steps,
        max_time=max_time,
        level=level,
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  Actions: {len(actions)}")
    print(f"  Reward:  {final_state.cumulative_reward:.2f}")
    print(f"  Score:   {final_state.score}")
    print(f"  Lives:   {final_state.lives}")

    gif_path = os.path.join(GIFS_DIR, f"mc_backtrack_{state_label}.gif") if save_gif else None
    trace_path = os.path.join(TRACE_DIR, f"mc_trace_{state_label}.npz")

    replay_and_record(env, initial_emu_state, initial_info, actions, trace_path, gif_path)

    env.close()

if __name__ == "__main__":
    main()
