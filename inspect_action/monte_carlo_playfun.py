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
SKIP = 4
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
            done=self.done,
        )


def compute_reward(state: State, info: dict) -> float:
    reward = 0.0
    curr_xscroll = info.get("xscroll", state.xscroll)
    curr_score = info.get("score", 0)
    curr_lives = info.get("lives", 0)

    if curr_xscroll > state.max_x_reached:
        state.idle_steps = 0
        state.max_x_reached = curr_xscroll
    else:
        state.idle_steps += 1

    if state.idle_steps > 20:
        reward -= 0.05
    else:
        pos_delta = curr_xscroll - state.xscroll
        reward += max(min(pos_delta, 3.0), 0) * (1 / 10)
        score_delta = curr_score - state.score
        reward += max(score_delta, 0)

    if curr_lives < state.lives:
        reward -= 10000  # Massive penalty for death

    state.xscroll = curr_xscroll
    state.score = curr_score
    state.lives = curr_lives
    return reward


def step_env(env, action_idx: int):
    nes_action = ACTION_TABLE[action_idx]
    done = False
    info = {}
    for i in range(SKIP):
        act = list(nes_action)
        if i == SKIP - 1:
            act[0] = 0
        _, _, term, trunc, info = env.step(act)
        if term or trunc:
            done = True
            break
    return info, done


def run_random_rollout(env, start_state: State, length: int) -> tuple:
    """Runs a random action sequence of `length`. Returns (sequence, final_reward, died, is_win)"""
    num_actions = len(ACTION_TABLE)
    seq = np.random.randint(0, num_actions, size=length)
    
    env.em.set_state(start_state.emu_state)
    env.data.update_ram()
    
    child = start_state.clone()
    died = False
    is_win = False
    
    for act in seq:
        info, done = step_env(env, act)
        reward = compute_reward(child, info)
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
                    patience: int, max_steps: int, max_time: int):
    num_actions = len(ACTION_TABLE)

    committed = State(
        emu_state=initial_emu_state,
        xscroll=initial_info.get("xscroll", 0),
        score=initial_info.get("score", 0),
        lives=initial_info.get("lives", 0),
        max_x_reached=initial_info.get("xscroll", 0),
    )
    committed_actions = []

    best_checkpoint = committed.clone()
    best_checkpoint_idx = 0
    best_reward = committed.cumulative_reward

    stale_count = 0
    rewind_count = 0
    max_trap_idx = 0
    t_start = time.time()

    print(f"\n{'Step':>5} {'Action':>6} {'Reward':>8} {'xscroll':>7} "
          f"{'Score':>5} {'Lives':>5} {'Stale':>5} {'Rewinds':>7} {'Time':>6}")
    print("-" * 70)

    while len(committed_actions) < max_steps:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            print(f"\n  ⏱ Time budget exhausted ({max_time:.0f}s)")
            break
            
        if committed.done:
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
        
        for _ in range(rollouts):
            seq, final_rwrd, died, is_win = run_random_rollout(env, committed, rollout_len)
            
            if is_win:
                found_win = True
                winning_seq = seq
                break
                
            if final_rwrd > best_future_reward:
                best_future_reward = final_rwrd
                best_seq = seq
                best_died = died

        # ==========================================
        # 2. COMMIT OR REWIND
        # ==========================================
        if found_win:
            print(f"\n  🎯 Rollout found a WIN! Committing full sequence...")
            actions_to_commit = winning_seq
        elif best_died:
            print(f"  ☠️  All {rollouts} futures end in death! Trap detected! Forcing backtrack...")
            stale_count = patience # Force rewind
            actions_to_commit = []
        else:
            # We survived the future! Commit the first chunk of the plan
            actions_to_commit = best_seq[:commit_steps]

        # Execute chosen actions
        first_action_name = ACTION_NAMES[actions_to_commit[0]] if len(actions_to_commit) > 0 else "?"
        
        for act in actions_to_commit:
            env.em.set_state(committed.emu_state)
            env.data.update_ram()
            info, done = step_env(env, act)
            
            new_committed = committed.clone()
            new_committed.emu_state = env.em.get_state()
            new_committed.done = done
            reward = compute_reward(new_committed, info)
            
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
            if (step_num // 10) > (prev_step_num // 10) or committed.done or found_win:
                print(f"{step_num:5d} {first_action_name:>6} "
                      f"{committed.cumulative_reward:8.2f} {committed.xscroll:7d} "
                      f"{committed.score:5d} {committed.lives:5d} "
                      f"{stale_count:5d} {rewind_count:7d} "
                      f"{elapsed:5.1f}s")
                      
        # BACKTRACK!
        if stale_count >= patience:
            rewind_count += 1
            max_trap_idx = max(max_trap_idx, len(committed_actions))
            
            # Rewind backwards dynamically - rewind further the more we fail consecutively!
            rewind_amount = (patience // 2) * rewind_count
            rewind_to = max(0, best_checkpoint_idx - rewind_amount)
            
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
            )
            
            for act in committed_actions[:rewind_to]:
                env.em.set_state(replay_state.emu_state)
                env.data.update_ram()
                info, done = step_env(env, act)
                replay_state.emu_state = env.em.get_state()
                replay_state.done = done
                reward = compute_reward(replay_state, info)
                if done and info.get("lives", 0) > 0:
                    reward += 100
                replay_state.cumulative_reward += reward
                
            committed = replay_state
            committed_actions = committed_actions[:rewind_to]
            
            # Inject random action to diverge timeline
            random_action = np.random.randint(num_actions)
            env.em.set_state(committed.emu_state)
            env.data.update_ram()
            info, done = step_env(env, random_action)
            committed.emu_state = env.em.get_state()
            committed.done = done
            reward = compute_reward(committed, info)
            committed.cumulative_reward += reward
            committed_actions.append(random_action)
            
            stale_count = 0
            
            # CRITICAL: We changed the timeline! We must forget the old high score 
            # and set the current state as the new checkpoint, otherwise we will 
            # instantly get "stale" because we can't beat the future we just erased!
            best_reward = committed.cumulative_reward
            best_checkpoint = committed.clone()
            best_checkpoint_idx = len(committed_actions)
            
            print(f"  → Injected random action {ACTION_NAMES[random_action]}, "
                  f"now at step {len(committed_actions)} (max_trap_idx: {max_trap_idx})\n")

    return committed_actions, committed


# =========================================================================
# REPLAY BEST SEQUENCE AS GIF
# =========================================================================
def replay_and_record(env, initial_emu_state: bytes, initial_info: dict, actions: list, trace_path: str, gif_path: str = None):
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
        nes_action = ACTION_TABLE[action_idx]
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
    state_file = "main/states/Level1_x3048_step921.state"
    rollouts = 256
    rollout_len = 16
    commit_steps = 4
    patience = 10
    max_steps = 4000
    max_time = 600
    save_gif = True

    # Make absolutely sure we get different numpy random streams each run
    np.random.seed(int(time.time() * 1000) % (2**32))

    import gzip
    with gzip.open(state_file, "rb") as f:
        custom_state_data = f.read()

    env = retro.make(
        game=GAME, state="Level1",
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    env.reset()

    env.em.set_state(custom_state_data)
    env.data.update_ram()
    env.step([0] * 9)

    initial_emu_state = env.em.get_state()
    initial_info = env.data.lookup_all()

    print("=" * 70)
    print("Playfun — Monte Carlo Search with Backtracking")
    print("=" * 70)
    print(f"  State:              {os.path.basename(state_file)}")
    print(f"  Rollouts/Step:      {rollouts} (random sequences evaluated)")
    print(f"  Rollout Length:     {rollout_len} actions ({rollout_len * 4} frames)")
    print(f"  Commit Steps:       {commit_steps} actions at a time")
    print(f"  Patience:           {patience} stale commits before rewind")
    print(f"  Time Budget:        {max_time}s")
    print("=" * 70)

    actions, final_state = search_and_play(
        env, initial_emu_state, initial_info,
        rollouts=rollouts,
        rollout_len=rollout_len,
        commit_steps=commit_steps,
        patience=patience,
        max_steps=max_steps,
        max_time=max_time
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  Actions: {len(actions)}")
    print(f"  Reward:  {final_state.cumulative_reward:.2f}")
    print(f"  Score:   {final_state.score}")
    print(f"  Lives:   {final_state.lives}")
    
    gif_path = os.path.join(GIFS_DIR, f"mc_backtrack_{os.path.basename(state_file)[:-6]}.gif") if save_gif else None
    trace_path = os.path.join(TRACE_DIR, f"mc_trace_{os.path.basename(state_file)[:-6]}.npz")
    
    replay_and_record(env, initial_emu_state, initial_info, actions, trace_path, gif_path)

    env.close()

if __name__ == "__main__":
    main()
