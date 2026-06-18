"""Build a behavior-cloning dataset from pruned Contra win traces.

Each trace stores only `actions (T,9)` + `initial_state` — no observations or
rewards. We regenerate them by replaying the trace through the **real**
`ContraWrapper` (same frame-skip / stack / channel layout / reward as PPO), so
the cloned policy sees exactly the observations it will see at PPO time. This
is the #1 correctness requirement (see ppo/cnn_pretrain_design.md §3).

Output per step: `(obs, action_idx, return_to_go)`:
  - obs            : (84, 84, stack) uint8, the wrapper observation *before* the action
  - action_idx     : discrete index into the canonical action space
  - return_to_go   : discounted sum of the wrapper's shaped reward (same gamma as PPO)

Usage:
    python ppo/pretrain_dataset.py --level 1            # build + cache + stats
    python ppo/pretrain_dataset.py --level 1 --dry-run  # stats only, no cache
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np

import contra  # noqa: F401  registers the custom ROM integration
import stable_retro as retro

sys.path.insert(0, os.path.dirname(__file__))
from contra_wrapper import ContraWrapper, process_frame  # noqa: E402

from contra.action_space import DEFAULT as ACTION_SPACE
from contra.reward import load as load_reward_config, progress_coord, xscroll

GAME = "Contra-Nes"
TRACE_DIR = "tmp/mc_trace"
CACHE_DIR = "tmp/ppo/pretrain"

# Exact-match map from a 9-button NES vector to its canonical discrete index.
# Verified (ppo/pretrain_dataset stats) that every pruned trace action is in
# this set, so an exact lookup never misses.
_VEC_TO_IDX = {tuple(int(b) for b in v): i for i, v in enumerate(ACTION_SPACE.actions_np())}

# Per-index button bits used for class stats / per-button recall.
# NES layout: index 0 = B (fire), index 8 = A (jump).
_FIRE_BIT = ACTION_SPACE.actions_np()[:, 0].astype(bool)
_JUMP_BIT = ACTION_SPACE.actions_np()[:, 8].astype(bool)


def discounted_returns(rewards, gamma):
    """Return-to-go G[t] = r[t] + gamma * G[t+1]."""
    g = np.zeros(len(rewards), dtype=np.float32)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        g[t] = acc
    return g


def _make_replay_wrapper(reward_weights, stack, max_steps):
    """A ContraWrapper around a fresh retro env, configured like PPO training.

    FILTERED actions + IMAGE obs match `ppo/train.py make_env`. warmup is 0
    because we overwrite the emulator with the trace's initial_state right after
    reset (see seed_wrapper_at_state)."""
    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    return ContraWrapper(
        env, warmup_frames=0, random_start_frames=0,
        stack=stack, level=1, max_episode_steps=max_steps,
        reward_weights=reward_weights,
    )


def seed_wrapper_at_state(wrapper, initial_bytes):
    """Reposition a (already reset) wrapper at a trace's initial_state and return obs.

    Mirrors ContraWrapper.reset's end-of-reset sync, but sources the frame from
    the loaded savestate via em.get_screen() (the same trick replay.py uses)."""
    wrapper.unwrapped.em.set_state(initial_bytes)
    wrapper.unwrapped.data.update_ram()

    ram = wrapper.unwrapped.get_ram()
    wrapper.prev_ram = ram.copy()
    wrapper.prev_xscroll = xscroll(ram)
    wrapper.max_progress = progress_coord(ram)
    wrapper.episode_start_progress = wrapper.max_progress
    wrapper.total_timesteps = 0
    wrapper._reset_episode_stats()

    frame = process_frame(wrapper.unwrapped.em.get_screen())
    wrapper._buf[:] = frame
    wrapper._buf_pos = 0
    return wrapper._get_obs()


def replay_trace(trace_path, reward_weights, stack, gamma):
    """Replay one trace → (obs, action_idx, return_to_go) arrays + outcome dict."""
    d = np.load(trace_path, allow_pickle=True)
    actions = d["actions"]
    initial = bytes(d["initial_state"])

    wrapper = _make_replay_wrapper(reward_weights, stack, max_steps=len(actions) + 16)
    wrapper.reset()
    obs = seed_wrapper_at_state(wrapper, initial)

    obs_list, act_list, rew_list = [], [], []
    end_reason = ""
    for vec in actions:
        idx = _VEC_TO_IDX[tuple(int(b) for b in vec)]
        obs_list.append(obs)                       # observation *before* the action
        act_list.append(idx)
        obs, reward, term, trunc, info = wrapper.step(idx)
        rew_list.append(reward)
        if term or trunc:
            end_reason = info.get("episode_end_reason", "")
            break
    progress = wrapper.max_progress - wrapper.episode_start_progress
    wrapper.close()

    obs_arr = np.asarray(obs_list, dtype=np.uint8)
    act_arr = np.asarray(act_list, dtype=np.int64)
    ret_arr = discounted_returns(rew_list, gamma)
    return obs_arr, act_arr, ret_arr, {"end_reason": end_reason, "progress": int(progress)}


def _print_stats(actions, returns):
    """Action-frequency histogram + fire/jump rate + critic-target range."""
    n = len(actions)
    counts = np.bincount(actions, minlength=ACTION_SPACE.num_actions)
    fire = int(counts[_FIRE_BIT].sum())
    jump = int(counts[_JUMP_BIT].sum())
    print(f"\n  dataset: {n} samples across {ACTION_SPACE.num_actions} actions")
    print(f"  fire rate: {fire / n:.3f}   jump rate: {jump / n:.3f}")
    print(f"  return-to-go: min {returns.min():.1f}  max {returns.max():.1f}  mean {returns.mean():.1f}")
    print(f"\n  {'idx':>3}  {'name':<6}  {'count':>7}  {'freq':>6}")
    order = np.argsort(-counts)
    for i in order:
        if counts[i] == 0:
            continue
        print(f"  {i:>3}  {ACTION_SPACE.names[i]:<6}  {counts[i]:>7}  {counts[i] / n:>6.3f}")


def build_dataset(traces, reward_config="stable", stack=3, gamma=0.99, verbose=True):
    """Replay all `traces` → concatenated (obs, actions, returns) + per-trace report."""
    reward_weights = load_reward_config(reward_config).reward_weights
    obs_all, act_all, ret_all = [], [], []
    wins = 0
    for t in traces:
        obs, act, ret, info = replay_trace(t, reward_weights, stack, gamma)
        won = info["end_reason"] == "win"
        wins += int(won)
        if verbose:
            flag = "win " if won else f"!{info['end_reason'] or 'no-end'}"
            print(f"  {os.path.basename(t):<40} {len(act):5d} steps  {flag}  prog={info['progress']}")
        obs_all.append(obs)
        act_all.append(act)
        ret_all.append(ret)

    obs_all = np.concatenate(obs_all)
    act_all = np.concatenate(act_all)
    ret_all = np.concatenate(ret_all)
    if verbose:
        print(f"\n  replayed {len(traces)} traces, {wins} reproduced a win "
              f"({wins}/{len(traces)})")
        if wins != len(traces):
            print("  WARNING: some traces did not reproduce a win through the wrapper — "
                  "obs/dynamics pipeline may diverge from trace generation.")
        _print_stats(act_all, ret_all)
    return obs_all, act_all, ret_all


def resolve_traces(level):
    pool = sorted(glob.glob(os.path.join(TRACE_DIR, f"level{level}", f"win_level{level}_*.npz")))
    if not pool:
        raise SystemExit(f"No traces found for level {level} in {TRACE_DIR}/level{level}/")
    return pool


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--reward-config", default="stable")
    p.add_argument("--stack", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--out", default=None, help="Cache npz path (default: CACHE_DIR/level<N>_bc.npz)")
    p.add_argument("--dry-run", action="store_true", help="Print stats without writing the cache")
    args = p.parse_args()

    traces = resolve_traces(args.level)
    print(f"Building BC dataset for level {args.level} from {len(traces)} pruned traces "
          f"(reward={args.reward_config}, stack={args.stack}, gamma={args.gamma})")
    obs, act, ret = build_dataset(traces, args.reward_config, args.stack, args.gamma)

    if args.dry_run:
        print("\n  --dry-run: cache not written")
        return
    out = args.out or os.path.join(CACHE_DIR, f"level{args.level}_bc.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, obs=obs, actions=act, returns=ret,
             stack=args.stack, gamma=args.gamma, reward_config=args.reward_config)
    print(f"\n  wrote {len(act)} samples → {out}  ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
