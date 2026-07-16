"""Trace-replay frame collector for Dreamer gates.

Replays the winning MC traces (tmp/mc_trace/level<N>/*.npz) through the emulator
to produce a diverse, deterministic, whole-level set of frames — far better
coverage than a Right+Fire loop that dies on the first screen. Each frame is the
screen the agent observes *after* a full SKIP block (matching dreamer.envs
DreamerObs), resized to `size`. Aligned RAM ground truth is captured per frame so
gates can probe player/enemy info.

Splitting is BY TRACE (`split_by_trace`): held-out frames come from playthroughs
never seen in training, so eval measures real generalization, not reconstruction
of near-duplicate neighbours.

    python -m dreamer.collect --smoke --level 1
"""

from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np
import stable_retro as retro

from contra.replay import GAME, SKIP, rewind_state
from contra.game_state import state_from_ram
from contra.action_space import DEFAULT as ACTION_SPACE
from contra.reward import load as _load_reward_config, reward_components, xscroll

# Use the tuned 'stable' reward config (shared with PPO/mc_search) rather than the
# raw DEFAULT weights: it keeps the levelup terminal small (1, not 100) and the
# dense progress stronger (0.1/px), so the reward scale is sane for the critic.
REWARD_WEIGHTS = _load_reward_config("stable").reward_weights


def _action_index_map() -> dict[tuple, int]:
    """Map a 9-bit NES button vector → its discrete action index (0..20).

    MC traces store raw button vectors; the RSSM/actor work in the shared
    21-action discrete space, so we look each trace action back up here.
    """
    return {tuple(int(b) for b in vec): i for i, vec in enumerate(ACTION_SPACE.actions)}


def fill_buffer_from_traces(buf, paths, max_traces=None, verbose=True) -> int:
    """Load whole-level trace transitions into a Component-2 ReplayBuffer.

    Records the PRE-action screen at each decision (the frame the agent observes
    before choosing — consistent with dreamer.envs DreamerObs), the discrete
    action index taken, the REAL shaped reward for that transition, is_first at
    each trace start, and is_terminal (the levelup that ends a winning trace).
    Winning traces have no death — mix in env-rollout data for cont=0 from death.
    Stride is implicitly 1: the RSSM needs consecutive transitions to learn
    dynamics, so we never skip frames here.
    """
    if max_traces:
        paths = paths[:max_traces]
    size = buf.obs_shape[0]
    amap = _action_index_map()
    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE, render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    n_added = 0
    try:
        for ti, p in enumerate(paths):
            z = np.load(p, allow_pickle=True)
            actions = np.asarray(z["actions"], dtype=np.uint8)
            rewind_state(env, bytes(z["initial_state"]))
            prev_ram = env.unwrapped.get_ram().copy()
            prev_xscroll = xscroll(prev_ram)
            carry_r, carry_term = 0.0, False
            for j, act in enumerate(actions):
                screen = env.em.get_screen()                       # F_j: pre-action
                img = cv2.resize(screen, (size, size), interpolation=cv2.INTER_AREA)
                a_idx = amap.get(tuple(int(b) for b in act), 0)
                # reward/terminal stored here describe the transition INTO F_j
                # (DreamerV3 convention — the state can predict reward-into-it).
                buf.add(img, a_idx, carry_r, is_first=(j == 0), is_terminal=carry_term)
                n_added += 1
                for _ in range(SKIP):
                    env.step(act.copy())
                curr_ram = env.unwrapped.get_ram()
                rewards = reward_components(prev_ram, curr_ram, REWARD_WEIGHTS,
                                            prev_xscroll, False)
                # At a level transition xscroll resets (~3072→0), so the raw
                # progress term reads a huge spurious negative; zero it so the
                # levelup terminal reward is a clean +levelup, not +1−307.
                if rewards.get("levelup", 0.0) != 0.0 and "progress" in rewards:
                    rewards["progress"] = 0.0
                carry_r = float(sum(rewards.values()))
                carry_term = rewards["levelup"] != 0.0 or rewards["player_die"] != 0.0
                prev_ram = curr_ram.copy()
                prev_xscroll = xscroll(curr_ram)
            # terminal observation carrying the final (levelup) reward + flag
            final = cv2.resize(env.em.get_screen(), (size, size), interpolation=cv2.INTER_AREA)
            buf.add(final, 0, carry_r, is_first=False, is_terminal=carry_term)
            n_added += 1
            if verbose:
                print(f"  [{ti+1}/{len(paths)}] {p.split('/')[-1]}: +{len(actions)} "
                      f"({n_added} total)")
    finally:
        env.close()
    return n_added


def trace_paths(level: int) -> list[str]:
    return sorted(glob.glob(f"tmp/mc_trace/level{level}/*.npz"))


def iter_trace_frames(paths: list[str], size: int = 128, stride: int = 1,
                      max_traces: int | None = None):
    """Stream ``(frame_u8 (size,size,3), ram_u8 (2048,), trace_id)`` one frame at a
    time by replaying traces through the single shared emulator.

    Accumulates nothing, so it scales to the full trace set (128GB of frames at
    native res if you kept them all — see dreamer.pretrain_ae.materialize_traces,
    which streams this into a disk memmap). ``stride`` records every Nth decision
    frame (steps all, yields some). The number of frames a trace yields is
    ``ceil(len(actions)/stride)``, so callers can size storage up-front from the
    action lengths without replaying.
    """
    if max_traces:
        paths = paths[:max_traces]
    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE, render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    try:
        for ti, p in enumerate(paths):
            z = np.load(p, allow_pickle=True)
            actions = np.asarray(z["actions"], dtype=np.uint8)
            rewind_state(env, bytes(z["initial_state"]))
            for j, act in enumerate(actions):
                for _ in range(SKIP):
                    env.step(act.copy())
                if j % stride:
                    continue
                frame = cv2.resize(env.em.get_screen(), (size, size),
                                   interpolation=cv2.INTER_AREA).astype(np.uint8)
                yield frame, env.unwrapped.get_ram().astype(np.uint8), ti
    finally:
        env.close()


def replay_traces(paths: list[str], size: int = 128, stride: int = 1,
                  max_traces: int | None = None, verbose: bool = True):
    """Replay traces → (frames, states, trace_ids), holding everything in memory.

    frames    uint8  (M, size, size, 3)   post-SKIP screens, resized
    states    float32(M, 3)               [player_x, player_y, n_enemies] from RAM
    trace_ids int    (M,)                  which trace each frame came from

    Fine for the small collections the verify gates use; for the full trace set at
    native res stream via :func:`iter_trace_frames` instead (won't fit in RAM).
    """
    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []
    trace_ids: list[int] = []
    for frame, ram, ti in iter_trace_frames(paths, size, stride, max_traces):
        s = state_from_ram(ram)
        n_en = float((s[26:90].reshape(16, 4)[:, 0] != 0).sum())
        frames.append(frame)
        states.append(np.array([s[3], s[4], n_en], dtype=np.float32))
        trace_ids.append(ti)
    if verbose:
        print(f"  replayed {len(np.unique(trace_ids))} traces → {len(frames)} frames")
    return (np.asarray(frames, dtype=np.uint8),
            np.asarray(states, dtype=np.float32),
            np.asarray(trace_ids, dtype=np.int64))


def split_by_trace(frames, states, trace_ids, n_eval_traces: int, seed: int = 0):
    """Hold out whole traces for eval (honest generalization split)."""
    uniq = np.unique(trace_ids)
    rng = np.random.default_rng(seed)
    eval_traces = set(rng.choice(uniq, size=min(n_eval_traces, len(uniq) - 1),
                                 replace=False).tolist())
    ev = np.array([t in eval_traces for t in trace_ids])
    return (frames[~ev], states[~ev]), (frames[ev], states[ev])


# ── Verification gate ────────────────────────────────────────────────────────

def _smoke(level: int, size: int, max_traces: int, stride: int) -> None:
    import imageio

    paths = trace_paths(level)
    print(f"[collect] level {level}: {len(paths)} traces found; "
          f"replaying {min(max_traces, len(paths))} (stride={stride})")
    frames, states, tids = replay_traces(paths, size=size, stride=stride,
                                         max_traces=max_traces)
    print(f"[collect] {len(frames)} frames, {len(np.unique(tids))} traces")
    print(f"  player_x range [{states[:,0].min():.0f},{states[:,0].max():.0f}]  "
          f"player_y range [{states[:,1].min():.0f},{states[:,1].max():.0f}]  "
          f"n_enemies mean {states[:,2].mean():.2f} max {states[:,2].max():.0f}")
    (xtr, _), (xev, _) = split_by_trace(frames, states, tids, n_eval_traces=2)
    print(f"  per-trace split: train={len(xtr)} eval={len(xev)} frames")

    # diversity check: evenly-spaced frames across the whole replay, as a GIF
    from dreamer import out_path
    idx = np.linspace(0, len(frames) - 1, 120, dtype=int)
    montage = out_path(f"collect_L{level}_montage.gif")
    imageio.mimsave(montage, list(frames[idx]), duration=120, loop=0)
    for n, i in [("start", 0), ("mid", len(frames)//2), ("late", len(frames)-1)]:
        imageio.imwrite(out_path(f"collect_L{level}_{n}.png"), frames[i])
    print(f"  {montage} + start/mid/late PNGs "
          f"← should show DIFFERENT parts of the level, not one stuck screen")


def main() -> None:
    p = argparse.ArgumentParser(description="MC-trace frame collector")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--max_traces", type=int, default=4)
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()
    if args.smoke:
        _smoke(args.level, args.size, args.max_traces, args.stride)
    else:
        p.error("nothing to do; pass --smoke")


if __name__ == "__main__":
    main()
