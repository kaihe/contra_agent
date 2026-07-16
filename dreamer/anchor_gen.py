"""Anchor-branched random rollouts — whole-level failure data for the world model.

Win traces cover the level but never fail; a random policy from the level START
only ever explores the first screen. This branches biased-random rollouts from
save-state ANCHORS taken along the win traces (every `stride` actions), so the
failure / off-policy data the continue head and dynamics need spans the WHOLE
level. Emulator save states are ~KB (vs ~192KB per 256px frame), so snapshotting
anchors from many traces is cheap — the frames are generated on the fly by
branching, never stored.

  snapshot_anchors   — replay traces once, grab em.get_state() every `stride` steps
  branch_into_buffer — set_state to a random anchor, run a biased-random rollout to
                       death, and append the transitions to a Component-2 buffer
                       (same reward/terminal convention as collect.fill_buffer_from_traces)

The rollout policy is forward-biased (mostly Right+Fire) on purpose: pure random
flails and dies in ~2 frames, teaching "twitch → death"; a forward bias makes real
progress and hits real hazards before dying.
"""

from __future__ import annotations

import cv2
import numpy as np
import stable_retro as retro

from contra.action_space import DEFAULT as ACTION_SPACE
from contra.replay import GAME, SKIP, rewind_state
from contra.reward import reward_components, xscroll
from dreamer.collect import REWARD_WEIGHTS

_RF = list(ACTION_SPACE.names).index("RF")          # Right+Fire action index
_N_ACTIONS = ACTION_SPACE.num_actions


def _make_env():
    return retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE, render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )


def snapshot_anchors(paths, stride: int = 20, max_traces: int | None = None,
                     verbose: bool = True) -> list[bytes]:
    """Replay traces, snapshotting an emulator save-state every `stride` actions.

    Returns a flat list of save-state `bytes` (each ~KB) spanning the whole level
    across all the given traces — the seeds to branch rollouts from.
    """
    if max_traces:
        paths = paths[:max_traces]
    env = _make_env()
    env.reset()
    anchors: list[bytes] = []
    try:
        for p in paths:
            z = np.load(p, allow_pickle=True)
            actions = np.asarray(z["actions"], dtype=np.uint8)
            rewind_state(env, bytes(z["initial_state"]))
            for j, act in enumerate(actions):
                if j % stride == 0:
                    anchors.append(env.em.get_state())
                for _ in range(SKIP):
                    env.step(act.copy())
    finally:
        env.close()
    if verbose:
        print(f"[anchor] {len(anchors)} anchors from {len(paths)} traces (every {stride} actions)")
    return anchors


def branch_into_buffer(buf, anchors, n_rollouts: int, max_steps: int = 150,
                       right_frac: float = 0.9, seed: int = 0) -> int:
    """Branch `n_rollouts` biased-random rollouts from random anchors into `buf`.

    Each rollout: set_state to a random anchor, act (mostly Right+Fire) until the
    player dies (or `max_steps`), appending transitions with the DreamerV3
    reward-INTO-observation convention. Returns the number of deaths produced.
    """
    size = buf.obs_shape[0]
    rng = np.random.default_rng(seed)
    env = _make_env()
    env.reset()
    deaths = 0
    try:
        for _ in range(n_rollouts):
            env.em.set_state(anchors[int(rng.integers(len(anchors)))])
            prev_ram = env.unwrapped.get_ram().copy()
            prev_x = xscroll(prev_ram)
            carry_r, carry_term = 0.0, False
            for t in range(max_steps):
                a_idx = _RF if rng.random() < right_frac else int(rng.integers(_N_ACTIONS))
                screen = env.em.get_screen()                      # pre-action frame
                img = cv2.resize(screen, (size, size), interpolation=cv2.INTER_AREA)
                buf.add(img, a_idx, carry_r, is_first=(t == 0), is_terminal=carry_term)
                if carry_term:                                    # just added the terminal obs
                    deaths += 1
                    break
                act = np.asarray(ACTION_SPACE.actions[a_idx], dtype=np.uint8)
                for _ in range(SKIP):
                    env.step(act.copy())
                curr_ram = env.unwrapped.get_ram()
                rewards = reward_components(prev_ram, curr_ram, REWARD_WEIGHTS, prev_x, False)
                if rewards.get("levelup", 0.0) != 0.0 and "progress" in rewards:
                    rewards["progress"] = 0.0                     # kill the xscroll-reset spike
                carry_r = float(sum(rewards.values()))
                carry_term = rewards["player_die"] != 0.0 or rewards["levelup"] != 0.0
                prev_ram = curr_ram.copy()
                prev_x = xscroll(curr_ram)
    finally:
        env.close()
    return deaths
