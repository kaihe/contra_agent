"""Component 1 — Contra → Dreamer env adapter.

Dreamer wants *plain RGB current frames*, not the PPO frame-stack trick
(R(t),G(t-1),B(t-3) channel slicing in contra_wrapper). That trick smuggles
motion into a memoryless CNN; Dreamer's RSSM is the memory, so the world model
should reconstruct honest single-moment frames. We therefore reuse ContraWrapper
purely for its reward / frame-skip / terminal logic and replace the observation
with the raw emulator screen, resized to `size`×`size` RGB.

Interface (gymnasium-style):
    obs        uint8 (size, size, 3), channels-last, 0..255   — current frame
    reward     float                                          — ContraWrapper reward
    terminated bool   — real MDP terminal (death / win / game over)
    truncated  bool   — timeout only (future should be bootstrapped, continue=1)
    action     int    — index into the discrete Contra action space

`terminated` vs `truncated` is exactly the Dreamer "continue" signal: the world
model's continue head learns continue = 1 - terminated (a timeout does NOT zero
the future).

Verification gate (run `python -m dreamer.envs --smoke`):
  * a random rollout completes,
  * obs is uint8 (size,size,3) in [0,255],
  * a GIF is written to tmp/ that visibly looks like Contra gameplay.
"""

from __future__ import annotations

import argparse

import cv2
import gymnasium as gym
import numpy as np
import stable_retro as retro

from contra.action_space import DEFAULT as ACTION_SPACE
from ppo.contra_wrapper import RandomStateWrapper, create_env

GAME = "Contra-Nes"
NUM_ACTIONS = ACTION_SPACE.num_actions
ACTION_NAMES = list(ACTION_SPACE.names)


class DreamerObs(gym.Wrapper):
    """Replace the wrapped env's observation with a plain resized RGB frame.

    The inner ContraWrapper still runs the frame-skip loop and computes
    reward/terminal; we discard its channel-sliced obs and read the current
    screen straight from the emulator after each step.
    """

    def __init__(self, env, size: int = 128):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

    def _frame(self) -> np.ndarray:
        # Raw NES screen (H, W, 3) uint8 → size×size RGB. INTER_AREA is the
        # right filter for downscaling (avoids the aliasing of INTER_LINEAR).
        screen = self.unwrapped.em.get_screen()
        return cv2.resize(screen, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._frame(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._frame(), reward, terminated, truncated, info


def make_contra_env(
    level: int = 1,
    size: int = 128,
    states: list[str] | None = None,
    max_episode_steps: int = 10000,
    reward_weights: dict | None = None,
):
    """Build a Dreamer-ready Contra env: discrete actions, plain RGB obs.

    `states`: optional list of .state files sampled uniformly per reset
    (multi-state training). When None, starts from the level's boot state.

    NOTE: stable_retro allows only ONE emulator per process — close this env
    before constructing another (see CLAUDE.md).
    """
    boot_state = f"Level{level}"
    env = retro.make(
        game=GAME,
        state=boot_state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if states:
        env = RandomStateWrapper(env, states=states)
    # stack=3 is required by ContraWrapper (HISTORY_OFFSETS), but DreamerObs
    # discards that observation, so the value is irrelevant here.
    env = create_env(
        env,
        stack=3,
        level=level,
        max_episode_steps=max_episode_steps,
        reward_weights=reward_weights,
    )
    return DreamerObs(env, size=size)


# ── Verification gate ────────────────────────────────────────────────────────

def _smoke(level: int, size: int, steps: int, seed: int) -> None:
    """Random rollout → assert obs invariants, print stats, dump a GIF."""
    import imageio

    rng = np.random.default_rng(seed)
    env = make_contra_env(level=level, size=size)
    try:
        obs, _ = env.reset(seed=seed)
        assert obs.shape == (size, size, 3), f"bad obs shape {obs.shape}"
        assert obs.dtype == np.uint8, f"bad obs dtype {obs.dtype}"
        assert 0 <= obs.min() and obs.max() <= 255, "obs out of [0,255]"

        frames = [obs]
        total_r, ep_returns, ep_steps = 0.0, [], 0
        end_reason = ""
        for t in range(steps):
            a = int(rng.integers(NUM_ACTIONS))
            obs, r, term, trunc, info = env.step(a)
            frames.append(obs)
            total_r += r
            ep_steps += 1
            if term or trunc:
                end_reason = info.get("episode_end_reason", "?")
                ep_returns.append(info.get("episode_reward", total_r))
                obs, _ = env.reset()
                frames.append(obs)

        from dreamer import out_path
        gif_path = out_path(f"env_smoke_L{level}.gif")
        imageio.mimsave(gif_path, frames[:600], duration=40, loop=0)

        print(f"[smoke] level={level} size={size} steps={steps}")
        print(f"  obs: shape={obs.shape} dtype={obs.dtype} "
              f"range=[{int(obs.min())},{int(obs.max())}]")
        print(f"  actions: {NUM_ACTIONS} -> {ACTION_NAMES}")
        print(f"  total reward over rollout: {total_r:.2f}")
        print(f"  episodes finished: {len(ep_returns)} "
              f"(last end_reason={end_reason or 'none — no episode ended'})")
        print(f"  GIF: {gif_path}  ← eyeball this: it must look like Contra")
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Contra→Dreamer env adapter")
    p.add_argument("--smoke", action="store_true", help="Run the verification gate")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.smoke:
        _smoke(args.level, args.size, args.steps, args.seed)
    else:
        p.error("nothing to do; pass --smoke")


if __name__ == "__main__":
    main()
