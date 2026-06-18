"""Contra (NES) gym wrapper: discrete actions, frame skip/stack, reward shaping."""

import gzip
import json
import os
import zipfile

import cv2
import gymnasium as gym
import numpy as np

from contra.action_space import DEFAULT as ACTION_SPACE
from contra.events import is_gameplay
from contra.reward import (
    DEFAULT_REWARD_WEIGHTS,
    reward_components,
    xscroll,
)


# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
#
# Flat discrete action space: each action is a length-9 NES button vector and
# the agent picks an index (gym Discrete). The action list + frame skip live in
# contra/action_space.py so mc_search and PPO share one canonical action space.
# Mirrored into module globals because apply_config() may override them at
# load time (a model can embed the exact actions it was trained with).
ACTIONS = [list(a) for a in ACTION_SPACE.actions]
ACTION_NAMES = list(ACTION_SPACE.names)
NUM_ACTIONS = len(ACTIONS)
ACTION_SKIP = ACTION_SPACE.skip

# Three RGB-sliced history channels: R(t), G(t-1), B(t-3).
RGB_CHANNELS = 3
HISTORY_OFFSETS = [0, 1, 3]
BUFFER_FRAMES = max(HISTORY_OFFSETS) + 1


def save_config_to_model(
    model_path: str,
    skip: int | None = None,
    stack: int = 3,
    train_config: dict | None = None,
) -> None:
    """Embed contra_config.json into an SB3 model .zip file."""
    if skip is None:
        skip = ACTION_SKIP
    config = {"actions": dict(zip(ACTION_NAMES, ACTIONS)),
              "skip": skip, "stack": stack,
              "history_offsets": HISTORY_OFFSETS}
    if train_config is not None:
        config["train_config"] = train_config
    with zipfile.ZipFile(model_path, "a") as zf:
        zf.writestr("contra_config.json", json.dumps(config, indent=2))


def load_config_from_model(model_path: str) -> dict | None:
    """Extract contra_config.json from an SB3 model .zip, or None if missing."""
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "contra_config.json" in zf.namelist():
                return json.loads(zf.read("contra_config.json"))
    except (zipfile.BadZipFile, KeyError):
        pass
    return None


def apply_config(config: dict) -> None:
    """Override the global action list with values from config."""
    global ACTIONS, ACTION_NAMES, NUM_ACTIONS, ACTION_SKIP
    actions = config["actions"]  # {name: vector}
    ACTION_NAMES = list(actions.keys())
    ACTIONS      = [list(v) for v in actions.values()]
    NUM_ACTIONS  = len(ACTIONS)
    ACTION_SKIP  = int(config.get("skip", ACTION_SKIP))


class Monitor:
    """Record raw RGB frames and/or display live via pygame."""

    def __init__(self, width, height, saved_path=None, render=False, skip=ACTION_SKIP):
        self.render = render
        self.saved_path = saved_path
        self.frames = [] if saved_path else None
        self.skip = skip
        if render:
            import pygame
            self._pygame = pygame
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Contra")
            self._clock = pygame.time.Clock()

    def record(self, image_array):
        if self.frames is not None:
            self.frames.append(image_array.copy())
        if self.render:
            pg = self._pygame
            surf = pg.surfarray.make_surface(image_array.swapaxes(0, 1))
            self.screen.blit(surf, (0, 0))
            pg.display.flip()
            self._clock.tick(120)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.render = False

    def close(self):
        if self.frames is not None and self.saved_path:
            import imageio
            frames = self.frames[::self.skip]
            duration = round(1000 * self.skip / 60 / 2)
            imageio.mimsave(self.saved_path, frames, duration=duration, loop=1)
        if self.render:
            self._pygame.quit()


def process_frame(frame):
    """RGB → resized RGB 84×84, shape (84, 84, 3) uint8."""
    return cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)


class RandomStateWrapper(gym.Wrapper):
    """On each reset, load a randomly chosen savestate from a list of .state files.

    Anchor states are mid-gameplay snapshots, so episodes start directly in
    active play (no title screen). Used for multi-state training: sampling
    starts across the level removes the exploration bottleneck of always
    replaying from x=0.
    """

    def __init__(self, env, states):
        super().__init__(env)
        self.state_names = list(states)
        self.state_data = []
        for path in states:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Savestate not found: {path}")
            with gzip.open(path, "rb") as f:
                self.state_data.append(f.read())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        idx = np.random.randint(len(self.state_data))
        self.unwrapped.em.set_state(self.state_data[idx])
        self.unwrapped.data.update_ram()
        # Step once with no-op to sync the observation with the loaded state.
        no_op = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        obs, _, _, _, info = self.env.step(no_op)
        return obs, info


class ContraWrapper(gym.Wrapper):
    """Reward shaping + frame skip + history sampling for Contra NES.

    Observation: (84, 84, stack) uint8 channels-last for SB3 CnnPolicy.
                 Channels are R(t), G(t-1), B(t-3).
    Actions:     Discrete(NUM_ACTIONS) — index into the flat ACTIONS vector list.
    """

    def __init__(self, env, monitor=None, random_start_frames=0,
                 warmup_frames=120, skip=None, stack=3, level=1,
                 max_episode_steps=10000,
                 reward_weights=None):
        super().__init__(env)
        weights = DEFAULT_REWARD_WEIGHTS.copy()
        if reward_weights is not None:
            unknown = set(reward_weights) - set(weights)
            if unknown:
                raise ValueError(f"Unknown reward weight(s): {sorted(unknown)}")
            weights.update(reward_weights)
        self.reward_weights = weights
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self._actions = np.array(ACTIONS, dtype=env.action_space.dtype)
        self.monitor = monitor
        self.random_start_frames = random_start_frames
        self.warmup_frames = warmup_frames
        self.skip = ACTION_SKIP if skip is None else skip
        self.stack = stack
        self.level = level
        self.max_episode_steps = max_episode_steps
        if stack != len(HISTORY_OFFSETS):
            raise ValueError(
                f"stack={stack} must match len(HISTORY_OFFSETS)={len(HISTORY_OFFSETS)}"
            )

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        # Small ring buffer for recent RGB 84×84 frames.
        self._buf = np.zeros((BUFFER_FRAMES, 84, 84, RGB_CHANNELS), dtype=np.uint8)
        self._buf_pos = 0
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, stack), dtype=np.uint8
        )

        self.prev_ram = np.zeros(2048, dtype=np.uint8)
        self.prev_xscroll = 0
        self.max_xscroll = 0
        self.episode_start_x = 0
        self.total_timesteps = 0
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.ep = {
            "reward": 0.0,
            "enemy_hp_cost": 0.0,
            "core_broken": 0.0,
            "end_reason": "",
        }

    def _get_obs(self) -> np.ndarray:
        indices = [(self._buf_pos - offset) % BUFFER_FRAMES for offset in HISTORY_OFFSETS]
        channels = [
            self._buf[indices[channel_idx], :, :, channel_idx]
            for channel_idx in range(RGB_CHANNELS)
        ]
        return np.stack(channels, axis=-1)

    def _compute_rewards(self, info, done):
        curr_ram = self.unwrapped.get_ram()
        curr_xscroll = xscroll(curr_ram)
        # High-water mark of progress. Used for episode_delta_x because a
        # levelup resets xscroll to ~0 on the winning frame; tracking the max
        # keeps the real furthest-right position instead of that reset value.
        self.max_xscroll = max(self.max_xscroll, curr_xscroll)
        timed_out = not done and self.total_timesteps >= self.max_episode_steps

        end_reason = ""
        rewards = reward_components(
            self.prev_ram,
            curr_ram,
            self.reward_weights,
            self.prev_xscroll,
            timed_out,
        )
        events = self._events_from_rewards(rewards)

        # Death is terminal (episodic-life trick): ending the episode makes a
        # death cost all remaining future reward, which is a far stronger
        # signal than the flat penalty alone. Levelup is checked first so a
        # simultaneous levelup+death still counts as a win.
        if rewards["levelup"] != 0.0:
            done = True
            end_reason = "win"
        elif rewards["player_die"] != 0.0:
            done = True
            end_reason = "death"
        elif done:
            end_reason = "game_over"
        elif timed_out:
            done = True
            end_reason = "time_out"

        if end_reason:
            self.ep["end_reason"] = end_reason

        self.prev_xscroll = curr_xscroll
        self.prev_ram = curr_ram.copy()

        return events, rewards, done

    def _events_from_rewards(self, rewards):
        events = {}
        for key, reward in rewards.items():
            weight = self.reward_weights[key]
            events[key] = reward / weight if weight else 0.0
        return events

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.total_timesteps = 0
        self._reset_episode_stats()

        # Warm up until active gameplay (title fade-in for boot states); anchor
        # savestates are already mid-gameplay so this loop exits immediately.
        for _ in range(self.warmup_frames):
            if is_gameplay(self.unwrapped.get_ram()):
                break
            observation, _, _, _, info = self.env.step(self._no_op)
            if self.monitor:
                self.monitor.record(observation)

        if self.random_start_frames > 0:
            for _ in range(np.random.randint(0, self.random_start_frames + 1)):
                observation, _, _, _, info = self.env.step(self._no_op)
                if self.monitor:
                    self.monitor.record(observation)

        # Snapshot all prev-state from the same frame after warmup/random
        ram = self.unwrapped.get_ram()
        self.prev_ram = ram.copy()
        self.prev_xscroll = xscroll(ram)
        self.max_xscroll = self.prev_xscroll
        self.episode_start_x = self.prev_xscroll

        self._buf[:] = process_frame(observation)
        self._buf_pos = 0

        return self._get_obs(), info

    def step(self, action):
        nes_action = self._actions[int(action)]
        done = False
        states = []

        for i in range(self.skip):
            act = nes_action
            state, _, term, trunc, info = self.env.step(act)
            if self.monitor:
                self.monitor.record(state)
            states.append(state)
            if term or trunc:
                done = True
                break

        self.total_timesteps += 1
        events, rewards, done = self._compute_rewards(info, done)
        reward = sum(rewards.values())

        self.ep["reward"] += reward
        self.ep["enemy_hp_cost"] += events["enemy_hp"]
        self.ep["core_broken"] += events.get("core_broken", 0.0)

        # Max-pool the last two raw frames to defeat NES sprite flicker.
        pooled = np.maximum.reduce(states[-2:])
        self._buf_pos = (self._buf_pos + 1) % BUFFER_FRAMES
        self._buf[self._buf_pos] = process_frame(pooled)

        if done:
            info.update({
                "episode_delta_x": self.max_xscroll - self.episode_start_x,
                "episode_enemy_hp_cost": self.ep["enemy_hp_cost"],
                "episode_core_broken": self.ep["core_broken"],
                "episode_reward": self.ep["reward"],
                "episode_end_reason": self.ep["end_reason"],
                "episode_steps":    self.total_timesteps,
            })

        # The step cap is an artificial cutoff, not an MDP terminal: report it as
        # truncation so PPO bootstraps V(s') instead of zeroing the future. Death
        # /win/game_over are real terminals.
        truncated = done and self.ep["end_reason"] == "time_out"
        terminated = done and not truncated
        return self._get_obs(), reward, terminated, truncated, info


def create_env(env, monitor=None, random_start_frames=0, warmup_frames=120,
               skip=None, stack=3, level=1, max_episode_steps=10000,
               reward_weights=None):
    """Wrap a retro env with reward shaping + frame skip + history sampling."""
    return ContraWrapper(env, monitor=monitor,
                         random_start_frames=random_start_frames,
                         warmup_frames=warmup_frames,
                         skip=skip, stack=stack, level=level,
                         max_episode_steps=max_episode_steps,
                         reward_weights=reward_weights)
