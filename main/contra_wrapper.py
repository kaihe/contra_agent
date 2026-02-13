"""
Contra (NES) Custom Wrapper for Stable-Baselines3
==================================================

Single wrapper: ContraWrapper
  - MultiBinary(6) action space: [B, UP, DOWN, LEFT, RIGHT, A]
  - Sticky actions: button state persists for `skip` frames (no forced release)
  - Dict observation: {"image": (84,84,4), "prev_action": MultiBinary(6)}
  - Reward shaping: position delta, score delta, death penalty, terminal bonuses
  - Max-pool last 2 raw frames (flicker removal)
  - Grayscale + resize to 84x84 + 4-frame stack
  - Optional monitor recording of raw frames
"""

import cv2
import gymnasium as gym
import numpy as np


# Agent buttons: [B, UP, DOWN, LEFT, RIGHT, A]
# Index:          0  1    2     3     4      5
NUM_BUTTONS = 6
BUTTON_NAMES = ["B", "UP", "DOWN", "LEFT", "RIGHT", "A"]

# Mapping from agent buttons (6) to NES buttons (9)
# NES layout: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
#              0  1     2       3      4   5     6     7      8
AGENT_TO_NES = [0, 4, 5, 6, 7, 8]  # agent index -> NES index


# =============================================================================
# FFMPEG MONITOR
# =============================================================================

class Monitor:
    """Record raw RGB frames and/or display live via pygame."""

    def __init__(self, width, height, saved_path=None, render=False):
        self.render = render
        self.saved_path = saved_path
        self.frames = [] if saved_path else None
        self.screen = None

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
            # Skip frames to speed up: keep every 4th frame at 20ms (~3x real speed)
            frames = self.frames[::4]
            imageio.mimsave(self.saved_path, frames, duration=20)
        if self.render:
            self._pygame.quit()


# =============================================================================
# PREPROCESSING
# =============================================================================

def process_frame(frame):
    """Convert RGB frame to grayscale 84x84, shape (1, 84, 84) as uint8."""
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[None, :, :]  # (1, 84, 84)
    return np.zeros((1, 84, 84), dtype=np.uint8)


def agent_to_nes(action):
    """Convert 6-button agent action to 9-button NES action."""
    nes = np.zeros(9, dtype=np.int8)
    for i, nes_idx in enumerate(AGENT_TO_NES):
        nes[nes_idx] = action[i]
    return nes


# =============================================================================
# UNIFIED WRAPPER: REWARD SHAPING + FRAME SKIP + STACKING
# =============================================================================

class ContraWrapper(gym.Wrapper):
    """Single wrapper combining reward shaping, frame skip, and frame stacking.

    Each agent step:
      1. Convert MultiBinary(6) action to NES 9-button layout
      2. Hold buttons for `skip` emulator frames (sticky — no forced release)
      3. Max-pool last 2 raw RGB frames (flicker removal)
      4. Grayscale + resize to 84x84, push into frame stack

    Observation: Dict with:
      - "image": (84, 84, stack) uint8 channels-last
      - "prev_action": MultiBinary(6) — previous step's button state
    """

    def __init__(self, env, monitor=None, reset_round=True, random_start_frames=0,
                 warmup_frames=120, skip=4, stack=4):
        super().__init__(env)
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self.monitor = monitor
        self.reset_round = reset_round
        self.random_start_frames = random_start_frames
        self.warmup_frames = warmup_frames
        self.skip = skip
        self.stack = stack

        # MultiBinary action space: 6 useful NES buttons
        self.action_space = gym.spaces.MultiBinary(NUM_BUTTONS)

        # Dict observation: image + previous action
        self.states = np.zeros((stack, 84, 84), dtype=np.uint8)
        self.prev_action = np.zeros(NUM_BUTTONS, dtype=np.int8)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(84, 84, stack), dtype=np.uint8
            ),
            "prev_action": gym.spaces.MultiBinary(NUM_BUTTONS),
        })

        # Reward tracking state
        self.prev_xscroll = 0
        self.prev_score = 0
        self.prev_lives = 0
        self.start_lives = 2
        self.max_x_reached = 0
        self.total_timesteps = 0
        self.max_episode_steps = 2000
        self.idle_steps = 0

        # Episode stats for logging
        self.episode_score = 0
        self.episode_reward = 0
        self.episode_distance_reward = 0
        self.episode_score_reward = 0
        self.episode_end_reason = ""

    def _get_obs(self):
        """Return Dict observation."""
        return {
            "image": np.transpose(self.states, (1, 2, 0)),
            "prev_action": self.prev_action.copy(),
        }

    def _compute_reward(self, info):
        """Compute shaped reward from a single emulator frame's info."""
        reward = 0.0

        curr_xscroll = info.get("xscroll", self.prev_xscroll)
        curr_score = info.get("score", 0)
        curr_lives = info.get("lives", 0)

        self.episode_score = curr_score

        # Idle detection: max_x_reached not pushed for 50 steps
        if curr_xscroll > self.max_x_reached:
            self.idle_steps = 0
            self.max_x_reached = curr_xscroll
        else:
            self.idle_steps += 1

        if self.idle_steps > 50:
            reward -= 0.05
        else:
            # Position reward: delta clipped [0, 0.3], in total 3000 points
            pos_delta = curr_xscroll - self.prev_xscroll
            pos_reward = max(min(pos_delta, 3.0), 0) * (1/30)
            self.episode_distance_reward += pos_reward
            reward += pos_reward

            # Score reward: positive delta only, around 100 points
            score_delta = curr_score - self.prev_score
            score_reward = max(score_delta, 0)
            self.episode_score_reward += score_reward
            reward += score_reward

        # Death penalty
        if curr_lives < self.prev_lives:
            reward -= 20

        # Update state
        self.prev_xscroll = curr_xscroll
        self.prev_score = curr_score
        self.prev_lives = curr_lives

        return reward

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.prev_xscroll = info.get("xscroll", 0)
        self.start_lives = info.get("lives", 2)
        self.prev_lives = self.start_lives
        self.prev_score = 0
        self.max_x_reached = self.prev_xscroll
        self.total_timesteps = 0
        self.idle_steps = 0
        self.prev_action = np.zeros(NUM_BUTTONS, dtype=np.int8)

        self.episode_score = 0
        self.episode_reward = 0
        self.episode_distance_reward = 0
        self.episode_score_reward = 0
        self.episode_end_reason = ""

        # Warmup: advance past spawn animation so player is controllable
        for _ in range(self.warmup_frames):
            observation, _, _, _, info = self.env.step(self._no_op)
            if self.monitor:
                self.monitor.record(observation)

        # Re-sync tracking state after warmup
        self.prev_xscroll = info.get("xscroll", 0)
        self.prev_score = info.get("score", 0)
        self.max_x_reached = self.prev_xscroll

        # Random startup freeze (additional, on top of warmup)
        if self.random_start_frames > 0:
            freeze_frames = np.random.randint(0, self.random_start_frames + 1)
            for _ in range(freeze_frames):
                observation, _, _, _, info = self.env.step(self._no_op)
                if self.monitor:
                    self.monitor.record(observation)

        # Initialize frame stack with processed current frame
        processed = process_frame(observation)
        self.states = np.concatenate([processed for _ in range(self.stack)], axis=0)
        return self._get_obs(), info

    def step(self, action):
        # Convert agent 6-button action to NES 9-button layout
        nes_action = agent_to_nes(action)
        last_two = [None, None]
        done = False

        # Sticky: hold the same button state for `skip` frames (no forced release)
        for _ in range(self.skip):
            state, _, term, trunc, info = self.env.step(nes_action)
            if self.monitor:
                self.monitor.record(state)
            last_two[0], last_two[1] = last_two[1], state
            if term or trunc:
                done = True
                break

        # Compute reward once for the entire agent step
        reward = self._compute_reward(info)
        self.total_timesteps += 1

        if not done and self.total_timesteps >= self.max_episode_steps:
            done = True
            self.episode_end_reason = "time_out"
            # penalty larger than fight 3 lives
            reward -= 100
        elif done:
            if info.get("lives", 0) > 0:
                self.episode_end_reason = "win"
                reward += 100
            else:
                self.episode_end_reason = "game_over"
        self.episode_reward += reward

        # Max-pool last 2 raw frames, grayscale + resize, push into stack
        pooled = np.maximum(last_two[0], last_two[1]) if last_two[0] is not None else last_two[1]
        self.states[:-1] = self.states[1:]
        self.states[-1] = process_frame(pooled)[0]

        # Update prev_action for next step's observation
        self.prev_action = np.array(action, dtype=np.int8)

        if done:
            info["episode_max_x"] = self.max_x_reached
            info["episode_score"] = self.episode_score
            info["episode_reward"] = self.episode_reward
            info["episode_steps"] = self.total_timesteps
            info["episode_distance_reward"] = self.episode_distance_reward
            info["episode_score_reward"] = self.episode_score_reward
            info["episode_end_reason"] = self.episode_end_reason

        if not self.reset_round:
            done = False

        return self._get_obs(), reward, done, False, info


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

def create_env(env, monitor=None, reset_round=True, random_start_frames=0, skip=4, stack=4):
    """Wrap a retro env with reward shaping + frame skip + stacking."""
    return ContraWrapper(env, monitor=monitor, reset_round=reset_round,
                         random_start_frames=random_start_frames,
                         skip=skip, stack=stack)
