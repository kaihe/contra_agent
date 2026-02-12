"""
Contra (NES) Custom Wrapper for Stable-Baselines3
==================================================

Single wrapper: ContraWrapper
  - Discrete 7-action space (NOOP, Fire, Left, Right, Up, Down, Jump)
  - Frame skip (4x) + no-op release frame = 5 inner steps per agent decision
  - Reward shaping: position delta, score delta, death penalty, terminal bonuses
  - Max-pool last 2 raw frames (flicker removal)
  - Grayscale + resize to 84x84 + 4-frame stack -> (84, 84, 4)
  - Optional monitor recording of raw frames
"""

import cv2
import gymnasium as gym
import numpy as np


# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Index:        0  1     2       3      4   5     6     7      8
#                                                                   Nick
ACTION_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP       N
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Fire (B)   F
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: Left       L
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 3: Right      R
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4: Up         U
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5: Down       D
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6: Jump (A)   J
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # 7: Jump+Right JR
    [0, 0, 0, 0, 0, 0, 1, 0, 1],  # 8: Jump+Left  JL
]
ACTION_NAMES = ["N", "F", "L", "R", "U", "D", "J", "JR", "JL"]


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


# =============================================================================
# UNIFIED WRAPPER: REWARD SHAPING + FRAME SKIP + STACKING
# =============================================================================

class ContraWrapper(gym.Wrapper):
    """Single wrapper combining reward shaping, frame skip, and frame stacking.

    Each agent step:
      1. Map discrete action to NES multi-binary buttons
      2. Repeat action for `skip` emulator frames (reward shaping each frame)
      3. Send one no-op frame for button release
      4. Max-pool last 2 raw RGB frames (flicker removal)
      5. Grayscale + resize to 84x84, push into frame stack

    Observation: (84, 84, stack) uint8 channels-last for SB3 CnnPolicy.
    """

    def __init__(self, env, monitor=None, reset_round=True, random_start_frames=0,
                 warmup_frames=120, skip=4, stack=4):
        super().__init__(env)
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self._action_table = np.array(ACTION_TABLE, dtype=env.action_space.dtype)
        self.monitor = monitor
        self.reset_round = reset_round
        self.random_start_frames = random_start_frames
        self.warmup_frames = warmup_frames
        self.skip = skip
        self.stack = stack

        # Discrete action space
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))

        # Frame stack: (stack, 84, 84) internal, exposed as (84, 84, stack)
        self.states = np.zeros((stack, 84, 84), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, stack), dtype=np.uint8
        )

        # Reward tracking state
        self.prev_xscroll = 0
        self.prev_score = 0
        self.prev_lives = 0
        self.start_lives = 2
        self.max_x_reached = 0
        self.total_timesteps = 0
        self.max_episode_steps = 2000
        self.idle_steps = 0  # consecutive steps with no xscroll change

        # Episode stats for logging
        self.episode_score = 0
        self.episode_reward = 0
        self.episode_distance_reward = 0
        self.episode_score_reward = 0
        self.episode_end_reason = ""

    def _get_obs(self):
        """Return (84, 84, stack) channels-last."""
        return np.transpose(self.states, (1, 2, 0))

    def _compute_reward(self, info):
        """Compute shaped reward from a single emulator frame's info."""
        reward = 0.0

        curr_xscroll = info.get("xscroll", self.prev_xscroll)
        curr_score = info.get("score", 0)
        curr_lives = info.get("lives", 0)

        self.episode_score = curr_score

        # Idle detection: max_x_reached not pushed for 10 steps
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
        multi_action = self._action_table[action]
        actions = [multi_action] * self.skip + [self._no_op]
        last_two = [None, None]
        done = False

        # Run action frames + 1 no-op release frame
        for act in actions:
            state, _, term, trunc, info = self.env.step(act)
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
