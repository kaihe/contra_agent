"""
Contra Force Custom Wrapper for Stable-Baselines3
==================================================

- Fixed frame skip for all actions
- Frame stacking with RGB channel extraction
- Reward structure (total range clamped to -10..10):
  - Speed (tanh): y = 2*tanh(0.8*(speed-1)), range (-2, 2)
  - Score (exponential): approaches max 8, fast change in (0, 50)
  - Death: fixed -8
- Episode ends on game over (all lives lost)
"""

import collections
import time
import math

import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ContraWrapper(gym.Wrapper):
    """Custom wrapper for Contra Force NES."""

    def __init__(self, env, reset_round=True, rendering=False, random_start_frames=0, render_fps=15):
        super().__init__(env)
        self.env = env

        # Frame stacking: 9 frames, extract every 3rd for RGB channels
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Frame skip: repeat action for this many frames
        self.num_step_frames = 4

        # Reward parameters
        self.death_penalty = -8.0       # Fixed death penalty
        self.score_decay_rate = 0.05    # Controls exponential curve for score (fast change in 0-50)
        self.score_max_reward = 8.0     # Max score reward (asymptote)

        self.total_timesteps = 0

        # Track previous state
        self.prev_score = 0
        self.prev_lives = 2
        self.prev_x_pos = 0
        self.wrap_count = 0
        self.max_x_reached = 0
        self.prev_max_x = 0

        # Episode stats for logging
        self.episode_distance = 0
        self.episode_score = 0
        self.episode_reward = 0

        # Observation space: downsampled RGB
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering
        self.random_start_frames = random_start_frames
        self.render_delay = 1.0 / render_fps  # Delay per wrapper step

        if self.rendering:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            self.im = None

    def _preprocess_frame(self, frame):
        """Downsample frame to 84x84."""
        # NES resolution is 256x224, downsample to 84x84
        h, w = frame.shape[:2]
        new_h, new_w = 84, 84
        h_step = h // new_h
        w_step = w // new_w
        return frame[::h_step, ::w_step, :][:new_h, :new_w, :]

    def _stack_observation(self):
        """Extract frames at [2, 5, 8] with R/G/B channel extraction."""
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i % 3] for i in range(3)], axis=-1
        )

    def _calculate_speed_reward(self, speed):
        """
        Calculate speed reward using tanh curve.
        y = 2 * tanh(0.8 * (speed - 1.0))
        Output range: (-2, 2)
        Key points:
          speed=-2 -> ~-1.8
          speed= 0 -> ~-1.3 (negative, penalizes stalling)
          speed= 2 -> ~+1.3
          speed= 5 -> ~+2.0 (near max)
        """
        return 2.0 * np.tanh(0.8 * (speed - 1.0))

    def _calculate_score_reward(self, score_diff):
        """
        Calculate score reward using exponential formula.
        Approaches max value of 2 asymptotically.
        Changes fast in range (0, 50).
        Formula: 2 * (1 - exp(-score_diff * decay_rate))
        """
        if score_diff <= 0:
            return 0.0

        # Exponential saturation: fast rise for small values, asymptotes at max
        reward = self.score_max_reward * (1 - math.exp(-score_diff * self.score_decay_rate))
        return reward

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Initialize tracking variables
        self.prev_score = info.get("score", 0)
        self.prev_lives = info.get("lives", 2)
        self.prev_x_pos = info.get("x_pos", 0)
        self.wrap_count = 0
        self.max_x_reached = self.prev_x_pos
        self.prev_max_x = self.prev_x_pos
        self.total_timesteps = 0

        # Reset episode stats
        self.episode_distance = 0
        self.episode_score = 0
        self.episode_reward = 0

        # Clear frame stack and fill with initial observation
        self.frame_stack.clear()
        processed = self._preprocess_frame(observation)
        for _ in range(self.num_frames):
            self.frame_stack.append(processed)

        # Random startup freeze
        if self.random_start_frames > 0:
            no_op = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
            freeze_frames = np.random.randint(0, self.random_start_frames + 1)
            for _ in range(freeze_frames):
                observation, _, _, _, info = self.env.step(no_op)
                self.frame_stack.append(self._preprocess_frame(observation))
                if self.rendering:
                    if self.im is None:
                        self.im = self.ax.imshow(observation)
                    else:
                        self.im.set_data(observation)
                    self.fig.canvas.flush_events()
                    plt.pause(0.001)

        return self._stack_observation(), info

    def step(self, action):
        done = False

        for i in range(self.num_step_frames):
            obs, reward, term, trunc, info = self.env.step(action)
            self.frame_stack.append(self._preprocess_frame(obs))

            if term or trunc:
                done = True
                break

        # Render only the final frame of the skip (controlled by fps)
        if self.rendering:
            if self.im is None:
                self.im = self.ax.imshow(obs)
            else:
                self.im.set_data(obs)
            self.fig.canvas.flush_events()
            plt.pause(self.render_delay)

        # Get current state
        curr_score = info.get("score", 0)
        curr_lives = info.get("lives", self.prev_lives)
        curr_x_pos = info.get("x_pos", self.prev_x_pos)

        self.total_timesteps += 1

        # Calculate absolute x position (handle wrap at 256)
        diff = curr_x_pos - self.prev_x_pos
        if diff < -128:
            self.wrap_count += 1  # Wrapped forward
        elif diff > 128:
            self.wrap_count -= 1  # Wrapped backward

        abs_x = self.wrap_count * 256 + curr_x_pos

        # Update max_x if we've progressed
        if abs_x > self.max_x_reached:
            progress = abs_x - self.max_x_reached
            self.episode_distance += progress
            self.max_x_reached = abs_x

        # Speed = max_x diff between current and previous step
        speed = self.max_x_reached - self.prev_max_x
        self.prev_max_x = self.max_x_reached

        # === CALCULATE REWARDS ===
        custom_reward = 0.0

        # Speed reward: tanh curve, range (-2, 8)
        speed_reward = self._calculate_speed_reward(speed)
        custom_reward += speed_reward

        # Score reward: exponential, max 2
        score_diff = curr_score - self.prev_score
        if score_diff > 0:
            self.episode_score += score_diff
            score_reward = self._calculate_score_reward(score_diff)
            custom_reward += score_reward

        # Death penalty: fixed -8
        if curr_lives < self.prev_lives:
            custom_reward += self.death_penalty

        # Game over: end episode when all lives lost
        if curr_lives == 0:
            done = True

        # Update previous state
        self.prev_score = curr_score
        self.prev_lives = curr_lives
        self.prev_x_pos = curr_x_pos

        self.episode_reward += custom_reward

        # Add episode stats to info for logging (on episode end)
        if done:
            info["episode_distance"] = self.episode_distance
            info["episode_score"] = self.episode_score
            info["episode_reward"] = self.episode_reward
            info["episode_steps"] = self.total_timesteps

        if not self.reset_round:
            done = False

        return self._stack_observation(), custom_reward, done, False, info
