"""
Contra Force Custom Wrapper for Stable-Baselines3
==================================================

- Fixed frame skip for all actions
- Frame stacking with RGB channel extraction
- Reward based on score and survival
- Episode ends on death (lives == 0)
"""

import collections
import time

import gymnasium as gym
import numpy as np


class ContraWrapper(gym.Wrapper):
    """Custom wrapper for Contra Force NES."""

    def __init__(self, env, reset_round=True, rendering=False, random_start_frames=0):
        super().__init__(env)
        self.env = env

        # Frame stacking: 9 frames, extract every 3rd for RGB channels
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Frame skip: repeat action for this many frames
        self.num_step_frames = 4

        # Reward scaling
        self.score_coeff = 1.0  # Reward for score increase
        self.death_penalty = -50.0  # Penalty for dying

        self.total_timesteps = 0

        # Track previous state
        self.prev_score = 0
        self.prev_lives = 2  # Contra Force starts with 2 lives

        # Observation space: downsampled RGB
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering
        self.random_start_frames = random_start_frames

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

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Initialize tracking variables
        self.prev_score = info.get("score", 0)
        self.prev_lives = info.get("lives", 2)
        self.total_timesteps = 0

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
                    self.env.render()
                    time.sleep(0.01)

        return self._stack_observation(), info

    def step(self, action):
        total_reward = 0
        done = False

        for _ in range(self.num_step_frames):
            obs, reward, term, trunc, info = self.env.step(action)
            self.frame_stack.append(self._preprocess_frame(obs))

            if self.rendering:
                self.env.render()
                time.sleep(0.01)

            if term or trunc:
                done = True
                break

        # Get current state
        curr_score = info.get("score", 0)
        curr_lives = info.get("lives", self.prev_lives)

        self.total_timesteps += self.num_step_frames

        # Calculate custom reward
        custom_reward = 0

        # Score reward
        score_diff = curr_score - self.prev_score
        if score_diff > 0:
            custom_reward += self.score_coeff * score_diff

        # Death penalty
        if curr_lives < self.prev_lives:
            custom_reward += self.death_penalty

        # Game over (lives == 0)
        if curr_lives == 0 or term:
            done = True

        # Update previous state
        self.prev_score = curr_score
        self.prev_lives = curr_lives

        # Normalize reward
        normalized_reward = custom_reward * 0.01

        if not self.reset_round:
            done = False

        return self._stack_observation(), normalized_reward, done, False, info
