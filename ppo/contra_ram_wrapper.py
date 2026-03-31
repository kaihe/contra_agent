"""Contra (NES) RAM-based gym wrapper: raw RAM observation, hp-loss + xscroll reward."""

import gymnasium as gym
import numpy as np

# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
DPAD_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: Right
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: Left
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: Down
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 5: Up+Right
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 6: Down+Right
]
BUTTON_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Fire (B)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2: Jump (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: Fire+Jump
]

NUM_DPAD = len(DPAD_TABLE)
NUM_BUTTONS = len(BUTTON_TABLE)

# RAM addresses for 16 enemy HP slots
HP_ADDRS = slice(1400, 1416)   # ram[1400:1416]
HP_INVALID = 240               # values >= 240 mean the slot is inactive
RAM_SIZE = 2048


def _hp_loss_step(prev_hp: np.ndarray, curr_hp: np.ndarray) -> float:
    """Compute total HP loss for one step across 16 slots.

    Both prev and curr must be valid (< HP_INVALID) for a slot to count.
    Only decrements are rewarded.
    """
    valid = (curr_hp < HP_INVALID) & (prev_hp < HP_INVALID)
    decr = (prev_hp.astype(int) - curr_hp.astype(int)).clip(min=0)
    return float(np.sum(np.where(valid, decr, 0)))


class ContraRamWrapper(gym.Wrapper):
    """RAM observation + hp-loss reward wrapper for Contra (NES).

    Observation: flat float32 array of shape (RAM_SIZE * stack,), normalised to [0, 1].
                 Frames are ordered oldest-first: [frame_{t-k+1}, ..., frame_t].
    Actions:     MultiDiscrete([7, 4]) — (dpad_idx, button_idx) combined via OR.
    Reward:      hp_loss_step + xscroll_delta / 30
    """

    def __init__(self, env, reset_round: bool = True,
                 warmup_frames: int = 120, skip: int = 3,
                 stack: int = 120, sample: int = 12,
                 max_episode_steps: int = 10000):
        super().__init__(env)
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self._dpad_table   = np.array(DPAD_TABLE,   dtype=env.action_space.dtype)
        self._button_table = np.array(BUTTON_TABLE, dtype=env.action_space.dtype)

        self.reset_round = reset_round
        self.warmup_frames = warmup_frames
        self.skip = skip
        self.stack = stack
        self.sample = sample
        self.max_episode_steps = max_episode_steps
        # Indices into the buffer sampled uniformly (oldest → newest)
        self._sample_idx: np.ndarray = np.round(
            np.linspace(0, stack - 1, sample)
        ).astype(int)

        self.action_space = gym.spaces.MultiDiscrete([NUM_DPAD, NUM_BUTTONS])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(RAM_SIZE * sample,), dtype=np.float32
        )

        # Circular buffer: shape (stack, RAM_SIZE)
        self._frames: np.ndarray = np.zeros((stack, RAM_SIZE), dtype=np.float32)

        self.prev_xscroll: int = 0
        self.prev_lives: int = 0
        self.prev_hp: np.ndarray = np.zeros(16, dtype=np.uint8)
        self.total_timesteps: int = 0
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self.ep = {
            "reward": 0.0,
            "hp_loss_reward": 0.0,
            "xscroll_reward": 0.0,
            "game_result_reward": 0.0,
            "end_reason": "",
        }

    def _push_frame(self) -> None:
        """Read current RAM and push it into the stack buffer (oldest dropped)."""
        ram = self.unwrapped.get_ram().astype(np.float32) / 255.0
        self._frames[:-1] = self._frames[1:]
        self._frames[-1]  = ram

    def _get_obs(self) -> np.ndarray:
        return self._frames[self._sample_idx].reshape(-1)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        for _ in range(self.warmup_frames):
            observation, _, _, _, info = self.env.step(self._no_op)

        self.prev_xscroll = info.get("xscroll", 0)
        self.prev_lives   = info.get("lives", 2)
        ram = self.unwrapped.get_ram()
        self.prev_hp = ram[HP_ADDRS].copy()
        self.total_timesteps = 0
        self._reset_episode_stats()

        # Fill all stack slots with the current frame so there are no zero-padding artefacts
        self._frames[:] = ram.astype(np.float32) / 255.0

        return self._get_obs(), info

    def step(self, action):
        nes_action = self._dpad_table[action[0]] | self._button_table[action[1]]
        done = False

        for i in range(self.skip):
            _, _, term, trunc, info = self.env.step(nes_action)
            if term or trunc:
                done = True
                break

        self.total_timesteps += 1

        ram = self.unwrapped.get_ram()
        curr_xscroll = info.get("xscroll", self.prev_xscroll)
        curr_lives   = info.get("lives", self.prev_lives)
        curr_hp      = ram[HP_ADDRS].copy()

        # --- reward components ---
        hp_loss_r   = _hp_loss_step(self.prev_hp, curr_hp)
        xscroll_r   = max(curr_xscroll - self.prev_xscroll, 0) / 30.0

        end_reason = ""
        result_r   = 0.0
        if curr_lives < self.prev_lives:
            done       = True
            end_reason = "game_over"
            result_r   = -50.0
        elif not done and self.total_timesteps >= self.max_episode_steps:
            done       = True
            end_reason = "time_out"
            result_r   = -50.0
        elif done:
            end_reason = "win"
            result_r   = 100.0

        reward = hp_loss_r + xscroll_r + result_r

        self.ep["reward"]             += reward
        self.ep["hp_loss_reward"]     += hp_loss_r
        self.ep["xscroll_reward"]     += xscroll_r
        self.ep["game_result_reward"] += result_r
        if end_reason:
            self.ep["end_reason"] = end_reason

        self.prev_xscroll = curr_xscroll
        self.prev_lives   = curr_lives
        self.prev_hp      = curr_hp
        self._push_frame()

        if done:
            info.update({
                "episode_max_x":   curr_xscroll,
                "episode_steps":   self.total_timesteps,
                **{f"episode_{k}": v for k, v in self.ep.items()},
            })

        if not self.reset_round:
            done = False

        return self._get_obs(), reward, done, False, info


def create_env(env, reset_round: bool = True,
               warmup_frames: int = 120, skip: int = 3,
               stack: int = 120, sample: int = 12,
               max_episode_steps: int = 10000) -> ContraRamWrapper:
    """Wrap a retro env with the RAM-based reward shaping."""
    return ContraRamWrapper(env, reset_round=reset_round,
                            warmup_frames=warmup_frames, skip=skip,
                            stack=stack, sample=sample,
                            max_episode_steps=max_episode_steps)
