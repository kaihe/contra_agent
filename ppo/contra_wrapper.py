"""Contra (NES) gym wrapper: discrete actions, frame skip/stack, reward shaping."""

import cv2
import gymnasium as gym
import json
import numpy as np
import zipfile


# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
#
# Two-head action space: agent outputs (dpad_idx, button_idx) independently.
# The NES action is the bitwise OR of the two selected rows.

# Head 1 — D-pad (7 options)
DPAD_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: Right
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: Left
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: Down
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 5: Up+Right
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 6: Down+Right
]
DPAD_NAMES = ["_", "R", "L", "U", "D", "UR", "DR"]

# Head 2 — Buttons (4 options)
BUTTON_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Fire (B)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2: Jump (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: Fire+Jump
]
BUTTON_NAMES = ["_", "F", "J", "FJ"]

NUM_DPAD    = len(DPAD_TABLE)
NUM_BUTTONS = len(BUTTON_TABLE)

# Grayscale frame history: keep 240 frames (12 s at 20 Hz), sample 4 uniformly
BUFFER_FRAMES = 240


def save_config_to_model(model_path: str, skip: int = 3, stack: int = 4) -> None:
    """Embed contra_config.json into an SB3 model .zip file."""
    config = {"dpad_table": DPAD_TABLE, "dpad_names": DPAD_NAMES,
              "button_table": BUTTON_TABLE, "button_names": BUTTON_NAMES,
              "skip": skip, "stack": stack}
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
    """Override the global action tables with values from config."""
    global DPAD_TABLE, DPAD_NAMES, BUTTON_TABLE, BUTTON_NAMES, NUM_DPAD, NUM_BUTTONS
    DPAD_TABLE   = config["dpad_table"]
    DPAD_NAMES   = config["dpad_names"]
    BUTTON_TABLE = config["button_table"]
    BUTTON_NAMES = config["button_names"]
    NUM_DPAD     = len(DPAD_TABLE)
    NUM_BUTTONS  = len(BUTTON_TABLE)


class Monitor:
    """Record raw RGB frames and/or display live via pygame."""

    def __init__(self, width, height, saved_path=None, render=False, skip=8):
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
    """RGB → grayscale 84×84, shape (84, 84) uint8."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Pure reward helpers — usable outside ContraWrapper (e.g. MC playfun)
# ---------------------------------------------------------------------------

def event_enemy_hit(pre_ram: np.ndarray, curr_ram: np.ndarray) -> float:
    """Sum of HP decrements across all 16 slots for one step."""
    pre_hp = pre_ram[1400:1416]
    curr_hp = curr_ram[1400:1416]
    valid = (curr_hp < 240) & (pre_hp < 240)
    decr  = (pre_hp.astype(int) - curr_hp.astype(int)).clip(min=0)
    return float(np.sum(np.where(valid, decr, 0)))

def event_push_forward(pre_xscroll: int, curr_xscroll: int,
                       max_x_reached: int, level: int) -> float:
    if level == 1:
        return (curr_xscroll - pre_xscroll) / 30.0
    if level == 2:
        if curr_xscroll > max_x_reached:
            rooms_advanced = (curr_xscroll - max_x_reached) // 256
            return rooms_advanced * 5.0
        return 0.0
    raise ValueError(f"Unsupported level: {level}")

def event_spread_gun(pre_ram: np.ndarray, curr_ram: np.ndarray) -> float:
    if curr_ram[0x010A] == 0x1F and pre_ram[0x010A] != 0x1F:
        if (curr_ram[0xAA] & 0x0F) == 3 and (pre_ram[0xAA] & 0x0F) != 3:
            return 1.0
    if (pre_ram[0xAA] & 0x0F) == 3 and (curr_ram[0xAA] & 0x0F) != 3:
        return -1.0
    return 0.0

def event_death(pre_lives: int, curr_lives: int) -> float:
    return 1.0 if curr_lives < pre_lives else 0.0


DEFAULT_REWARD_WEIGHTS = {
    "distance": 1.0,
    "score": 1.0,
    "enemy_hit": 1.0,
    "spread_gun": 10.0,
    "death": -50.0,
    "time_out": -50.0,
    "win": 100.0,
}

class ContraWrapper(gym.Wrapper):
    """Reward shaping + frame skip + history sampling for Contra NES.

    Observation: (84, 84, stack) uint8 channels-last for SB3 CnnPolicy.
                 `stack` frames sampled uniformly from a BUFFER_FRAMES history.
    Actions:     MultiDiscrete([7, 4]) — (dpad_idx, button_idx) combined via OR.
    """

    def __init__(self, env, monitor=None, random_start_frames=0,
                 warmup_frames=120, skip=3, stack=4, level=1, reward_weights=None):
        super().__init__(env)
        self.reward_weights = reward_weights if reward_weights is not None else DEFAULT_REWARD_WEIGHTS.copy()
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self._dpad_table   = np.array(DPAD_TABLE,   dtype=env.action_space.dtype)
        self._button_table = np.array(BUTTON_TABLE, dtype=env.action_space.dtype)
        self.monitor = monitor
        self.random_start_frames = random_start_frames
        self.warmup_frames = warmup_frames
        self.skip = skip
        self.stack = stack
        self.level = level
        self.max_episode_steps = 10000

        # Uniformly-spaced sample indices into the history buffer
        self._sample_idx = np.round(
            np.linspace(0, BUFFER_FRAMES - 1, stack)
        ).astype(int)

        self.action_space = gym.spaces.MultiDiscrete([NUM_DPAD, NUM_BUTTONS])
        # Grayscale 84×84 history; 1.7 MB shift per step
        self._buf = np.zeros((BUFFER_FRAMES, 84, 84), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, stack), dtype=np.uint8
        )

        self.prev_ram = np.zeros(2048, dtype=np.uint8)
        self.prev_xscroll = 0
        self.prev_score = 0
        self.prev_lives = 0
        self.max_x_reached = 0
        self.total_timesteps = 0
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.ep = {
            "reward": 0.0,
            "end_reason": "",
        }
        for k in self.reward_weights.keys():
            self.ep[f"{k}_event"] = 0.0
            self.ep[f"{k}_reward"] = 0.0

    def _get_obs(self) -> np.ndarray:
        return np.stack(self._buf[self._sample_idx], axis=-1)

    def _compute_rewards(self, info, done):
        curr_ram = self.unwrapped.get_ram()
        curr_xscroll = info.get("xscroll", self.prev_xscroll)
        curr_score   = info.get("score", self.prev_score)
        curr_lives   = info.get("lives", self.prev_lives)

        end_reason = ""
        events = {k: 0.0 for k in self.reward_weights.keys()}

        # Core Game State Events
        if curr_lives < self.prev_lives:
            done = True
            end_reason = "game_over"
        elif not done and self.total_timesteps >= self.max_episode_steps:
            done = True
            end_reason = "time_out"
            events["time_out"] = 1.0
        elif done:
            end_reason = "win"
            events["win"] = 1.0

        events["death"] = event_death(self.prev_lives, curr_lives)
        events["distance"] = event_push_forward(self.prev_xscroll, curr_xscroll, self.max_x_reached, self.level)
        events["score"] = max(curr_score - self.prev_score, 0)
        events["enemy_hit"] = event_enemy_hit(self.prev_ram, curr_ram)
        events["spread_gun"] = event_spread_gun(self.prev_ram, curr_ram)
        
        rewards = {k: events.get(k, 0.0) * self.reward_weights.get(k, 0.0) for k in self.reward_weights.keys()}

        if end_reason:
            self.ep["end_reason"] = end_reason

        self.prev_xscroll = curr_xscroll
        self.prev_score = curr_score
        self.prev_lives = curr_lives
        self.prev_ram = curr_ram.copy()
        if curr_xscroll > self.max_x_reached:
            self.max_x_reached = curr_xscroll

        return events, rewards, done

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.total_timesteps = 0
        self._reset_episode_stats()

        for _ in range(self.warmup_frames):
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
        self.prev_xscroll = info.get("xscroll", 0)
        self.prev_score = info.get("score", 0)
        self.prev_lives = info.get("lives", 2)
        self.max_x_reached = self.prev_xscroll

        self._buf[:] = process_frame(observation)

        return self._get_obs(), info

    def step(self, action):
        nes_action = self._dpad_table[action[0]] | self._button_table[action[1]]
        done = False

        for _ in range(self.skip):
            state, _, term, trunc, info = self.env.step(nes_action)
            if self.monitor:
                self.monitor.record(state)
            if term or trunc:
                done = True
                break

        self.total_timesteps += 1
        events, rewards, done = self._compute_rewards(info, done)
        reward = sum(rewards.values())

        self.ep["reward"] += reward
        for k in events:
            self.ep[f"{k}_event"] += events[k]
            self.ep[f"{k}_reward"] += rewards.get(k, 0.0)

        self._buf[:-1] = self._buf[1:]
        self._buf[-1]  = process_frame(state)

        if done:
            info.update({
                "episode_max_x":    self.max_x_reached,
                "episode_score":    self.prev_score,
                "episode_steps":    self.total_timesteps,
                **{f"episode_{k}": v for k, v in self.ep.items()},
            })

        return self._get_obs(), reward, done, False, info


def create_env(env, monitor=None, random_start_frames=0, skip=3, stack=4, level=1, reward_weights=None):
    """Wrap a retro env with reward shaping + frame skip + history sampling."""
    return ContraWrapper(env, monitor=monitor,
                         random_start_frames=random_start_frames,
                         skip=skip, stack=stack, level=level, reward_weights=reward_weights)
