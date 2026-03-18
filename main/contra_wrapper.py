"""Contra (NES) gym wrapper: discrete actions, frame skip/stack, reward shaping."""

import cv2
import gymnasium as gym
import json
import numpy as np
import zipfile


# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Top-7 human-frequency combos (derived from 55 recorded gameplay traces).
# Fire (B) is not forced on every action — agent chooses when to shoot.
# Up / Up+Fire added for Level 2 (top-down "walk into screen" perspective).
ACTION_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 0: Right       R  (31.6%)
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: Right+Fire  RF (15.6%)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2: Down        D  ( 7.4%)
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # 3: Right+Jump  RJ ( 6.1%)
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: Down+Fire   DF ( 5.6%)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 5: Fire        F  ( 5.2%)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 6: Left        L  ( 3.2%)
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 7: Up          U
]
ACTION_NAMES = ["R", "RF", "D", "RJ", "DF", "F", "L", "U"]
NUM_ACTIONS = len(ACTION_TABLE)


def save_config_to_model(model_path: str, skip: int = 8, stack: int = 4) -> None:
    """Embed contra_config.json into an SB3 model .zip file."""
    config = {"action_table": ACTION_TABLE, "action_names": ACTION_NAMES,
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
    """Override the global ACTION_TABLE/ACTION_NAMES with values from config."""
    global ACTION_TABLE, ACTION_NAMES, NUM_ACTIONS
    ACTION_TABLE = config["action_table"]
    ACTION_NAMES = config["action_names"]
    NUM_ACTIONS = len(ACTION_TABLE)


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
    """RGB → grayscale 84×84, shape (1, 84, 84) uint8."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)[None, :, :]


# ---------------------------------------------------------------------------
# Pure reward helpers — usable outside ContraWrapper (e.g. MC playfun)
# ---------------------------------------------------------------------------

def get_enemy_hp_sum(ram, xscroll: int, level: int) -> int:
    """Level-aware sum of all active enemy HP."""
    if level == 1:
        if xscroll < 3072:
            return 0
        return int(ram[1412]) + int(ram[1414]) + int(ram[1415])
    if level == 2:
        core = 0 if int(ram[1414]) >= 240 else int(ram[1414])
        if xscroll < 1280:
            return core
        boss_gun_addrs = [1409, 1410, 1411, 1412, 1413]
        total = core + sum(0 if int(ram[a]) >= 240 else int(ram[a]) for a in boss_gun_addrs)
        total += int(ram[1479])
        return total
    raise ValueError(f"Unsupported level: {level}")


def reward_enemy_progress(prev_hp_sum: int, curr_hp_sum: int) -> float:
    """Reward for dealing damage; capped at 16 to avoid transition spikes."""
    hit_diff = prev_hp_sum - curr_hp_sum
    if hit_diff <= 0:
        return 0.0
    return min(hit_diff, 16) * 0.5


def reward_distance(curr_xscroll: int, prev_xscroll: int,
                    max_x_reached: int, idle_steps: int,
                    level: int) -> tuple[float, int, int]:
    """Returns (reward, updated_max_x_reached, updated_idle_steps)."""
    if level == 1:
        if curr_xscroll > max_x_reached:
            idle_steps = 0
            max_x_reached = curr_xscroll
        else:
            idle_steps += 1
        if idle_steps > 20 and curr_xscroll < 3072:
            r = -0.05
        else:
            r = max(min(curr_xscroll - prev_xscroll, 3.0), 0) * 0.1
        return r, max_x_reached, idle_steps
    if level == 2:
        if curr_xscroll > max_x_reached:
            rooms_advanced = (curr_xscroll - max_x_reached) // 256
            max_x_reached = curr_xscroll
            return rooms_advanced * 5.0, max_x_reached, idle_steps
        return 0.0, max_x_reached, idle_steps
    raise ValueError(f"Unsupported level: {level}")


class ContraWrapper(gym.Wrapper):
    """Reward shaping + frame skip + stacking for Contra NES.

    Observation: (84, 84, stack) uint8 channels-last for SB3 CnnPolicy.
    Actions:     Discrete(7) mapped via ACTION_TABLE.
    """

    def __init__(self, env, monitor=None, reset_round=True, random_start_frames=0,
                 warmup_frames=120, skip=8, stack=4, level=1):
        super().__init__(env)
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self._action_table = np.array(ACTION_TABLE, dtype=env.action_space.dtype)
        self.monitor = monitor
        self.reset_round = reset_round
        self.random_start_frames = random_start_frames
        self.warmup_frames = warmup_frames
        self.skip = skip
        self.stack = stack
        self.level = level
        self.max_episode_steps = 4000

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.states = np.zeros((stack, 84, 84), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, stack), dtype=np.uint8
        )

        self.prev_xscroll = 0
        self.prev_score = 0
        self.prev_lives = 0
        self.max_x_reached = 0
        self.total_timesteps = 0
        self.idle_steps = 0
        self.prev_enemy_hp_sum = 0
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.ep = {
            "reward": 0.0,
            "distance_reward": 0.0,
            "score_reward": 0.0,
            "death_reward": 0.0,
            "game_result_reward": 0.0,
            "enemy_progress_reward": 0.0,
            "end_reason": "",
        }

    def _get_obs(self):
        return np.transpose(self.states, (1, 2, 0))

    def _get_enemy_hp_sum(self, ram, xscroll=0):
        return get_enemy_hp_sum(ram, xscroll, self.level)

    def _reward_distance(self, curr_xscroll):
        r, self.max_x_reached, self.idle_steps = reward_distance(
            curr_xscroll, self.prev_xscroll, self.max_x_reached, self.idle_steps, self.level
        )
        return r

    def _reward_enemy_progress(self, curr_enemy_hp_sum):
        return reward_enemy_progress(self.prev_enemy_hp_sum, curr_enemy_hp_sum)

    def _compute_rewards(self, info, done):
        ram = self.unwrapped.get_ram()
        curr_xscroll = info.get("xscroll", self.prev_xscroll)
        curr_score = info.get("score", 0)
        curr_lives = info.get("lives", 0)
        curr_enemy_hp_sum = self._get_enemy_hp_sum(ram, curr_xscroll)


        end_reason = ""
        result_reward = 0.0
        if not done and self.total_timesteps >= self.max_episode_steps:
            done = True
            end_reason = "time_out"
            result_reward = -50.0
        elif done:
            end_reason = "win" if curr_lives > 0 else "game_over"
            result_reward = 100.0 if curr_lives > 0 else 0.0

        rewards = {
            "distance":         self._reward_distance(curr_xscroll),
            "score":            max(curr_score - self.prev_score, 0),
            "death":            -20.0 if curr_lives < self.prev_lives else 0.0,
            "enemy_progress":   self._reward_enemy_progress(curr_enemy_hp_sum),
            "game_result":      result_reward,
        }

        if end_reason:
            self.ep["end_reason"] = end_reason

        self.prev_xscroll = curr_xscroll
        self.prev_score = curr_score
        self.prev_lives = curr_lives
        self.prev_enemy_hp_sum = curr_enemy_hp_sum

        return rewards, done

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.prev_xscroll = info.get("xscroll", 0)
        self.prev_lives = info.get("lives", 2)
        self.prev_score = 0
        self.max_x_reached = self.prev_xscroll
        self.total_timesteps = 0
        self.idle_steps = 0
        self._reset_episode_stats()

        for _ in range(self.warmup_frames):
            observation, _, _, _, info = self.env.step(self._no_op)
            if self.monitor:
                self.monitor.record(observation)

        self.prev_xscroll = info.get("xscroll", 0)
        self.prev_score = info.get("score", 0)
        self.max_x_reached = self.prev_xscroll

        if self.random_start_frames > 0:
            for _ in range(np.random.randint(0, self.random_start_frames + 1)):
                observation, _, _, _, info = self.env.step(self._no_op)
                if self.monitor:
                    self.monitor.record(observation)

        ram = self.unwrapped.get_ram()
        self.prev_enemy_hp_sum = self._get_enemy_hp_sum(ram, self.prev_xscroll)

        self.states[:] = process_frame(observation)
        return self._get_obs(), info

    def step(self, action):
        nes_action = self._action_table[action]
        last_two = [None, None]
        done = False

        for i in range(self.skip):
            act = nes_action.copy()
            if i == self.skip - 1:
                act[0] = 0  # Release B on last frame for rapid fire
            state, _, term, trunc, info = self.env.step(act)
            if self.monitor:
                self.monitor.record(state)
            last_two[0], last_two[1] = last_two[1], state
            if term or trunc:
                done = True
                break

        self.total_timesteps += 1
        rewards, done = self._compute_rewards(info, done)
        reward = sum(rewards.values())

        self.ep["reward"] += reward
        self.ep["distance_reward"] += rewards["distance"]
        self.ep["score_reward"] += rewards["score"]
        self.ep["death_reward"] += rewards["death"]
        self.ep["game_result_reward"] += rewards["game_result"]
        self.ep["enemy_progress_reward"] += rewards["enemy_progress"]

        pooled = np.maximum(last_two[0], last_two[1]) if last_two[0] is not None else last_two[1]
        self.states[:-1] = self.states[1:]
        self.states[-1] = process_frame(pooled)[0]

        if done:
            info.update({
                "episode_max_x":    self.max_x_reached,
                "episode_score":    self.prev_score,
                "episode_steps":    self.total_timesteps,
                **{f"episode_{k}": v for k, v in self.ep.items()},
            })

        if not self.reset_round:
            done = False

        return self._get_obs(), reward, done, False, info


def create_env(env, monitor=None, reset_round=True, random_start_frames=0,
               skip=8, stack=4, level=1):
    """Wrap a retro env with reward shaping + frame skip + stacking."""
    return ContraWrapper(env, monitor=monitor, reset_round=reset_round,
                         random_start_frames=random_start_frames,
                         skip=skip, stack=stack, level=level)
