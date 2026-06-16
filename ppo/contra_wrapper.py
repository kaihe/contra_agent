"""Contra (NES) gym wrapper: discrete actions, frame skip/stack, reward shaping."""

import gzip
import json
import os
import zipfile

import cv2
import gymnasium as gym
import numpy as np

from contra.events import (
    ADDR_LEVEL,
    ADDR_XSCROLL_HI,
    EV_BOSS_HIT,
    EV_CORE_BROKEN,
    EV_LEVELUP,
    EV_PLAYER_DIE,
    EV_PUSH_INSIDE,
    EV_PUSH_UP,
    EV_REGULAR_ENEMY_HIT,
    EV_ROOM_ENTER,
    EV_SPREAD_PICK,
    is_gameplay,
    level_advance_style,
)


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

# Three RGB-sliced history channels: R(t), G(t-1), B(t-3).
RGB_CHANNELS = 3
HISTORY_OFFSETS = [0, 1, 3]
BUFFER_FRAMES = max(HISTORY_OFFSETS) + 1

ENEMY_HP_REGION_SIZE_PX = 256
DEFAULT_ENEMY_HP_CAP_PER_REGION = 5.0


def save_config_to_model(
    model_path: str,
    skip: int = 3,
    stack: int = 3,
    train_config: dict | None = None,
) -> None:
    """Embed contra_config.json into an SB3 model .zip file."""
    config = {"dpad_table": DPAD_TABLE, "dpad_names": DPAD_NAMES,
              "button_table": BUTTON_TABLE, "button_names": BUTTON_NAMES,
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


def xscroll(ram: np.ndarray) -> int:
    return int(ram[100]) << 8 | int(ram[101])


def reward_components(
    pre_ram: np.ndarray,
    curr_ram: np.ndarray,
    weights: dict[str, float],
    prev_xscroll: int,
    max_progress_px: float,
    enemy_hp_event_cap: float | None = None,
    timed_out: bool = False,
) -> dict[str, float]:
    """Level-aware reward components.

    The combat / item / terminal components are level-agnostic. The *advancement*
    component is selected from the level read out of RAM (ADDR_LEVEL), using the
    same per-level advancement style as the mc_search event system:
      "forward" : horizontal scroll progress (side-scroll levels)
      "inside"  : core destroyed + walking through door + entering next room (indoor)
      "up"      : vertical scroll progress (climbing levels)
    So one wrapper supports every level — it just needs to start in the right state.
    """
    enemy_hp = EV_REGULAR_ENEMY_HIT.trigger(pre_ram, curr_ram)
    if enemy_hp_event_cap is not None:
        enemy_hp = min(enemy_hp, max(enemy_hp_event_cap, 0.0))

    components = {
        "enemy_hp": weights["enemy_hp"] * enemy_hp,
        "boss_hp": weights["boss_hp"] * EV_BOSS_HIT.trigger(pre_ram, curr_ram),
        "spread_pick": weights["spread_pick"] * EV_SPREAD_PICK.trigger(pre_ram, curr_ram),
        "levelup": weights["levelup"] * EV_LEVELUP.trigger(pre_ram, curr_ram),
        "player_die": weights["player_die"] * EV_PLAYER_DIE.trigger(pre_ram, curr_ram),
        "time_out": weights["time_out"] * float(timed_out),
    }

    style = level_advance_style(int(pre_ram[ADDR_LEVEL]))
    if style == "inside":
        components["core_broken"] = weights["core_broken"] * EV_CORE_BROKEN.trigger(pre_ram, curr_ram)
        components["push_inside"] = weights["push_inside"] * EV_PUSH_INSIDE.trigger(pre_ram, curr_ram)
        components["room_enter"] = weights["room_enter"] * EV_ROOM_ENTER.trigger(pre_ram, curr_ram)
    elif style == "up":
        components["push_up"] = weights["push_up"] * EV_PUSH_UP.trigger(pre_ram, curr_ram)
    else:  # "forward"
        progress = float(np.clip(
            xscroll(curr_ram) - prev_xscroll,
            -max_progress_px,
            max_progress_px,
        ))
        components["progress"] = weights["progress"] * progress

    return components


DEFAULT_REWARD_WEIGHTS = {
    "enemy_hp": 1.0,
    "boss_hp": 1.0,
    "progress": 1.0 / 60.0,    # "forward" levels: per xscroll pixel
    "core_broken": 10.0,       # "inside" levels: wall core destroyed (sparse)
    "push_inside": 0.5,        # "inside" levels: per step walking through the door
    "room_enter": 10.0,        # "inside" levels: entered the next indoor screen
    "push_up": 0.5,            # "up" levels: per vertical-scroll pixel
    "spread_pick": 20.0,
    "levelup": 100.0,
    "player_die": -15.0,
    "time_out": -10.0,
}


class ContraWrapper(gym.Wrapper):
    """Reward shaping + frame skip + history sampling for Contra NES.

    Observation: (84, 84, stack) uint8 channels-last for SB3 CnnPolicy.
                 Channels are R(t), G(t-1), B(t-3).
    Actions:     MultiDiscrete([7, 4]) — (dpad_idx, button_idx) combined via OR.
    """

    def __init__(self, env, monitor=None, random_start_frames=0,
                 warmup_frames=120, skip=3, stack=3, level=1,
                 max_episode_steps=10000,
                 enemy_hp_cap_per_region=DEFAULT_ENEMY_HP_CAP_PER_REGION,
                 reward_weights=None):
        super().__init__(env)
        weights = DEFAULT_REWARD_WEIGHTS.copy()
        if reward_weights is not None:
            unknown = set(reward_weights) - set(weights)
            if unknown:
                raise ValueError(f"Unknown reward weight(s): {sorted(unknown)}")
            weights.update(reward_weights)
        self.reward_weights = weights
        self.max_progress_px = 30.0
        self._no_op = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        self._dpad_table   = np.array(DPAD_TABLE,   dtype=env.action_space.dtype)
        self._button_table = np.array(BUTTON_TABLE, dtype=env.action_space.dtype)
        self.monitor = monitor
        self.random_start_frames = random_start_frames
        self.warmup_frames = warmup_frames
        self.skip = skip
        self.stack = stack
        self.level = level
        self.max_episode_steps = max_episode_steps
        self.enemy_hp_cap_per_region = float(enemy_hp_cap_per_region)
        if self.enemy_hp_cap_per_region < 0:
            raise ValueError("enemy_hp_cap_per_region must be non-negative")
        if stack != len(HISTORY_OFFSETS):
            raise ValueError(
                f"stack={stack} must match len(HISTORY_OFFSETS)={len(HISTORY_OFFSETS)}"
            )

        self.action_space = gym.spaces.MultiDiscrete([NUM_DPAD, NUM_BUTTONS])
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
            "end_reason": "",
        }
        self.enemy_hp_events_by_region = {}

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
        # Region key for the anti-farming cap. On "forward" levels regions are
        # horizontal scroll bands; on indoor/climb levels xscroll is ~flat, so
        # key the cap on the screen/room number instead (else the whole level is
        # one region and the cap is exhausted on the first screen).
        if level_advance_style(int(curr_ram[ADDR_LEVEL])) == "forward":
            enemy_hp_region = curr_xscroll // ENEMY_HP_REGION_SIZE_PX
        else:
            enemy_hp_region = int(curr_ram[ADDR_XSCROLL_HI])
        enemy_hp_event_cap = max(
            self.enemy_hp_cap_per_region
            - self.enemy_hp_events_by_region.get(enemy_hp_region, 0.0),
            0.0,
        )
        timed_out = not done and self.total_timesteps >= self.max_episode_steps

        end_reason = ""
        rewards = reward_components(
            self.prev_ram,
            curr_ram,
            self.reward_weights,
            self.prev_xscroll,
            self.max_progress_px,
            enemy_hp_event_cap,
            timed_out,
        )
        events = self._events_from_rewards(rewards)
        self.enemy_hp_events_by_region[enemy_hp_region] = (
            self.enemy_hp_events_by_region.get(enemy_hp_region, 0.0)
            + events["enemy_hp"]
        )

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
        nes_action = self._dpad_table[action[0]] | self._button_table[action[1]]
        done = False
        states = []

        for i in range(self.skip):
            act = nes_action
            # Release B on the last skip frame so auto-fire retriggers next
            # step (holding B fires only once — the "B-stuck" bug).
            if i == self.skip - 1 and act[0]:
                act = act.copy()
                act[0] = 0
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

        # Max-pool the last two raw frames to defeat NES sprite flicker.
        pooled = np.maximum.reduce(states[-2:])
        self._buf_pos = (self._buf_pos + 1) % BUFFER_FRAMES
        self._buf[self._buf_pos] = process_frame(pooled)

        if done:
            info.update({
                "episode_delta_x": self.max_xscroll - self.episode_start_x,
                "episode_enemy_hp_cost": self.ep["enemy_hp_cost"],
                "episode_reward": self.ep["reward"],
                "episode_end_reason": self.ep["end_reason"],
                "episode_steps":    self.total_timesteps,
            })

        return self._get_obs(), reward, done, False, info


def create_env(env, monitor=None, random_start_frames=0, warmup_frames=120,
               skip=3, stack=3, level=1, max_episode_steps=10000,
               enemy_hp_cap_per_region=DEFAULT_ENEMY_HP_CAP_PER_REGION,
               reward_weights=None):
    """Wrap a retro env with reward shaping + frame skip + history sampling."""
    return ContraWrapper(env, monitor=monitor,
                         random_start_frames=random_start_frames,
                         warmup_frames=warmup_frames,
                         skip=skip, stack=stack, level=level,
                         max_episode_steps=max_episode_steps,
                         enemy_hp_cap_per_region=enemy_hp_cap_per_region,
                         reward_weights=reward_weights)
