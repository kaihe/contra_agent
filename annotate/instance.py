

import os
import random
import warnings

import numpy as np
import stable_retro as retro

from contra.replay import replay_actions, rewind_state, step_env, GAME, SKIP
from contra.events import EV_LEVELUP, EV_GAME_CLEAR, compute_reward
from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from pixel2play.model.nes_actions import encode, encode_combined, N_DPAD, N_BUTTONS
from annotate.prune import prune_actions

warnings.filterwarnings("ignore", message=".*Gym.*")

# NES MultiBinary(9): [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
_NES_KEY_MAP = [(0, "f"), (4, "w"), (5, "s"), (6, "a"), (7, "d"), (8, "j")]


def _nes_keys(nes: np.ndarray) -> list[str]:
    return [key for idx, key in _NES_KEY_MAP if nes[idx]]



# ── BCDataSample ───────────────────────────────────────────────────────────────
class BCDataSample:
    """Universal data holder for one recorded episode."""

    def __init__(self, features_path: str) -> None:
        self.features_path = features_path
        self.uuid          = os.path.splitext(os.path.basename(features_path))[0]

    @classmethod
    def load(cls, features_path: str) -> "BCDataSample":
        if not os.path.isfile(features_path):
            raise FileNotFoundError(f"{features_path!r} not found")
        return cls(features_path)

    @classmethod
    def create(cls, npz_path: str, game: str, out_dir: str | None = None) -> "BCDataSample":
        """Return a BCDataSample handle pointing to the output path.
        Call compute_features() afterwards to produce the training data.
        """
        rec_id  = os.path.splitext(os.path.basename(npz_path))[0]
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "bc_data", game)
        os.makedirs(out_dir, exist_ok=True)
        return cls(os.path.join(out_dir, f"{rec_id}.npz"))

    def save_from_features(self, ram: np.ndarray, dpad: np.ndarray, button: np.ndarray) -> str:
        """Persist RAM snapshots plus action labels. Returns output path."""
        np.savez(self.features_path, ram=ram, dpad=dpad, button=button)
        return self.features_path

    def replay_game(self, npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replay the emulator and return (ram, dpad, button) arrays without saving.

        Raises AssertionError if the trace ends in 'lose'.
        """
        npz_data    = np.load(npz_path, allow_pickle=True)
        initial_state = bytes(npz_data["initial_state"])
        raw_actions = prune_actions(
            npz_data["actions"], initial_state, verbose=True,
        )                                                          # (N, 9) uint8 pruned
        N           = len(raw_actions)

        dpad    = np.empty(N, dtype=np.int8)
        button  = np.empty(N, dtype=np.int8)
        ram_buf = []                                               # list of (2048,) uint8

        env = retro.make(
            game=GAME,
            state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.RAM,
            render_mode=None,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        env.reset()
        rewind_state(env, initial_state)

        ram_buf.append(env.unwrapped.get_ram().copy())             # ram[0]: initial state
        leveled_up   = False
        game_cleared = False
        emu_states   = []
        reward_list  = []

        for i, act in enumerate(raw_actions):
            act_arr = np.asarray(act, dtype=np.uint8)
            pre_ram = env.unwrapped.get_ram().copy()
            emu_states.append(env.em.get_state())

            for _ in range(SKIP):
                env.step(act_arr.copy())

            dpad[i], button[i] = encode(_nes_keys(act_arr))

            curr_ram = env.unwrapped.get_ram().copy()
            ram_buf.append(curr_ram)                               # ram[i+1]: after action i
            reward_list.append(compute_reward(pre_ram, curr_ram))

            if EV_LEVELUP.trigger(pre_ram, curr_ram):
                leveled_up = True
            if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
                game_cleared = True

        env.close()

        self.emu_states  = emu_states
        self.reward_arr  = np.array(reward_list, dtype=np.float32)  # (N,)

        outcome = "game_clear" if game_cleared else "level_up" if leveled_up else "lose"
        assert outcome != "lose", \
            f"{self.uuid}: replay ended in 'lose' — skipping feature computation"

        ram = np.stack(ram_buf[:-1], axis=0)                      # (N, 2048) uint8
        return ram, dpad, button

    def compute_features(self, npz_path: str) -> str:
        """Replay the emulator and save features to disk. Returns output path."""
        if os.path.isfile(self.features_path):
            print(f"  {self.uuid}: already exists, skipping")
            return self.features_path
        ram, dpad, button = self.replay_game(npz_path)
        return self.save_from_features(ram, dpad, button)

    @property
    def has_features(self) -> bool:
        return os.path.isfile(self.features_path)

    def __repr__(self) -> str:
        return f"BCDataSample(uuid={self.uuid!r})"
