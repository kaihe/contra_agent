"""
BCData Base Classes
===================
BCDataSample  - universal data holder for one recorded episode.

Recording / Training-data Sequence
===================================
Each logical step = SKIP=3 NES frames (logical FPS=20).
The RAM snapshot is captured before each action is applied, giving
the model the full current game state when predicting the next action.

  t=0
  ┌─────────────────────────────────────────────────────────────┐
  │  ENV                                                        │
  │  env.reset() + rewind_state()                               │
  │  RAM:    ram[0] = initial RAM state                         │
  │  model:  predicts action_0 from ram[0]                      │
  │  input:  action_0                                           │
  │  NES:    nes_1 → nes_2 → nes_3   (SKIP=3)                   │
  └─────────────────────────────────────────────────────────────┘

  t=1
  ┌─────────────────────────────────────────────────────────────┐
  │  ENV                                                        │
  │  RAM:    ram[1] = RAM after action_0                        │
  │  model:  predicts action_1 from ram[1]                      │
  │  input:  action_1                                           │
  │  NES:    nes_4 → nes_5 → nes_6   (SKIP=3)                   │
  └─────────────────────────────────────────────────────────────┘

  ...  (pattern repeats for all N actions)

Output file: bc_features.npz
  ram    : uint8   (N, 2048)          NES RAM snapshot per step
  dpad   : int8    (N,)               encoded dpad class 0..8
  button : int8    (N,)               encoded button class 0..3
  text   : float16 (N, 1, 768)        Gemma embeddings; zeros if use_text=False

Training chunk  (get_chunk(start, T=200)):
  position t:  ram[start+t], dpad[start+t], button[start+t], text[start+t]
  model target: predict action_t from ram[0..t] and action[0..t-1]  (causal)
"""

import os

import numpy as np

from contra.replay import replay_actions, rewind_state, step_env, GAME, SKIP
from pixel2play.model.nes_actions import encode

# NES MultiBinary(9): [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
_NES_KEY_MAP = [(0, "f"), (4, "w"), (5, "s"), (6, "a"), (7, "d"), (8, "j")]

# ── Action pruning ─────────────────────────────────────────────────────────────

# RAM bytes that mirror the controller input directly (not gameplay state).
#   $f1/$f2 = CONTROLLER_STATE (P1/P2 currently-pressed buttons)
#   $f5/$f6 = CONTROLLER_STATE_DIFF (delta between reads)
#   $f9/$fa = CTRL_KNOWN_GOOD (last valid read)
_INPUT_RAM_INDICES = np.array([0xf1, 0xf2, 0xf5, 0xf6, 0xf9, 0xfa], dtype=np.intp)

# The 6 independently-testable input bits.
_CRITICAL_BITS = [(0, "fire"), (8, "jump"), (4, "up"), (5, "down"), (6, "left"), (7, "right")]

_NOOP = np.zeros(9, dtype=np.uint8)


def _ram_eq(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a and b differ only in controller-mirror bytes."""
    if np.array_equal(a, b):
        return True
    diff = np.where(a != b)[0]
    return bool(np.all(np.isin(diff, _INPUT_RAM_INDICES)))


def prune_actions(actions: np.ndarray, initial_emu_state: bytes, verbose: bool = True) -> np.ndarray:
    """Zero out each of the 6 critical input bits when they leave RAM unchanged.

    Each bit (fire, jump, up, down, left, right) is tested independently
    against the original action's RAM result.  Controller-mirror bytes are
    excluded from the comparison so input-register noise never blocks pruning.

    To avoid creating fake 'Just Pressed' events (0->1 transitions) that diverge
    the PRNG and game state later, we process the sequence backwards and only
    allow pruning a bit if the NEXT frame also has that bit as 0.

    Returns
    -------
    pruned : np.ndarray (N, 9) uint8
    """
    import stable_retro as retro

    n = len(actions)
    pruned = np.array(actions, dtype=np.uint8).copy()

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial_emu_state)

    # 1. Forward pass to save all true states and true RAMs
    true_states = []
    true_rams = []
    for i in range(n):
        true_states.append(env.em.get_state())
        step_env(env, actions[i])
        true_rams.append(env.unwrapped.get_ram().copy())

    # 2. Backward pass to prune safely
    arrays_pruned = 0

    for i in range(n - 1, -1, -1):
        act_orig = actions[i]
        
        # Only consider bits that are 1 and where the NEXT frame (if any) is 0
        active_bits = []
        for idx, name in _CRITICAL_BITS:
            if act_orig[idx] == 1:
                # Safe to prune if it's the last frame OR the next frame also has this bit as 0
                if i == n - 1 or pruned[i+1][idx] == 0:
                    active_bits.append((idx, name))

        if not active_bits:
            continue

        candidate = act_orig.copy()
        cur_state = true_states[i]
        true_ram = true_rams[i]
        
        for idx, name in active_bits:
            probe = candidate.copy()
            probe[idx] = 0
            rewind_state(env, cur_state)
            step_env(env, probe)
            if _ram_eq(true_ram, env.unwrapped.get_ram()):
                candidate = probe
            
        pruned[i] = candidate
        if not np.array_equal(candidate, act_orig):
            arrays_pruned += 1

    env.close()

    if verbose:
        print(f"    prune: {arrays_pruned}/{n} action arrays modified")

    return pruned


def _nes_keys(nes: np.ndarray) -> list[str]:
    return [key for idx, key in _NES_KEY_MAP if nes[idx]]


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

    def replay_arrays(self, npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replay the emulator and return (ram, dpad, button) arrays without saving.

        Raises AssertionError if the trace ends in 'lose'.
        """
        import stable_retro as retro
        from contra.events import EV_LEVELUP, EV_GAME_CLEAR

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

        for i, act in enumerate(raw_actions):
            act_arr = np.asarray(act, dtype=np.uint8)
            pre_ram = env.unwrapped.get_ram().copy()

            for _ in range(SKIP):
                env.step(act_arr.copy())

            dpad[i], button[i] = encode(_nes_keys(act_arr))

            curr_ram = env.unwrapped.get_ram().copy()
            ram_buf.append(curr_ram)                               # ram[i+1]: after action i

            if EV_LEVELUP.trigger(pre_ram, curr_ram):
                leveled_up = True
            if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
                game_cleared = True

        env.close()

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
        ram, dpad, button = self.replay_arrays(npz_path)
        return self.save_from_features(ram, dpad, button)

    @property
    def has_features(self) -> bool:
        return os.path.isfile(self.features_path)

    def __repr__(self) -> str:
        return f"BCDataSample(uuid={self.uuid!r})"
