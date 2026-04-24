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
import random
import warnings

import numpy as np
import stable_retro as retro

from contra.replay import replay_actions, rewind_state, step_env, GAME, SKIP
from contra.events import EV_LEVELUP, EV_GAME_CLEAR, compute_reward
from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from pixel2play.model.nes_actions import encode, encode_combined, N_DPAD, N_BUTTONS

warnings.filterwarnings("ignore", message=".*Gym.*")

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


def prune_actions(actions: np.ndarray, initial_emu_state: bytes, verbose: bool = True, env=None) -> np.ndarray:
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
    n = len(actions)
    pruned = np.array(actions, dtype=np.uint8).copy()

    _own_env = env is None
    if _own_env:
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

    if _own_env:
        env.close()

    if verbose:
        print(f"    prune: {arrays_pruned}/{n} action arrays modified")

    return pruned


def _nes_keys(nes: np.ndarray) -> list[str]:
    return [key for idx, key in _NES_KEY_MAP if nes[idx]]


# ── DPO pair generation ────────────────────────────────────────────────────────

_DPO_CHUNK_LEN  = 128
_DPO_N_PIVOTS   = 8
_DPO_N_ROLLOUTS = 16

# (N_DPAD * N_BUTTONS, 9) — maps combined action index to 9-bit NES action
_DPO_ACTION_LOOKUP = np.array(
    [np.bitwise_or(DPAD_TABLE[d], BUTTON_TABLE[b])
     for d in range(N_DPAD) for b in range(N_BUTTONS)],
    dtype=np.uint8,
)


def _run_rollout(env, pivot_state: bytes, rollout_len: int, rng) -> tuple:
    """Run one rollout on an already-open env. Returns (ram_buf, actions_buf, cumulative_reward)."""
    rewind_state(env, pivot_state)

    ram_buf           = np.empty((rollout_len, 2048), dtype=np.uint8)
    actions_buf       = np.empty(rollout_len, dtype=np.int16)
    cumulative_reward = 0.0

    for i in range(rollout_len):
        pre_ram        = env.unwrapped.get_ram()
        ram_buf[i]     = pre_ram
        combined       = int(rng.integers(0, N_DPAD * N_BUTTONS))
        for _ in range(SKIP):
            env.step(_DPO_ACTION_LOOKUP[combined])
        cumulative_reward += compute_reward(pre_ram, env.unwrapped.get_ram())
        actions_buf[i] = combined

    return ram_buf.copy(), actions_buf.copy(), cumulative_reward


def collect_dpo_pairs(
    npz_path: str,
    n_pivots:   int = _DPO_N_PIVOTS,
    n_rollouts: int = _DPO_N_ROLLOUTS,
    seed:       int = 0,
) -> "dict | None":
    """Thin wrapper — see DPODataSample.collect_pairs for full documentation."""
    return DPODataSample(npz_path).collect_pairs(
        n_pivots=n_pivots, n_rollouts=n_rollouts, seed=seed,
    )


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


# ── DPODataSample ──────────────────────────────────────────────────────────────

class DPODataSample(BCDataSample):
    """DPO pair generation from one winning mc_trace NPZ.

    Extends BCDataSample so the mc_trace path is stored as features_path and
    the uuid is derived from the filename — no separate constructor needed.
    """

    def collect_pairs(
        self,
        n_pivots:   int = _DPO_N_PIVOTS,
        n_rollouts: int = _DPO_N_ROLLOUTS,
        seed:       int = 0,
    ) -> "dict | None":
        """Generate DPO training pairs from the mc_trace at self.features_path.

        Returns None if no valid pairs were found (trace too short, or all
        random rollouts score >= original).

        Returns a dict with keys:
          chosen_ram       : (M, CHUNK_LEN, 2048) uint8
          chosen_actions   : (M, CHUNK_LEN)       int16
          rejected_ram     : (M, CHUNK_LEN, 2048) uint8
          rejected_actions : (M, CHUNK_LEN)       int16
          pivot            : (M,)                 int16  loss computed on [pivot:]
        """
        ram, dpad, button = self.replay_game(self.features_path)
        N = len(ram)
        ram_arr      = ram
        combined_arr = np.array(
            [encode_combined(int(dpad[i]), int(button[i])) for i in range(N)],
            dtype=np.int16,
        )
        emu_states  = self.emu_states
        reward_arr  = self.reward_arr

        n_chunks = N // _DPO_CHUNK_LEN
        if n_chunks == 0:
            return None

        rng = random.Random(seed)
        chosen_rams, chosen_acts     = [], []
        rejected_rams, rejected_acts = [], []
        pivots = []

        env = retro.make(
            game=GAME, state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.RAM, render_mode=None,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        env.reset()

        for ci in range(n_chunks):
            cs          = ci * _DPO_CHUNK_LEN
            orig_reward = float(reward_arr[cs : cs + _DPO_CHUNK_LEN].sum())
            if orig_reward <= 0:
                continue

            pivot_indices = sorted(rng.sample(range(1, _DPO_CHUNK_LEN), n_pivots))

            for p in pivot_indices:
                rollout_len = _DPO_CHUNK_LEN - p
                pivot_state = emu_states[cs + p]

                best_ram     = None
                best_actions = None
                best_score   = -float("inf")

                for k in range(n_rollouts):
                    np_rng = np.random.default_rng(seed * 10_000_000 + ci * 10_000 + p * 100 + k)
                    r_ram, r_acts, r_reward = _run_rollout(env, pivot_state, rollout_len, np_rng)
                    if r_reward < orig_reward and r_reward > best_score:
                        best_score   = r_reward
                        best_ram     = r_ram
                        best_actions = r_acts

                if best_ram is None:
                    continue

                c_ram  = ram_arr[cs : cs + _DPO_CHUNK_LEN]
                c_acts = combined_arr[cs : cs + _DPO_CHUNK_LEN]

                r_ram_full  = np.empty((_DPO_CHUNK_LEN, 2048), dtype=np.uint8)
                r_acts_full = np.empty(_DPO_CHUNK_LEN, dtype=np.int16)
                r_ram_full[:p]  = c_ram[:p]
                r_ram_full[p:]  = best_ram
                r_acts_full[:p] = c_acts[:p]
                r_acts_full[p:] = best_actions

                chosen_rams.append(c_ram.copy())
                chosen_acts.append(c_acts.copy())
                rejected_rams.append(r_ram_full)
                rejected_acts.append(r_acts_full)
                pivots.append(p)

        env.close()

        if not chosen_rams:
            return None

        return dict(
            chosen_ram       = np.stack(chosen_rams).astype(np.uint8),
            chosen_actions   = np.stack(chosen_acts).astype(np.int16),
            rejected_ram     = np.stack(rejected_rams).astype(np.uint8),
            rejected_actions = np.stack(rejected_acts).astype(np.int16),
            pivot            = np.array(pivots, dtype=np.int16),
        )

    def __repr__(self) -> str:
        return f"DPODataSample(uuid={self.uuid!r})"
