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
import tempfile  # used in compute_features when use_text=True

import numpy as np

from contra.replay import replay_actions
from pixel2play.model.nes_actions import encode

_NOOP = np.zeros(9, dtype=np.uint8)

_NOOP = np.zeros(9, dtype=np.uint8)

# NES MultiBinary(9): [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
_NES_KEY_MAP = [(0, "f"), (4, "w"), (5, "s"), (6, "a"), (7, "d"), (8, "j")]


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

    def save_from_features(
        self,
        ram: np.ndarray,
        dpad: np.ndarray,
        button: np.ndarray,
        extra: dict | None = None,
    ) -> str:
        """Persist RAM snapshots plus action labels. Returns output path."""
        arrays = {} if extra is None else dict(extra)
        arrays.update(ram=ram, dpad=dpad, button=button)
        np.savez(self.features_path, **arrays)
        return self.features_path

    def compute_features(
        self,
        npz_path: str,
        use_text: bool = False,
        device: str = "cuda",
    ) -> str:
        """Step the emulator, collect RAM snapshots per step, save features.

        Saves bc_features.npz with:
            ram    : uint8   (N, 2048)
            dpad   : int8    (N,)
            button : int8    (N,)
            text   : float16 (N, 1, 768)   only present if use_text=True

        Returns the path to the saved file.
        """
        import stable_retro as retro
        from contra.replay import rewind_state, step_env, GAME, SKIP
        from contra.events import EV_LEVELUP, EV_GAME_CLEAR

        out_path = self.features_path
        if os.path.isfile(out_path):
            print(f"  {self.uuid}: already exists, skipping")
            return out_path

        npz_data    = np.load(npz_path, allow_pickle=True)
        raw_actions = npz_data["actions"]                          # (N, 9) uint8
        N           = len(raw_actions)

        dpad   = np.empty(N, dtype=np.int8)
        button = np.empty(N, dtype=np.int8)
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
        rewind_state(env, bytes(npz_data["initial_state"]))

        ram_buf.append(env.unwrapped.get_ram().copy())             # ram[0]: initial state
        leveled_up   = False
        game_cleared = False
        raw_frames   = None

        if use_text:
            raw_frames = [env.em.get_screen().copy()]              # for video annotation

        for i, act in enumerate(raw_actions):
            act_arr   = np.asarray(act, dtype=np.uint8)
            pre_ram   = env.unwrapped.get_ram().copy()
            cur_state = env.em.get_state()

            # Apply original action
            for j in range(SKIP):
                obs, _, _, _, _ = env.step(act_arr.copy())
                if use_text and j == 0:
                    raw_frames.append(obs.copy())
            ram_orig  = env.unwrapped.get_ram().copy()
            next_orig = env.em.get_state()

            # Test NOOP from same state
            rewind_state(env, cur_state)
            step_env(env, _NOOP)
            ram_noop = env.unwrapped.get_ram().copy()

            # If action had no effect, label as no-op (env stays in noop state)
            # If effective, rewind to original progression
            if np.array_equal(ram_orig, ram_noop):
                dpad[i], button[i] = encode(_nes_keys(_NOOP))
            else:
                dpad[i], button[i] = encode(_nes_keys(act_arr))
                rewind_state(env, next_orig)

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

        # ram_buf has N+1 entries; drop the final one so ram[i] aligns with action[i]
        ram = np.stack(ram_buf[:-1], axis=0)                      # (N, 2048) uint8

        # --- Text annotation ---
        extra = {}
        if use_text:
            from contra.replay import save_video
            from annotate.get_text_annotation import annotate_video, _get_gemma_st
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")
            fps = int(npz_data.get("fps", 20))

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name
            try:
                save_video(np.stack(raw_frames), video_path)
                _, macros = annotate_video(video_path, api_key, fps)
            finally:
                os.unlink(video_path)

            text  = np.zeros((N, 1, 768), dtype=np.float16)
            gemma = _get_gemma_st()
            for macro in macros:
                vec   = gemma.encode(macro["instruction"], convert_to_numpy=True)
                start = macro["start_frame"] - 1
                end   = macro["end_frame"] if macro["end_frame"] is not None else N
                text[start:end] = vec.reshape(1, 768).astype(np.float16)
            extra["text"] = text

        return self.save_from_features(ram, dpad, button, extra)

    @property
    def has_features(self) -> bool:
        return os.path.isfile(self.features_path)

    def __repr__(self) -> str:
        return f"BCDataSample(uuid={self.uuid!r})"
