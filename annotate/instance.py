"""
BCData Base Classes
===================
BCDataSample  - universal data holder for one recorded episode.

Recording / Training-data Sequence
===================================
Each logical step = SKIP=3 NES frames (logical FPS=20).
The FIRST NES frame of each skip group is captured as the model input,
giving the model 2 extra NES frames (~33 ms at 60 fps) of inference time.

  t=0
  ┌─────────────────────────────────────────────────────────────┐
  │  ENV                                                        │
  │  env.reset() + rewind_state()                               │
  │  NES:  nes_0 recorded as img_features[0]                    │
  │  model:  predicts action_0 from nes_0                       │
  │  input:  action_0                                           │
  │  NES:    nes_1 → nes_2 → nes_3   (SKIP=3)                   │
  └─────────────────────────────────────────────────────────────┘

  t=1
  ┌─────────────────────────────────────────────────────────────┐
  │  ENV                                                        │
  │  NES:    nes_1 recorded as img_features[1]                  │
  │  model:  predicts action_1 from nes_1                       │
  │          (this gives the model the duration of nes_2 and    │
  │           nes_3 to compute the prediction in real-time)     │
  │  input:  action_1                                           │
  │  NES:    nes_4 → nes_5 → nes_6   (SKIP=3)                   │
  └─────────────────────────────────────────────────────────────┘

  ...  (pattern repeats for all N actions)

Output file: bc_features.npz
  img_features : float16  (N, 1, embed_dim)
  dpad         : int8     (N,)    encoded dpad class 0..8
  button       : int8     (N,)    encoded button class 0..3
  text         : float16  (N, 1, 768)  Gemma embeddings; zeros if use_text=False

Training chunk  (get_chunk(start, T=200)):
  position t:  img_features[start+t], dpad[start+t], button[start+t], text[start+t]
  model target: predict action_t from img[0..t] and action[0..t-1]  (causal)
"""

import os
import tempfile  # used in compute_features when use_text=True

import numpy as np

from contra.replay import replay_actions
from pixel2play.model.nes_actions import encode

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
    def create(cls, npz_path: str, game: str) -> "BCDataSample":
        """Return a BCDataSample handle pointing to the output path.
        Call compute_features() afterwards to produce the training data.
        """
        rec_id  = os.path.splitext(os.path.basename(npz_path))[0]
        out_dir = os.path.join(os.path.dirname(__file__), "bc_data", game)
        os.makedirs(out_dir, exist_ok=True)
        return cls(os.path.join(out_dir, f"{rec_id}.npz"))

    def save_from_features(
        self,
        img_features: np.ndarray,
        dpad: np.ndarray,
        button: np.ndarray,
        extra: dict | None = None,
    ) -> str:
        """Persist pre-computed img_features plus action labels. Returns output path."""
        arrays = {} if extra is None else dict(extra)
        arrays.update(img_features=img_features, dpad=dpad, button=button)
        np.savez(self.features_path, **arrays)
        return self.features_path

    def compute_features(
        self,
        npz_path: str,
        tokenizer,
        use_text: bool = False,
        chunk_size: int = 32,
        device: str = "cuda",
    ) -> str:
        """Step the emulator, encode frames in chunks of chunk_size, save features.

        Peak frame memory = chunk_size frames (uint8) + chunk_size frames (float32)
        regardless of recording length.

        Saves bc_features.npz with:
            img_features : float16  (N, n_tokens, D)
            dpad         : int8     (N,)
            button       : int8     (N,)
            text         : float16  (N, 1, 768)   only present if use_text=True

        Returns the path to the saved file.
        """
        import torch
        import torch.nn.functional as F
        import stable_retro as retro
        from contra.replay import rewind_state, GAME, SKIP
        from contra.events import EV_LEVELUP, EV_GAME_CLEAR

        out_path = self.features_path
        if os.path.isfile(out_path):
            print(f"  {self.uuid}: already exists, skipping")
            return out_path

        npz_data    = np.load(npz_path, allow_pickle=True)
        raw_actions = npz_data["actions"]                          # (N, 9) uint8
        N           = len(raw_actions)

        # --- Encode dpad / button class indices ---
        dpad   = np.empty(N, dtype=np.int8)
        button = np.empty(N, dtype=np.int8)
        for i, act in enumerate(raw_actions):
            dpad[i], button[i] = encode(_nes_keys(act))

        tokenizer = tokenizer.to(device).eval()
        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        def encode_chunk(frames: list) -> np.ndarray:
            """Preprocess and encode a list of (H, W, 3) uint8 frames. Returns float16."""
            arr = np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32)
            arr = (arr / 255.0 - _mean) / _std
            t = torch.from_numpy(arr)
            if t.shape[-2:] != (192, 192):
                t = F.interpolate(t, (192, 192), mode="bilinear", align_corners=False)
            with torch.no_grad():
                feats = tokenizer(t.to(device).unsqueeze(1)).squeeze(1)  # (B, n_tokens, D)
            return feats.cpu().float().numpy().astype(np.float16)

        # --- Text annotation (requires full video for Gemini) ---
        extra = {}
        raw_frames_for_text = None
        if use_text:
            from contra.replay import save_video
            from annotate.get_text_annotation import annotate_video, _get_gemma_st
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")
            fps = int(npz_data.get("fps", 20))

            result = replay_actions(npz_path, want_video=True, verbose=False)
            assert result["result"] != "lose", \
                f"{self.uuid}: replay ended in 'lose' — skipping feature computation"
            raw_frames_for_text = result["video"]                  # (N+1, H, W, 3) uint8

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name
            try:
                save_video(raw_frames_for_text, video_path)
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

        # --- Stream replay + chunk encode ---
        all_feats = []

        if raw_frames_for_text is not None:
            # Already have frames from the use_text replay; encode them in chunks
            for i in range(0, len(raw_frames_for_text), chunk_size):
                chunk = list(raw_frames_for_text[i:i + chunk_size])
                all_feats.append(encode_chunk(chunk))
            del raw_frames_for_text
        else:
            # Step emulator directly; never hold more than chunk_size frames at once
            env = retro.make(
                game=GAME,
                state=retro.State.NONE,
                use_restricted_actions=retro.Actions.ALL,
                obs_type=retro.Observations.IMAGE,
                render_mode=None,
                inttype=retro.data.Integrations.CUSTOM_ONLY,
            )
            env.reset()
            rewind_state(env, bytes(npz_data["initial_state"]))

            frame_buf    = [env.em.get_screen().copy()]            # nes_0
            leveled_up   = False
            game_cleared = False

            for act in raw_actions:
                pre_ram = env.unwrapped.get_ram().copy()
                act_arr = np.asarray(act, dtype=np.uint8)
                for i in range(SKIP):
                    obs, _, _, _, _ = env.step(act_arr.copy())
                    if i == 0:
                        frame_buf.append(obs.copy())
                curr_ram = env.unwrapped.get_ram()

                if EV_LEVELUP.trigger(pre_ram, curr_ram):
                    leveled_up = True
                if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
                    game_cleared = True

                if len(frame_buf) >= chunk_size:
                    all_feats.append(encode_chunk(frame_buf))
                    frame_buf = []

            env.close()

            outcome = "game_clear" if game_cleared else "level_up" if leveled_up else "lose"
            assert outcome != "lose", \
                f"{self.uuid}: replay ended in 'lose' — skipping feature computation"

            if frame_buf:
                all_feats.append(encode_chunk(frame_buf))

        img_features = np.concatenate(all_feats, axis=0)[:-1]     # drop final nes_N frame

        return self.save_from_features(img_features, dpad, button, extra)

    @property
    def has_features(self) -> bool:
        return os.path.isfile(self.features_path)

    def __repr__(self) -> str:
        return f"BCDataSample(uuid={self.uuid!r})"
