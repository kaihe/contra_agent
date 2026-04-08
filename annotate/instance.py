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

    def compute_features(
        self,
        npz_path: str,
        tokenizer,
        use_text: bool = False,
        batch_size: int = 64,
        device: str = "cuda"
    ) -> str:
        """Record a video via replay_actions, extract frames, tokenize, optionally embed text.

        Saves bc_features.npz with:
            img_features : float16  (N+1, n_tokens, D)
            dpad         : int8     (N,)
            button       : int8     (N,)
            text         : float16  (N, 1, 768)   only present if use_text=True

        Returns the path to the saved file.
        """
        import torch
        from contra.replay import save_video

        out_path = self.features_path
        if os.path.isfile(out_path):
            print(f"  {self.uuid}: already exists, skipping")
            return out_path

        npz_data    = np.load(npz_path, allow_pickle=True)
        raw_actions = npz_data["actions"]                          # (N, 9) uint8
        fps         = int(npz_data.get("fps", 20))
        N           = len(raw_actions)

        # --- Encode dpad / button class indices ---
        dpad   = np.empty(N, dtype=np.int8)
        button = np.empty(N, dtype=np.int8)
        for i, act in enumerate(raw_actions):
            dpad[i], button[i] = encode(_nes_keys(act))

        # --- Replay env, get frames ---
        result     = replay_actions(npz_path, want_video=True, verbose=False)
        assert result["result"] != "lose", f"{self.uuid}: replay ended in 'lose' — skipping feature computation"
        raw_frames = result["video"]                               # (N+1, H, W, 3) uint8

        # --- Text embeddings (optional) — save video only if Gemini needs it ---
        arrays = {}
        if use_text:
            from annotate.get_text_annotation import annotate_video, _get_gemma_st
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name
            try:
                save_video(raw_frames, video_path)
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
            arrays["text"] = text

        # --- Tokenize frames ---
        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        frames   = (raw_frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - _mean) / _std
        frames_t = torch.from_numpy(frames)

        # Resize to tokenizer input resolution if needed
        if frames_t.shape[-2:] != (192, 192):
            frames_t = torch.nn.functional.interpolate(
                frames_t, size=(192, 192), mode="bilinear", align_corners=False
            )

        tokenizer = tokenizer.to(device).eval()
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(frames_t), batch_size):
                batch = frames_t[i:i + batch_size].to(device).unsqueeze(1)  # (B, 1, 3, H, W)
                feats = tokenizer(batch).squeeze(1)                          # (B, n_tokens, D)
                all_feats.append(feats.cpu().float().numpy())
        img_features = np.concatenate(all_feats, axis=0).astype(np.float16)  # (N+1, n_tokens, D)

        # Drop the final nes_N frame so img_features length matches actions
        img_features = img_features[:-1]

        # --- Save ---
        arrays.update(img_features=img_features, dpad=dpad, button=button)
        np.savez(out_path, **arrays)

        return out_path

    @property
    def has_features(self) -> bool:
        return os.path.isfile(self.features_path)

    def __repr__(self) -> str:
        return f"BCDataSample(uuid={self.uuid!r})"
