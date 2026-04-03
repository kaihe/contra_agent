"""
NES dataset: loads recordings from bc_data and returns fixed-length chunks.

Each recording directory contains:
  192x192.mp4       – raw gameplay video (20 fps, uint8)
  annotation.proto  – per-frame action labels + optional text embeddings

A chunk of T consecutive frames yields:
  frames  : (T, 3, 192, 192)  float32, ImageNet-normalised
  dpad    : (T,)               int64, 0..8
  button  : (T,)               int64, 0..3
  text    : (T, 1, 768)        float32, Gemma embedding (zeros if absent)
"""

import os
import sys
from typing import List

import av
import numpy as np
import torch
from torch.utils.data import Dataset

# annotate lives one level up from pixel2play/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from annotate.proto.video_annotation_pb2 import VideoAnnotation  # noqa: E402

from pixel2play.model.nes_actions import encode  # noqa: E402

_TEXT_MODEL  = "gemini-3-flash-preview"
_TEXT_KEY    = "gemma"
_TEXT_DIM    = 768

# ImageNet normalisation (EfficientNet backbone expects this)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Recording:
    """Lazy-loading wrapper for one recording directory."""

    def __init__(self, rec_dir: str):
        self.rec_dir = rec_dir
        self._meta_loaded = False
        _feat_path = os.path.join(rec_dir, "img_features.npy")
        self._img_features = np.load(_feat_path, mmap_mode="r") if os.path.isfile(_feat_path) else None

    def _load_meta(self):
        """Load only the annotation proto (cheap). Populates n, dpad, button, text."""
        if self._meta_loaded:
            return

        va = VideoAnnotation()
        with open(os.path.join(self.rec_dir, "annotation.proto"), "rb") as f:
            va.ParseFromString(f.read())

        fps = va.metadata.frames_per_second or 20.0
        fas = va.frame_annotations[1:]   # index 0 is blank initial state
        n = len(fas)

        dpad   = np.empty(n, dtype=np.int64)
        button = np.empty(n, dtype=np.int64)
        text   = np.zeros((n, 1, _TEXT_DIM), dtype=np.float32)

        for i, fa in enumerate(fas):
            keys = list(fa.user_action.keyboard.keys)
            dpad[i], button[i] = encode(keys)

            if fa.frame_text_annotation:
                fta = fa.frame_text_annotation[0]
                gem = fta.text_embedding_dict.get(_TEXT_MODEL, None)
                if gem is not None:
                    emb = gem.text_embeddings.get(_TEXT_KEY, None)
                    if emb is not None and emb.values:
                        vec = np.array(emb.values, dtype=np.float32).reshape(list(emb.shape))
                        span = round(fta.duration * fps)
                        text[i:i + span] = vec

        self._n      = n
        self._dpad   = torch.from_numpy(dpad)
        self._button = torch.from_numpy(button)
        self._text   = torch.from_numpy(text)
        self._meta_loaded = True

    def __len__(self):
        self._load_meta()
        return self._n

    def get_chunk(self, start: int, length: int):
        """Return a chunk of frames (or precomputed features) with actions."""
        self._load_meta()
        end = start + length

        if self._img_features is not None:
            # Fast path: load precomputed EfficientNet tokens from .npy
            img = torch.from_numpy(
                np.array(self._img_features[start:end], dtype=np.float32)
            )  # (T, n_tokens, D)
        else:
            # Slow path: seek + decode video frames on demand
            raw_frames = []
            with av.open(os.path.join(self.rec_dir, "192x192.mp4")) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)

                if start > 0:
                    target_ts = int(start / fps / float(stream.time_base))
                    container.seek(target_ts, stream=stream, backward=True)

                for frame in container.decode(video=0):
                    current = round(float(frame.pts * stream.time_base) * fps)
                    if current < start:
                        continue
                    if current >= end:
                        break
                    raw_frames.append(frame.to_ndarray(format="rgb24"))
                    if len(raw_frames) >= length:
                        break

            img = torch.from_numpy(np.stack(raw_frames)).permute(0, 3, 1, 2).float() / 255.0
            img = (img - _MEAN) / _STD  # (T, 3, H, W)

        return (
            img,                        # (T, n_tokens, D) or (T, 3, H, W)
            self._dpad[start:end],      # (T,)
            self._button[start:end],    # (T,)
            self._text[start:end],      # (T, 1, 768)
        )


class NESDataset(Dataset):
    def __init__(self, data_root: str, n_steps: int = 200, stride: int = 1):
        """
        Args:
            data_root: directory containing recording sub-directories.
            n_steps:   sequence length T.
            stride:    step between consecutive chunks.
        """
        self.n_steps = n_steps

        rec_dirs = sorted([
            os.path.join(data_root, d)
            for d in os.listdir(data_root)
            if os.path.isfile(os.path.join(data_root, d, "annotation.proto"))
        ])
        if not rec_dirs:
            raise FileNotFoundError(f"No recordings found in {data_root!r}")

        self.recordings: List[Recording] = [Recording(d) for d in rec_dirs]

        # Build a flat index: list of (recording_idx, chunk_start)
        self._index: List[tuple] = []
        for rec_idx, rec in enumerate(self.recordings):
            n = len(rec)
            for start in range(0, n - n_steps + 1, stride):
                self._index.append((rec_idx, start))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        rec_idx, start = self._index[idx]
        return self.recordings[rec_idx].get_chunk(start, self.n_steps)
