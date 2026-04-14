"""
NES dataset: loads bc_features.npz recordings and returns fixed-length chunks.

Each recording contains bc_features.npz with:
  ram    : uint8   (N, 2048)          NES RAM snapshot per step
  dpad   : int8    (N,)               dpad class 0..8
  button : int8    (N,)               button class 0..3
  text   : float16 (N, 1, 768)        Gemma embedding (zeros if absent)

Chunks are non-overlapping with stride=T. The final tail chunk is padded with
the last RAM/action repeated, and a valid_mask marks the real frames.

A chunk of T consecutive steps yields:
  ram        : (T, 2048)  uint8
  dpad       : (T,)       int64,  0..8
  button     : (T,)       int64,  0..3
  text       : (T, 1, 768) float32
  valid_mask : (T,)       bool    -- False for padding steps
"""

import logging
import os
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class Recording:
    """Memory-mapped wrapper for one bc_features.npz recording."""

    def __init__(self, features_path: str, n_text_tokens: int = 1):
        self.path = features_path
        data = np.load(features_path, mmap_mode="r")
        self._ram    = data["ram"]                                           # (N, 2048) uint8
        self._dpad   = torch.from_numpy(data["dpad"].astype(np.int64))      # (N,)
        self._button = torch.from_numpy(data["button"].astype(np.int64))    # (N,)
        self._n      = len(self._dpad)

        if "text" in data and n_text_tokens > 0:
            self._text = torch.from_numpy(data["text"].astype(np.float32))[:, :n_text_tokens, :]
        else:
            self._text = torch.zeros(self._n, n_text_tokens, 768)

    def __len__(self) -> int:
        return self._n

    def get_chunk(self, start: int, length: int):
        valid_len = min(length, self._n - start)

        ram    = torch.from_numpy(self._ram[start:start + valid_len].copy())
        dpad   = self._dpad[start:start + valid_len]
        button = self._button[start:start + valid_len]
        text   = self._text[start:start + valid_len]

        valid_mask = torch.zeros(length, dtype=torch.bool)
        valid_mask[:valid_len] = True

        if valid_len < length:
            pad = length - valid_len
            ram    = torch.cat([ram,    ram[-1:].expand(pad, -1)],              dim=0)
            dpad   = torch.cat([dpad,   torch.zeros(pad, dtype=dpad.dtype)],    dim=0)
            button = torch.cat([button, torch.zeros(pad, dtype=button.dtype)],  dim=0)
            text   = torch.cat([text,   text[-1:].expand(pad, -1, -1)],         dim=0)

        return ram, dpad, button, text, valid_mask


class NESDataset(Dataset):
    def __init__(self, data_root: str, n_steps: int = 200, n_text_tokens: int = 1):
        """
        Args:
            data_root:     directory containing .npz recordings.
            n_steps:       chunk length T.
            n_text_tokens: number of text tokens to keep per step.
        """
        self.n_steps = n_steps

        npz_files = sorted([
            os.path.join(data_root, f)
            for f in os.listdir(data_root)
            if f.endswith(".npz")
        ])
        if not npz_files:
            raise FileNotFoundError(f"No .npz recordings found in {data_root!r}")

        self.recordings: List[Recording] = [Recording(p, n_text_tokens=n_text_tokens) for p in npz_files]
        self._build_index(rng=None)

    @classmethod
    def from_recordings(cls, recordings: List[Recording], n_steps: int) -> "NESDataset":
        """Construct a dataset from an existing list of Recording objects (no file I/O)."""
        obj = object.__new__(cls)
        obj.n_steps = n_steps
        obj.recordings = recordings
        obj._build_index(rng=None)
        return obj

    def _build_index(self, rng: np.random.RandomState = None):
        """Build chunk index. If rng is provided, each recording gets a random
        start offset in [0, min(T-1, len-1)] before striding by T."""
        self._index = []
        for rec_idx, rec in enumerate(self.recordings):
            if rng is not None:
                cap = min(self.n_steps - 1, len(rec) - 1)
                offset = int(rng.randint(0, cap + 1))
            else:
                offset = 0
            for start in range(offset, len(rec), self.n_steps):
                self._index.append((rec_idx, start))

    def resample(self, rng: np.random.RandomState):
        """Rebuild chunk boundaries with a new random jitter. Call at epoch start."""
        self._build_index(rng=rng)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        rec_idx, start = self._index[idx]
        t0 = time.perf_counter()
        result = self.recordings[rec_idx].get_chunk(start, self.n_steps)
        elapsed = time.perf_counter() - t0
        if elapsed > 0.1:
            logging.warning(f"[timing] slow get_chunk rec={rec_idx} start={start} {elapsed:.3f}s")
        return result
