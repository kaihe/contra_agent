"""
NES dataset: loads bc_features recordings and returns fixed-length chunks.

Two storage formats are supported:

  Individual npz (legacy):
    data_root/
      rec_0001.npz   # keys: ram (N,2048) uint8, dpad (N,) int8, button (N,) int8

  Sharded npy (fast, mmap-able):
    data_root/
      shard_0000/
        ram.npy      # (total_steps, 2048) uint8
        dpad.npy     # (total_steps,) int8
        button.npy   # (total_steps,) int8
        index.npy    # (n_recs, 2) int64: (start_step, length) per recording

  Pack shards with:
    python annotate/gen_data.py pack <src_dir> --output-dir <shard_dir>

Chunks are non-overlapping with stride=T. The final tail chunk is padded with
the last RAM/action repeated, and a valid_mask marks the real frames.

A chunk of T consecutive steps yields:
  ram        : (T, 2048)  uint8
  dpad       : (T,)       int64,  0..8
  button     : (T,)       int64,  0..3
  text       : (T, 1, 768) float32   (zeros — text not used for RAM-only training)
  valid_mask : (T,)       bool    -- False for padding steps
"""

import logging
import os
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Recording backends
# ---------------------------------------------------------------------------

class Recording:
    """Lazy-loading wrapper for one bc_features.npz recording.

    Only the length is read at construction time; arrays are loaded from disk
    on each get_chunk() call.  The OS page cache handles repeated access.
    """

    def __init__(self, features_path: str):
        self.path = features_path
        data = np.load(features_path)
        self._n = int(len(data["dpad"]))
        data.close()

    def __len__(self) -> int:
        return self._n

    def get_chunk(self, start: int, length: int):
        valid_len = min(length, self._n - start)
        sl = slice(start, start + valid_len)

        data   = np.load(self.path)
        ram    = torch.from_numpy(data["ram"][sl].astype(np.uint8))
        dpad   = torch.from_numpy(data["dpad"][sl].astype(np.int64))
        button = torch.from_numpy(data["button"][sl].astype(np.int64))
        data.close()

        return _pad_chunk(ram, dpad, button, valid_len, length)


class ShardRecording:
    """One recording stored inside a mmap-able shard directory.

    The shard's .npy arrays are memory-mapped: all DataLoader workers that
    access the same shard share the OS page-cache pages, with zero per-worker
    RAM overhead for the data itself.
    """

    def __init__(self, shard_ram: np.ndarray, shard_dpad: np.ndarray,
                 shard_button: np.ndarray, start: int, length: int, name: str = ""):
        self._ram    = shard_ram
        self._dpad   = shard_dpad
        self._button = shard_button
        self._start  = start
        self._n      = length
        self.path    = name   # original filename — used for level detection in train.py

    def __len__(self) -> int:
        return self._n

    def get_chunk(self, start: int, length: int):
        valid_len = min(length, self._n - start)
        abs_start = self._start + start
        sl = slice(abs_start, abs_start + valid_len)

        ram    = torch.from_numpy(self._ram[sl].astype(np.uint8))
        dpad   = torch.from_numpy(self._dpad[sl].astype(np.int64))
        button = torch.from_numpy(self._button[sl].astype(np.int64))

        return _pad_chunk(ram, dpad, button, valid_len, length)


def _pad_chunk(ram, dpad, button, valid_len, length):
    valid_mask = torch.zeros(length, dtype=torch.bool)
    valid_mask[:valid_len] = True

    if valid_len < length:
        pad = length - valid_len
        ram    = torch.cat([ram,    ram[-1:].expand(pad, -1)],           dim=0)
        dpad   = torch.cat([dpad,   torch.zeros(pad, dtype=dpad.dtype)], dim=0)
        button = torch.cat([button, torch.zeros(pad, dtype=button.dtype)], dim=0)

    return ram, dpad, button, valid_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_recordings(data_root: str) -> List:
    """Auto-detect format and return a list of Recording or ShardRecording objects."""
    shard_dirs = sorted(
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and d.startswith("shard_")
    )
    if shard_dirs:
        recordings = []
        for shard_dir in shard_dirs:
            ram    = np.load(os.path.join(shard_dir, "ram.npy"),    mmap_mode="r")
            dpad   = np.load(os.path.join(shard_dir, "dpad.npy"),   mmap_mode="r")
            button = np.load(os.path.join(shard_dir, "button.npy"), mmap_mode="r")
            index  = np.load(os.path.join(shard_dir, "index.npy"))
            names_path = os.path.join(shard_dir, "names.npy")
            names  = np.load(names_path, allow_pickle=True) if os.path.exists(names_path) else [""] * len(index)
            for (start, length), name in zip(index, names):
                recordings.append(ShardRecording(ram, dpad, button, int(start), int(length), str(name)))
        return recordings

    npz_files = sorted(
        os.path.join(data_root, f)
        for f in os.listdir(data_root) if f.endswith(".npz")
    )
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files or shard_NNNN/ directories found in {data_root!r}"
        )
    return [Recording(p) for p in npz_files]


class NESDataset(Dataset):
    def __init__(self, data_root: str, n_steps: int = 200):
        self.n_steps = n_steps
        self.recordings: List = _load_recordings(data_root)
        self._build_index(rng=None)

    @classmethod
    def from_recordings(cls, recordings: List, n_steps: int) -> "NESDataset":
        obj = object.__new__(cls)
        obj.n_steps = n_steps
        obj.recordings = recordings
        obj._build_index(rng=None)
        return obj

    def _build_index(self, rng: np.random.RandomState = None):
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
