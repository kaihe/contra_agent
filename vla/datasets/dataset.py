"""ContraVLA dataset and collate function."""

from __future__ import annotations

import glob
import io
import os
from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Fixed text instruction per level (level_id → string)
LEVEL_TEXTS: dict[int, str] = {
    0: "Run right through the jungle, shoot soldiers and gun emplacements, dodge bullets and grenades, and destroy the enemy base entrance.",
    1: "Level 2: infiltrate the base, avoid traps, and destroy the enemy core.",
    2: "Level 3: climb the waterfalls and defeat the enemies.",
    3: "Level 4: navigate the enemy base and eliminate all threats.",
    4: "Level 5: fight through the snow field and advance.",
    5: "Level 6: infiltrate the energy zone and destroy the generator.",
    6: "Level 7: battle through the alien lair.",
    7: "Level 8: defeat the alien boss and end the invasion.",
}

# ImageNet normalisation constants (SmolVLM processor default)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """frames: [2, H, W, 3] uint8 → [2, 3, H, W] float32, ImageNet-normalised"""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)  # [2, 3, H, W]
    return (t - _MEAN) / _STD


class ContraVLADataset(Dataset):
    """
    Loads pre-extracted VLA shard files produced by preprocess.py.

    Supports two shard layouts:
      New (dedup + JPEG):
        shard_NNNN_frames.npz   blob [M] uint8, offsets [M+1] int32
        shard_NNNN_indices.npy  [S, 2] int32
        shard_NNNN_proprio.npy  [S, 118] float32
        shard_NNNN_actions.npy  [S, T] int8
      Old (raw + mmap):
        shard_NNNN_frames.npy   [S, 2, H, W, 3] uint8
        shard_NNNN_meta.npz     proprio [S, 118], actions [S, T]

    __getitem__ returns:
        images   [2, 3, H, W] float32
        proprio  [118]  float32
        actions  [T]    int64
        level_id int    (0 = Level1)
    """

    def __init__(self, shard_dir: str, level_id: int = 0) -> None:
        self._frame_paths: list[str]        = []
        self._proprio:     list[np.ndarray] = []   # fully in RAM
        self._actions:     list[np.ndarray] = []   # fully in RAM
        self._frame_indices: list[np.ndarray] = []
        self._index:       list[tuple[int, int]] = []
        self._format:      str = ""

        # mmap format: blob/offsets as flat .npy — opened once, shared across workers
        self._mmap_blobs:   list[np.ndarray] = []
        self._mmap_offsets: list[np.ndarray] = []

        mmap_frames  = sorted(glob.glob(os.path.join(shard_dir, "*_frames_blob.npy")))
        legacy_frames = sorted(glob.glob(os.path.join(shard_dir, "*_frames.npz")))
        old_frames   = sorted(glob.glob(os.path.join(shard_dir, "*_frames.npy")))

        if mmap_frames:
            self._format = "jpeg_mmap"
            for blob_path in mmap_frames:
                base = blob_path.replace("_frames_blob.npy", "")
                offsets_path = base + "_frames_offsets.npy"
                indices_path = base + "_indices.npy"
                proprio_path = base + "_proprio.npy"
                actions_path = base + "_actions.npy"
                if not all(os.path.isfile(p) for p in [offsets_path, indices_path, proprio_path, actions_path]):
                    continue
                indices = np.load(indices_path)
                S = len(indices)
                si = len(self._mmap_blobs)
                self._mmap_blobs.append(np.load(blob_path, mmap_mode="r"))
                self._mmap_offsets.append(np.load(offsets_path, mmap_mode="r"))
                self._frame_indices.append(indices)
                self._proprio.append(np.load(proprio_path))
                self._actions.append(np.load(actions_path))
                self._index.extend((si, j) for j in range(S))

        elif legacy_frames:
            self._format = "jpeg_dedup"
            for frames_path in legacy_frames:
                base = frames_path.replace("_frames.npz", "")
                indices_path = base + "_indices.npy"
                proprio_path = base + "_proprio.npy"
                actions_path = base + "_actions.npy"
                if not all(os.path.isfile(p) for p in [indices_path, proprio_path, actions_path]):
                    continue
                indices = np.load(indices_path)
                S = len(indices)
                si = len(self._frame_paths)
                self._frame_paths.append(frames_path)
                self._frame_indices.append(indices)
                self._proprio.append(np.load(proprio_path))
                self._actions.append(np.load(actions_path))
                self._index.extend((si, j) for j in range(S))

        elif old_frames:
            self._format = "raw"
            for frames_path in old_frames:
                meta_path = frames_path.replace("_frames.npy", "_meta.npz")
                if not os.path.isfile(meta_path):
                    continue
                meta = np.load(meta_path)
                S = len(meta["proprio"])
                si = len(self._frame_paths)
                self._frame_paths.append(frames_path)
                self._proprio.append(meta["proprio"])
                self._actions.append(meta["actions"])
                self._index.extend((si, j) for j in range(S))

        if not self._index:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

        self.level_id = level_id
        self._mmaps: dict[int, np.ndarray] = {}                        # raw format
        self._jpeg_data: OrderedDict[int, tuple] = OrderedDict()       # legacy npz, LRU=2

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        si, j = self._index[i]

        if self._format == "jpeg_mmap":
            blob    = self._mmap_blobs[si]
            offsets = self._mmap_offsets[si]
            idx0, idx1 = self._frame_indices[si][j]
            f0 = np.array(Image.open(io.BytesIO(bytes(blob[offsets[idx0]:offsets[idx0 + 1]]))).convert("RGB"))
            f1 = np.array(Image.open(io.BytesIO(bytes(blob[offsets[idx1]:offsets[idx1 + 1]]))).convert("RGB"))
            frames = np.stack([f0, f1])  # [2, H, W, 3] uint8

        elif self._format == "raw":
            if si not in self._mmaps:
                self._mmaps[si] = np.load(self._frame_paths[si], mmap_mode="r")
            frames = np.array(self._mmaps[si][j])   # [2, H, W, 3] uint8, copy to RAM

        else:  # jpeg_dedup (legacy npz)
            if si not in self._jpeg_data:
                if len(self._jpeg_data) >= 2:
                    self._jpeg_data.popitem(last=False)  # evict LRU
                data = np.load(self._frame_paths[si])
                self._jpeg_data[si] = (data["blob"], data["offsets"])
            else:
                self._jpeg_data.move_to_end(si)
            blob, offsets = self._jpeg_data[si]
            idx0, idx1 = self._frame_indices[si][j]
            f0 = np.array(Image.open(io.BytesIO(bytes(blob[offsets[idx0]:offsets[idx0 + 1]]))).convert("RGB"))
            f1 = np.array(Image.open(io.BytesIO(bytes(blob[offsets[idx1]:offsets[idx1 + 1]]))).convert("RGB"))
            frames = np.stack([f0, f1])  # [2, H, W, 3] uint8

        images = _preprocess_frames(frames)

        return {
            "images":   images,
            "proprio":  torch.from_numpy(self._proprio[si][j].copy()),
            "actions":  torch.from_numpy(self._actions[si][j].astype(np.int64)),
            "level_id": self.level_id,
        }


def build_collate_fn(tokenizer, max_text_len: int = 32) -> Callable:
    """Returns a collate_fn that tokenises level text and pads input_ids."""

    def collate_fn(batch: list[dict]) -> dict:
        images   = torch.stack([b["images"]  for b in batch])   # [B, 2, 3, H, W]
        proprio  = torch.stack([b["proprio"] for b in batch])   # [B, 118]
        actions  = torch.stack([b["actions"] for b in batch])   # [B, T]
        texts    = [LEVEL_TEXTS[b["level_id"]] for b in batch]

        tok = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_text_len,
            truncation=True,
        )
        return {
            "images":    images,
            "input_ids": tok["input_ids"],          # [B, L]
            "proprio":   proprio,
            "actions":   actions,
        }

    return collate_fn
