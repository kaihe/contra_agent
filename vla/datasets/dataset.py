"""Contra VLA behavior-cloning dataset."""

from __future__ import annotations

import glob
import io
import os
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert one uint8 HWC RGB image to normalized CHW float32."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
    return (tensor - _MEAN) / _STD


def _decode_jpg(blob: np.ndarray, start: int, end: int) -> np.ndarray:
    data = bytes(blob[start:end])
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


class ContraVLADataset(Dataset):
    """Loads behavior-cloning shards produced by ``vla.datasets.preprocess``.

    Each sample contains one image, one RAM-derived state vector, and one action
    token. ``images`` keeps a singleton frame dimension for compatibility with
    the VLA model path: [1, 3, 192, 192].
    """

    def __init__(self, shard_dir: str, level_id: int = 0, load_meta: bool = False) -> None:
        self.level_id = level_id
        self.load_meta = load_meta
        self._blobs: list[np.ndarray] = []
        self._offsets: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._metas: list[dict[str, np.ndarray] | None] = []
        self._index: list[tuple[int, int]] = []

        blob_paths = sorted(glob.glob(os.path.join(shard_dir, "*_frames_blob.npy")))
        for blob_path in blob_paths:
            base = blob_path.removesuffix("_frames_blob.npy")
            offsets_path = base + "_frames_offsets.npy"
            states_path = base + "_states.npy"
            actions_path = base + "_actions.npy"
            if not all(os.path.isfile(p) for p in (offsets_path, states_path, actions_path)):
                continue

            shard_idx = len(self._blobs)
            blob = np.load(blob_path, mmap_mode="r")
            offsets = np.load(offsets_path, mmap_mode="r")
            states = np.load(states_path, mmap_mode="r")
            actions = np.load(actions_path, mmap_mode="r")
            n_samples = len(actions)

            if len(states) != n_samples or len(offsets) != n_samples + 1:
                raise ValueError(f"Shard has inconsistent lengths: {base}")

            meta = None
            meta_path = base + "_meta.npz"
            if load_meta and os.path.isfile(meta_path):
                with np.load(meta_path, allow_pickle=True) as data:
                    meta = {key: data[key] for key in data.files}

            self._blobs.append(blob)
            self._offsets.append(offsets)
            self._states.append(states)
            self._actions.append(actions)
            self._metas.append(meta)
            self._index.extend((shard_idx, i) for i in range(n_samples))

        if not self._index:
            raise FileNotFoundError(f"No behavior-cloning shards found in {shard_dir}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        shard_idx, sample_idx = self._index[idx]
        offsets = self._offsets[shard_idx]
        image = _decode_jpg(
            self._blobs[shard_idx],
            int(offsets[sample_idx]),
            int(offsets[sample_idx + 1]),
        )
        action = int(self._actions[shard_idx][sample_idx])

        sample = {
            "images": _preprocess_image(image).unsqueeze(0),
            "state": torch.from_numpy(np.asarray(self._states[shard_idx][sample_idx]).copy()),
            "proprio": torch.from_numpy(np.asarray(self._states[shard_idx][sample_idx]).copy()),
            "actions": torch.tensor([action], dtype=torch.long),
            "level_id": self.level_id,
        }

        meta = self._metas[shard_idx]
        if meta is not None:
            sample["meta"] = {key: value[sample_idx] for key, value in meta.items()}
        return sample


def build_collate_fn(tokenizer, max_text_len: int = 32) -> Callable:
    """Build a collate function that tokenizes fixed level-goal text."""

    def collate_fn(batch: list[dict]) -> dict:
        images = torch.stack([item["images"] for item in batch])
        states = torch.stack([item["state"] for item in batch])
        actions = torch.stack([item["actions"] for item in batch])
        texts = [LEVEL_TEXTS[item["level_id"]] for item in batch]

        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_text_len,
            truncation=True,
        )

        return {
            "images": images,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "state": states,
            "proprio": states,
            "actions": actions,
        }

    return collate_fn
