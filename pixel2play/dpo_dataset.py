"""
DPO dataset: loads graph-based DPO pairs and returns chosen/rejected trajectories.

Each sample is a single (chosen, rejected) pair from a search graph.
Actions are converted from 9-bit binary vectors to combined class indices.

Data format (NPZ files produced by annotate/instance_dpo.py):
  chosen_ram        : uint8   (N, T, 2048)
  chosen_actions    : uint8   (N, T, 9)   9-bit NES action vectors
  rejected_ram      : uint8   (N, T, 2048)
  rejected_actions  : uint8   (N, T, 9)
  pivot             : int16   (N,)        shared prefix length within chunk
  chosen_len        : int16   (N,)        real length
  rejected_len      : int16   (N,)        real length
  kind              : uint8   (N,)        0=dead  1=secondary
  good_reward       : float32 (N,)
  bad_reward        : float32 (N,)
  n_pairs           : int32   scalar
  level             : int32   scalar

T = chunk_len = 128 (fixed).
"""

import glob
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from pixel2play.model.nes_actions import encode_combined


# ---------------------------------------------------------------------------
# Action conversion: 9-bit vector -> combined class index
# ---------------------------------------------------------------------------

def _build_action_lookup() -> np.ndarray:
    """Build a lookup table: 9-bit vector (as int 0..511) -> combined action class."""
    lookup = np.zeros(512, dtype=np.int64)
    for dpad_idx, dpad_vec in enumerate(DPAD_TABLE):
        for btn_idx, btn_vec in enumerate(BUTTON_TABLE):
            combined_vec = [int(dpad_vec[i] or btn_vec[i]) for i in range(9)]
            idx = sum(bit << (8 - i) for i, bit in enumerate(combined_vec))
            lookup[idx] = encode_combined(dpad_idx, btn_idx)
    return lookup


_ACTION_LOOKUP = _build_action_lookup()


def _convert_actions(actions_9bit: np.ndarray) -> np.ndarray:
    """Convert (..., 9) uint8 binary vectors to (...) int64 combined class indices."""
    idx = (
        actions_9bit[..., 0].astype(np.int64) << 8
        | actions_9bit[..., 1].astype(np.int64) << 7
        | actions_9bit[..., 2].astype(np.int64) << 6
        | actions_9bit[..., 3].astype(np.int64) << 5
        | actions_9bit[..., 4].astype(np.int64) << 4
        | actions_9bit[..., 5].astype(np.int64) << 3
        | actions_9bit[..., 6].astype(np.int64) << 2
        | actions_9bit[..., 7].astype(np.int64) << 1
        | actions_9bit[..., 8].astype(np.int64)
    )
    return _ACTION_LOOKUP[idx]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    def __init__(self, data_root: str, kind_filter: int | None = None):
        """Args:
            data_root: directory containing DPO .npz files.
            kind_filter: if 0, keep only dead branches; if 1, keep only secondary;
                         if None, keep all.
        """
        self.samples = []  # list of (npz_path, index_within_file)
        npz_paths = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        if not npz_paths:
            raise FileNotFoundError(f"No .npz files found in {data_root!r}")

        for path in npz_paths:
            data = np.load(path)
            n = int(data["n_pairs"])
            if n == 0:
                data.close()
                continue
            kinds = data["kind"]
            for i in range(n):
                if kind_filter is not None and int(kinds[i]) != kind_filter:
                    continue
                self.samples.append((path, i))
            data.close()

        logging.info(f"DPODataset: {len(self.samples)} pairs from {len(npz_paths)} files"
                     f"{' (kind=' + str(kind_filter) + ' only)' if kind_filter is not None else ''}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, i = self.samples[idx]
        data = np.load(path)

        chosen_action = _convert_actions(data["chosen_actions"][i])     # (T,) int64
        rejected_action = _convert_actions(data["rejected_actions"][i]) # (T,) int64

        result = {
            "chosen_ram": torch.from_numpy(data["chosen_ram"][i].astype(np.uint8)),
            "chosen_action": torch.from_numpy(chosen_action.astype(np.int64)),
            "rejected_ram": torch.from_numpy(data["rejected_ram"][i].astype(np.uint8)),
            "rejected_action": torch.from_numpy(rejected_action.astype(np.int64)),
            "pivot": torch.tensor(int(data["pivot"][i]), dtype=torch.int64),
            "chosen_len": torch.tensor(int(data["chosen_len"][i]), dtype=torch.int64),
            "rejected_len": torch.tensor(int(data["rejected_len"][i]), dtype=torch.int64),
        }

        data.close()
        return result
