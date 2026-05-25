"""
Convert legacy _frames.npz shards to mmappable _frames_blob.npy + _frames_offsets.npy.

Usage:
    python -m vla.datasets.migrate_shards vla/data/level1_action2
    python -m vla.datasets.migrate_shards vla/data/level1_action2 --delete-old
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def migrate_dir(shard_dir: str, delete_old: bool) -> int:
    npz_files = sorted(glob.glob(os.path.join(shard_dir, "*_frames.npz")))
    if not npz_files:
        return 0

    for npz_path in npz_files:
        base         = npz_path.replace("_frames.npz", "")
        blob_path    = base + "_frames_blob.npy"
        offsets_path = base + "_frames_offsets.npy"

        if os.path.exists(blob_path) and os.path.exists(offsets_path):
            print(f"  skip (already converted): {os.path.basename(base)}")
            continue

        data = np.load(npz_path)
        np.save(blob_path,    data["blob"])
        np.save(offsets_path, data["offsets"])
        data.close()

        size_mb = (os.path.getsize(blob_path) + os.path.getsize(offsets_path)) / 1e6
        print(f"  {os.path.basename(base)}  →  {size_mb:.1f} MB")

        if delete_old:
            os.remove(npz_path)

    return len(npz_files)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", help="Root data directory (e.g. vla/data/level1_action2)")
    p.add_argument("--delete-old", action="store_true",
                   help="Remove _frames.npz after successful conversion")
    args = p.parse_args()

    total = 0
    # Support both flat dirs and dirs with train/val subdirectories
    subdirs = [os.path.join(args.data_dir, s) for s in ("train", "val")
               if os.path.isdir(os.path.join(args.data_dir, s))]
    if not subdirs:
        subdirs = [args.data_dir]

    for d in subdirs:
        label = os.path.relpath(d, args.data_dir)
        print(f"[{label}]")
        n = migrate_dir(d, args.delete_old)
        total += n
        if n == 0:
            print("  nothing to convert")

    print(f"\nDone — {total} shard(s) converted.")


if __name__ == "__main__":
    main()
