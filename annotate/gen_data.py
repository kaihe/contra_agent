import argparse
import glob
import os
import shutil
import sys

import numpy as np

from annotate.instance import BCDataSample


def _worker(npz_path: str) -> tuple:
    """Replay one mc_trace in a subprocess. Returns (npz_path, ram, dpad, button, err)."""
    try:
        ram, dpad, button = BCDataSample(npz_path).replay_arrays(npz_path)
        return npz_path, ram, dpad, button, None
    except Exception as e:
        import traceback
        return npz_path, None, None, None, traceback.format_exc()


def _collect_npz(sources: list[str]) -> list[str]:
    npz_paths = []
    for src in sources:
        src = src.rstrip("/\\")
        if os.path.isfile(src) and src.endswith(".npz"):
            npz_paths.append(src)
        elif os.path.isdir(src):
            found = sorted(
                os.path.join(src, f) for f in os.listdir(src) if f.endswith(".npz")
            )
            if not found:
                sys.exit(f"Error: no .npz files found in {src!r}")
            npz_paths.extend(found)
        else:
            found = sorted(p for p in glob.glob(src) if p.endswith(".npz"))
            if not found:
                sys.exit(f"Error: {src!r} is not a .npz file, a folder, or a valid glob pattern")
            npz_paths.extend(found)
    return sorted(set(npz_paths))


def _write_shard(shard_dir: str, rams, dpads, buttons, names):
    os.makedirs(shard_dir, exist_ok=True)
    index = np.array(
        [(sum(len(d) for d in dpads[:i]), len(dpads[i])) for i in range(len(dpads))],
        dtype=np.int64,
    )
    np.save(os.path.join(shard_dir, "ram.npy"),    np.concatenate(rams,    axis=0))
    np.save(os.path.join(shard_dir, "dpad.npy"),   np.concatenate(dpads,   axis=0))
    np.save(os.path.join(shard_dir, "button.npy"), np.concatenate(buttons, axis=0))
    np.save(os.path.join(shard_dir, "index.npy"),  index)
    np.save(os.path.join(shard_dir, "names.npy"),  np.array(names, dtype=object))


def main():
    parser = argparse.ArgumentParser(
        description="Convert mc_traces to mmap-able shards for fast DataLoader access"
    )
    parser.add_argument("source", nargs="+",
                        help="mc_trace .npz files, directories, or globs")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write shard_NNNN/ subdirectories")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for emulator replay (default: 8)")
    parser.add_argument("--shard-size", type=int, default=256,
                        help="Number of recordings per shard (default: 256)")
    args = parser.parse_args()

    from tqdm import tqdm

    npz_paths = _collect_npz(args.source)
    if not npz_paths:
        sys.exit("No .npz files found.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Replaying {len(npz_paths)} mc_traces → {args.output_dir}")

    shard_idx = 0
    rams, dpads, buttons, names = [], [], [], []

    def _flush():
        nonlocal shard_idx, rams, dpads, buttons, names
        shard_dir = os.path.join(args.output_dir, f"shard_{shard_idx:04d}")
        _write_shard(shard_dir, rams, dpads, buttons, names)
        shard_idx += 1
        rams, dpads, buttons, names = [], [], [], []

    bad_prune_dir = "synthetic/mc_trace_bad_prune"

    def _handle(npz_path, ram, dpad, button, err):
        if err:
            tqdm.write(f"[skip] {os.path.basename(npz_path)}: {err.splitlines()[-1]}")
            os.makedirs(bad_prune_dir, exist_ok=True)
            shutil.move(npz_path, os.path.join(bad_prune_dir, os.path.basename(npz_path)))
            return
        rams.append(ram)
        dpads.append(dpad)
        buttons.append(button)
        names.append(os.path.basename(npz_path))
        if len(rams) == args.shard_size:
            _flush()

    if args.workers == 1:
        for npz_path in tqdm(npz_paths, desc="pack", unit="rec"):
            _handle(*_worker(npz_path))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker, npz_paths),
                total=len(npz_paths),
                desc="pack",
                unit="rec",
            ):
                _handle(*result)

    if rams:
        _flush()

    print(f"Done. {shard_idx} shards → load with NESDataset('{args.output_dir}', sharded=True)")


if __name__ == "__main__":
    main()
