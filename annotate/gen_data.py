import argparse
import glob
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from annotate.instance import BCDataSample, collect_dpo_pairs


# ── Shared helpers ─────────────────────────────────────────────────────────────

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


# ── pack subcommand ────────────────────────────────────────────────────────────

def _pack_worker(npz_path: str) -> tuple:
    """Replay one mc_trace in a subprocess. Returns (npz_path, ram, dpad, button, err)."""
    try:
        ram, dpad, button = BCDataSample(npz_path).replay_game(npz_path)
        return npz_path, ram, dpad, button, None
    except Exception:
        return npz_path, None, None, None, traceback.format_exc()


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


def cmd_pack(args):
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
            _handle(*_pack_worker(npz_path))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_pack_worker, npz_paths),
                total=len(npz_paths),
                desc="pack",
                unit="rec",
            ):
                _handle(*result)

    if rams:
        _flush()

    print(f"Done. {shard_idx} shards → load with NESDataset('{args.output_dir}')")


# ── dpo subcommand ─────────────────────────────────────────────────────────────

def _dpo_trace_worker(args):
    npz_path, out_path, n_pivots, n_rollouts, seed = args
    result = collect_dpo_pairs(npz_path, n_pivots=n_pivots, n_rollouts=n_rollouts, seed=seed)
    if result is not None:
        np.savez_compressed(out_path, **result)
    return npz_path, result


def cmd_dpo(args):
    npz_paths = _collect_npz(args.source)
    if not npz_paths:
        sys.exit("No .npz files found.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating DPO pairs from {len(npz_paths)} traces → {args.output_dir}")

    tasks = []
    for npz_path in npz_paths:
        name     = os.path.splitext(os.path.basename(npz_path))[0]
        out_path = os.path.join(args.output_dir, f"{name}.npz")
        if os.path.exists(out_path):
            tqdm.write(f"  {name}: already exists, skipping")
            continue
        tasks.append((npz_path, out_path, args.n_pivots, args.n_rollouts, args.seed))

    total_pairs = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_dpo_trace_worker, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="dpo", unit="trace"):
            npz_path = futures[fut]
            name = os.path.splitext(os.path.basename(npz_path))[0]
            try:
                _, result = fut.result()
            except Exception:
                tqdm.write(f"  [error] {name}:\n{traceback.format_exc()}")
                continue
            if result is None:
                tqdm.write(f"  {name}: no valid pairs found, skipping")
                continue
            n = len(result["pivot"])
            total_pairs += n
            tqdm.write(f"  {name}: {n} pairs")

    print(f"Done. {total_pairs} total DPO pairs in {args.output_dir!r}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate BC shards or DPO pairs from Contra mc_trace files"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── pack ──────────────────────────────────────────────────────────────────
    p_pack = sub.add_parser("pack", help="Convert mc_traces to mmap-able BC shards")
    p_pack.add_argument("source", nargs="+",
                        help="mc_trace .npz files, directories, or globs")
    p_pack.add_argument("--output-dir", required=True,
                        help="Directory to write shard_NNNN/ subdirectories")
    p_pack.add_argument("--workers",    type=int, default=8,
                        help="Parallel workers for emulator replay (default: 8)")
    p_pack.add_argument("--shard-size", type=int, default=256,
                        help="Number of recordings per shard (default: 256)")

    # ── dpo ───────────────────────────────────────────────────────────────────
    p_dpo = sub.add_parser("dpo", help="Generate DPO preference pairs from winning traces")
    p_dpo.add_argument("source", nargs="+",
                       help="mc_trace .npz files, directories, or globs")
    p_dpo.add_argument("--output-dir", required=True,
                       help="Directory to write one NPZ per trace")
    p_dpo.add_argument("--workers",    type=int, default=4,
                       help="Parallel trace workers (default: 4)")
    p_dpo.add_argument("--n-pivots",   type=int, default=10,
                       help="Pivot points sampled per 128-step chunk (default: 10)")
    p_dpo.add_argument("--n-rollouts", type=int, default=64,
                       help="Random rollouts per pivot (default: 64)")
    p_dpo.add_argument("--seed",       type=int, default=0)

    args = parser.parse_args()
    if args.command == "pack":
        cmd_pack(args)
    else:
        cmd_dpo(args)


if __name__ == "__main__":
    main()
