"""
gen_dpo_data.py — Batch-process search graphs into DPO training NPZs and/or BC traces.

Usage:
    python annotate/gen_dpo_data.py synthetic/mc_graph/ --out-dir synthetic/dpo_data
    python annotate/gen_dpo_data.py synthetic/mc_graph/*.npz --out-dir synthetic/dpo_data --workers 4
    python annotate/gen_dpo_data.py synthetic/mc_graph/ --bc-out-dir synthetic/bc_traces --skip-dpo --workers 4
"""

import glob
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


# ── batch processing ───────────────────────────────────────────────────────────

def _collect_graphs(sources: list[str]) -> list[str]:
    paths = []
    for src in sources:
        src = src.rstrip("/\\")
        if os.path.isfile(src):
            paths.append(src)
        elif os.path.isdir(src):
            found = sorted(
                os.path.join(src, f) for f in os.listdir(src)
                if f.endswith((".npz", ".pkl"))
            )
            if not found:
                sys.exit(f"Error: no graph files found in {src!r}")
            paths.extend(found)
        else:
            sys.exit(f"Error: {src!r} is not a file or directory")
    return sorted(set(paths))


def _process_worker(args) -> tuple:
    graph_path, out_dir, bc_raw_dir, skip_dpo = args
    try:
        from annotate.instance_dpo import DPOGraphSample
        out_path = DPOGraphSample(graph_path).process(
            out_dir, bc_out_dir=bc_raw_dir, skip_dpo=skip_dpo
        )
        if skip_dpo:
            return graph_path, 0, 1 if out_path else 0, None
        n_pairs = int(np.load(out_path)["n_pairs"]) if out_path else 0
        return graph_path, n_pairs, 0, None
    except Exception:
        return graph_path, 0, 0, traceback.format_exc()


# ── pack helpers ───────────────────────────────────────────────────────────────

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


def _pack_bc_dir(npz_dir: str, output_dir: str, shard_size: int = 256):
    """Pack individual BC trace .npz files into sharded .npy format."""
    npz_paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_paths:
        return 0

    print(f"Packing {len(npz_paths)} BC traces → {output_dir}")
    shard_idx = 0
    rams, dpads, buttons, names = [], [], [], []

    def _flush():
        nonlocal shard_idx, rams, dpads, buttons, names
        shard_dir = os.path.join(output_dir, f"shard_{shard_idx:04d}")
        _write_shard(shard_dir, rams, dpads, buttons, names)
        shard_idx += 1
        rams, dpads, buttons, names = [], [], [], []

    for npz_path in tqdm(npz_paths, desc="pack", unit="rec"):
        data = np.load(npz_path)
        rams.append(data["ram"])
        dpads.append(data["dpad"])
        buttons.append(data["button"])
        names.append(os.path.basename(npz_path))
        data.close()
        if len(rams) == shard_size:
            _flush()

    if rams:
        _flush()

    print(f"Done. {shard_idx} shards → load with NESDataset('{output_dir}')")
    return shard_idx


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-convert search graphs into DPO training NPZs and/or BC traces"
    )
    parser.add_argument("source", nargs="+", help="Graph .npz/.pkl files or directories")
    parser.add_argument("--out-dir", default=None, help="Output directory for DPO NPZs")
    parser.add_argument("--bc-out-dir", default=None, help="Output directory for full BC trace shards")
    parser.add_argument("--bc-shard-size", type=int, default=256, help="Recordings per BC shard (default: 256)")
    parser.add_argument("--skip-dpo", action="store_true",
                        help="Skip DPO pair generation; export BC traces only")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (default: 16)")
    args = parser.parse_args()

    if args.out_dir is None and args.bc_out_dir is None:
        sys.exit("Error: specify at least one of --out-dir or --bc-out-dir")
    if args.skip_dpo and args.out_dir is not None:
        sys.exit("Error: --skip-dpo is incompatible with --out-dir")

    graph_paths = _collect_graphs(args.source)
    if not graph_paths:
        sys.exit("No graph files found.")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # Workers write raw .npz files to a temp dir; main process packs them into shards.
    bc_raw_dir = None
    if args.bc_out_dir:
        os.makedirs(args.bc_out_dir, exist_ok=True)
        bc_raw_dir = os.path.join(args.bc_out_dir, ".raw")
        os.makedirs(bc_raw_dir, exist_ok=True)

    out_label = args.out_dir or args.bc_out_dir
    print(f"Processing {len(graph_paths)} graphs → {out_label}")

    tasks       = [(p, args.out_dir, bc_raw_dir, args.skip_dpo) for p in graph_paths]
    total_pairs = 0
    total_bc    = 0

    ctx = mp.get_context("spawn")
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_process_worker, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing graphs"):
            try:
                _, n_pairs, n_bc, err = fut.result()
            except Exception:
                errors += 1
                continue
            if err:
                errors += 1
            else:
                total_pairs += n_pairs
                total_bc    += n_bc

    # ── pack BC traces into shards ───────────────────────────────────────────
    if bc_raw_dir:
        n_shards = _pack_bc_dir(bc_raw_dir, args.bc_out_dir, shard_size=args.bc_shard_size)
        if n_shards > 0:
            shutil.rmtree(bc_raw_dir)

    parts = []
    if total_pairs:
        parts.append(f"{total_pairs} DPO pairs")
    if total_bc:
        parts.append(f"{total_bc} BC traces")
    summary = ", ".join(parts) if parts else "nothing"
    if errors:
        print(f"Done. {summary} ({errors} errors)")
    else:
        print(f"Done. {summary}")


if __name__ == "__main__":
    main()
