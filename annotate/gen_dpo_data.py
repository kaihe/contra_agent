"""
gen_dpo_data.py — Batch-process search graphs into DPO training NPZs.

Usage:
    python annotate/gen_dpo_data.py synthetic/mc_graph/ --out-dir synthetic/dpo_data
    python annotate/gen_dpo_data.py synthetic/mc_graph/*.npz --out-dir synthetic/dpo_data --workers 4
"""

import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


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
    graph_path, out_dir = args
    try:
        from annotate.instance_dpo import DPOGraphSample
        out_path = DPOGraphSample(graph_path).process(out_dir)
        n_pairs  = int(np.load(out_path)["n_pairs"]) if out_path else 0
        return graph_path, n_pairs, None
    except Exception:
        return graph_path, 0, traceback.format_exc()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-convert search graphs into DPO training NPZs"
    )
    parser.add_argument("source", nargs="+", help="Graph .npz/.pkl files or directories")
    parser.add_argument("--out-dir", required=True, help="Output directory for training NPZs")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (default: 16)")
    args = parser.parse_args()

    graph_paths = _collect_graphs(args.source)
    if not graph_paths:
        sys.exit("No graph files found.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Processing {len(graph_paths)} graphs → {args.out_dir}")

    tasks       = [(p, args.out_dir) for p in graph_paths]
    total_pairs = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_process_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            graph_path = futures[fut]
            name = os.path.basename(graph_path)
            try:
                _, n_pairs, err = fut.result()
            except Exception:
                print(f"  [error] {name}:\n{traceback.format_exc()}")
                continue
            if err:
                print(f"  [error] {name}:\n{err.splitlines()[-1]}")
            else:
                total_pairs += n_pairs
                print(f"  {name}: {n_pairs} pairs")

    print(f"Done. {total_pairs} total DPO pairs in {args.out_dir!r}")


if __name__ == "__main__":
    main()
