import argparse
import os
import sys

import numpy as np
import torch

from annotate import BCDataSample


def _worker(args: tuple) -> tuple[str, str | None]:
    """Process one recording in a subprocess. Returns (npz_path, error_msg or None)."""
    npz_path, use_text, chunk_size, device = args

    from pixel2play.model.tokenizer import ConvTokenizer
    tokenizer = ConvTokenizer(frame_height=192, frame_width=192, embed_dim=1024, n_tokens=1)
    tokenizer.eval()

    data   = np.load(npz_path, allow_pickle=True)
    game   = str(data["game"]) if "game" in data else "Contra-Nes"
    sample = BCDataSample.create(npz_path, game)

    if sample.has_features:
        return npz_path, None

    try:
        sample.compute_features(npz_path, tokenizer, use_text=use_text,
                                chunk_size=chunk_size, device=device)
        return npz_path, None
    except AssertionError as e:
        return npz_path, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ recording(s) to bc_features.npz training format."
    )
    parser.add_argument("source", help="Path to a .npz file or a folder of .npz files")
    parser.add_argument("--use-text", action="store_true",
                        help="Run Gemini annotation and embed text with Gemma")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="Frames per encoder pass per worker (default: 32)")
    args = parser.parse_args()

    src      = args.source.rstrip("/\\")
    use_text = args.use_text
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isfile(src) and src.endswith(".npz"):
        _worker((src, use_text, args.chunk_size, device))

    elif os.path.isdir(src):
        npz_files = sorted(f for f in os.listdir(src) if f.endswith(".npz"))
        if not npz_files:
            sys.exit(f"Error: no .npz files found in {src!r}")

        from tqdm import tqdm

        worker_args = [
            (os.path.join(src, f), use_text, args.chunk_size, device)
            for f in npz_files
        ]

        if args.workers == 1:
            for wa in tqdm(worker_args, desc="gen_data", unit="rec"):
                npz_path, err = _worker(wa)
                if err:
                    tqdm.write(f"  Deleting bad trace: {os.path.basename(npz_path)} ({err})")
                    os.remove(npz_path)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers) as pool:
                for npz_path, err in tqdm(
                    pool.imap_unordered(_worker, worker_args),
                    total=len(worker_args),
                    desc="gen_data",
                    unit="rec",
                ):
                    if err:
                        tqdm.write(f"  Deleting bad trace: {os.path.basename(npz_path)} ({err})")
                        os.remove(npz_path)
    else:
        sys.exit(f"Error: {src!r} is not a .npz file or a folder")


if __name__ == "__main__":
    main()
