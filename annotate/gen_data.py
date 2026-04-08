import argparse
import os
import sys

import numpy as np
import torch

from annotate import BCDataSample


_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from pixel2play.model.tokenizer import ConvTokenizer
        _tokenizer = ConvTokenizer(frame_height=192, frame_width=192, embed_dim=1024, n_tokens=1)
        _tokenizer.eval()
    return _tokenizer


def process(npz_path: str, use_text: bool = False) -> None:
    data = np.load(npz_path, allow_pickle=True)
    game = str(data["game"]) if "game" in data else "Contra-Nes"

    # print(f"  Processing {os.path.basename(npz_path)} (game={game})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample = BCDataSample.create(npz_path, game)
    sample.compute_features(npz_path, _get_tokenizer(), use_text=use_text, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ recording(s) to bc_features.npz training format."
    )
    parser.add_argument("source", help="Path to a .npz file or a folder of .npz files")
    parser.add_argument("--use-text", action="store_true",
                        help="Run Gemini annotation and embed text with Gemma")
    args = parser.parse_args()

    src      = args.source.rstrip("/\\")
    use_text = args.use_text

    if os.path.isfile(src) and src.endswith(".npz"):
        process(src, use_text=use_text)
    elif os.path.isdir(src):
        npz_files = sorted(f for f in os.listdir(src) if f.endswith(".npz"))
        if not npz_files:
            sys.exit(f"Error: no .npz files found in {src!r}")
        from tqdm import tqdm
        for npz in tqdm(npz_files, desc="gen_data", unit="rec"):
            process(os.path.join(src, npz), use_text=use_text)
    else:
        sys.exit(f"Error: {src!r} is not a .npz file or a folder")


if __name__ == "__main__":
    main()
