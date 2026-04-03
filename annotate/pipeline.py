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


def process(npz_path: str, annotate: bool = True, img_features: bool = True) -> None:
    data = np.load(npz_path, allow_pickle=True)
    game = str(data["game"]) if "game" in data else "Contra-Nes"

    print(f"  Processing {os.path.basename(npz_path)} (game={game})...")
    sample = BCDataSample.create(npz_path, game, annotate=annotate)

    if img_features:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sample.precompute_image_features(_get_tokenizer(), device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ recording(s) to final_data training format."
    )
    parser.add_argument("source", help="Path to a .npz file or a folder of .npz files")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip Gemini text annotation; write actions-only proto")
    parser.add_argument("--no-img-features", action="store_true",
                        help="Skip EfficientNet feature precomputation")
    args = parser.parse_args()

    src = args.source.rstrip("/\\")
    annotate = not args.no_annotate

    if os.path.isfile(src) and src.endswith(".npz"):
        process(src, annotate=annotate, img_features=not args.no_img_features)
    elif os.path.isdir(src):
        npz_files = sorted(f for f in os.listdir(src) if f.endswith(".npz"))
        if not npz_files:
            sys.exit(f"Error: no .npz files found in {src!r}")
        print(f"Found {len(npz_files)} NPZ file(s) in {src}\n")
        for npz in npz_files:
            process(os.path.join(src, npz), annotate=annotate, img_features=not args.no_img_features)
    else:
        sys.exit(f"Error: {src!r} is not a .npz file or a folder of .npz files")


if __name__ == "__main__":
    main()
