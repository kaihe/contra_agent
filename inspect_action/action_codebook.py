"""
Test Codebook Reconstruction
==============================

Take the winning human traces, encode each 16-frame window to its nearest
codebook entry, then replay the quantized actions through the emulator.

If wins are preserved, the codebook is a lossless-enough action space
for PPO to learn from.

Usage:
    python test_codebook.py
"""

import os
import sys
import glob
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
from contra_wrapper import Monitor

# =========================================================================
# CONFIG
# =========================================================================
WINDOW = 4
CODEBOOK_SIZE = 32
TRACE_DIR = os.path.join(os.path.dirname(__file__), "human_trace")
GIFS_DIR = os.path.join(os.path.dirname(__file__), "gifs")
GAME = "Contra-Nes"

BUTTON_NAMES_SHORT = ["B", "U", "D", "L", "R", "A"]
# Useful column indices in the full 9-button array
USEFUL_COLS = [0, 4, 5, 6, 7, 8]  # B, U, D, L, R, A


def load_all_chunks(trace_dir, window):
    """Load all traces and chunk into (window, 6) patches for codebook learning."""
    npz_files = sorted(glob.glob(os.path.join(trace_dir, "*.npz")))
    all_chunks = []
    for f in npz_files:
        data = np.load(f)
        # Skip the first dummy zero-action representation of env.reset() state
        actions = data["actions"][1:, USEFUL_COLS] if len(data.get("actions", [])) > 1 else np.array([])
        n_chunks = len(actions) // window
        for i in range(n_chunks):
            all_chunks.append(actions[i * window : (i + 1) * window])
    return np.array(all_chunks, dtype=np.float32)


def learn_codebook(chunks, k):
    """K-means on flattened chunks, return binarized codebook."""
    M, W, B = chunks.shape
    codebook_dir = os.path.join(os.path.dirname(__file__), "codebooks")
    os.makedirs(codebook_dir, exist_ok=True)
    file_path = os.path.join(codebook_dir, f"codebook_w{W}_S{k}.npz")
    
    if os.path.exists(file_path):
        print(f"Loading cached codebook from {file_path} ...")
        data = np.load(file_path)
        return data["codebook_bin"], data["centers"], data["labels"]

    print(f"\nRunning k-means: {M} samples, k={k} ...")
    flat = chunks.reshape(M, W * B)
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    km.fit(flat)
    
    codebook = km.cluster_centers_.reshape(k, W, B)
    codebook_bin = (codebook > 0.5).astype(np.int8)
    
    print(f"Saving newly learned codebook to {file_path} ...")
    np.savez(file_path, codebook_bin=codebook_bin, centers=km.cluster_centers_, labels=km.labels_)
    
    return codebook_bin, km.cluster_centers_, km.labels_


def encode_trace(actions_useful, codebook, centers):
    """Encode a full trace into codebook indices, return quantized actions."""
    W = codebook.shape[1]
    B = codebook.shape[2]
    n_chunks = len(actions_useful) // W
    
    quantized = []
    code_indices = []
    
    for i in range(n_chunks):
        chunk = actions_useful[i * W : (i + 1) * W].reshape(1, -1).astype(np.float32)
        # Find closest center using euclidean distance squared
        idx = np.argmin(np.sum((centers - chunk) ** 2, axis=1))
        code_indices.append(idx)
        quantized.append(codebook[idx])  # (W, 6)
    
    # Handle leftover frames (repeat last code or use original)
    leftover = len(actions_useful) - n_chunks * W
    if leftover > 0:
        # Just use the original for the tail
        quantized.append(actions_useful[n_chunks * W:])
    
    quantized = np.concatenate(quantized, axis=0)  # (N, 6)
    return quantized, code_indices


def useful_to_full(actions_useful):
    """Convert (N, 6) useful-only actions back to (N, 9) full NES actions."""
    N = len(actions_useful)
    full = np.zeros((N, 9), dtype=np.int8)
    for i, col in enumerate(USEFUL_COLS):
        full[:, col] = actions_useful[:, i]
    return full


def display_codebook(codebook, labels):
    """Pretty-print each code with usage frequency."""
    K, W, B = codebook.shape
    btn_names = ["B", "U", "D", "L", "R", "A"]  # matches column order

    # Count usage
    counts = np.bincount(labels, minlength=K)
    total = len(labels)

    # Sort by frequency (most used first)
    order = np.argsort(-counts)

    print(f"\n{'='*70}")
    print(f" LEARNED CODEBOOK: {K} codes × {W} frames")
    print(f"{'='*70}")

    for rank, idx in enumerate(order):
        freq = counts[idx] / total * 100
        code = codebook[idx]  # (W, 6)

        # Build a compact visual per frame
        print(f"\n  Code {rank:2d} (cluster {idx:2d}) — {freq:5.1f}% usage ({counts[idx]} chunks)")
        print(f"  {'Frame':<6}", end="")
        for b in btn_names:
            print(f" {b:>3}", end="")
        print("   Summary")
        print(f"  {'-'*42}")

        for t in range(W):
            frame_btns = code[t]
            active = [btn_names[j] for j in range(6) if frame_btns[j] == 1]
            summary = "+".join(active) if active else "---"
            print(f"  {t:5d} ", end="")
            for b in frame_btns:
                print(f"  {'█' if b else '·'}", end="")
            print(f"   {summary}")

    # Also print a compact action table for easy copy-paste
    print(f"\n{'='*70}")
    print(f" COMPACT ACTION TABLE (sorted by usage)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Usage':>6} {'Pattern Summary'}")
    print(f"{'-'*50}")
    for rank, idx in enumerate(order):
        freq = counts[idx] / total * 100
        code = codebook[idx]

        # Summarize: find the dominant pattern across the frames
        # Group consecutive identical frames
        segments = []
        t = 0
        while t < W:
            frame = tuple(code[t])
            length = 1
            while t + length < W and tuple(code[t + length]) == frame:
                length += 1
            active = [btn_names[j] for j in range(6) if code[t][j] == 1]
            name = "+".join(active) if active else "NOOP"
            segments.append(f"{name}×{length}")
            t += length
        pattern = " → ".join(segments)
        print(f"  {rank:<3}  {freq:5.1f}%  {pattern}")


def replay_and_check(env, actions_full, label, gif_path=None):
    """Replay actions through the emulator from Level1, report outcome."""
    env.reset()
    
    monitor = None
    if gif_path:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        monitor = Monitor(240, 224, saved_path=gif_path)
        monitor.record(env.get_screen())
    
    max_xscroll = 0
    max_score = 0
    last_info = {}
    for i, action in enumerate(actions_full):
        obs, _, term, trunc, info = env.step(list(action))
        last_info = info
        max_xscroll = max(max_xscroll, info.get("xscroll", 0))
        max_score = max(max_score, info.get("score", 0))
        if monitor:
            monitor.record(obs)
            
        # Early stop if we run out of lives mapped to -1 (uint8 255)
        if info.get("lives", 0) == 255 or info.get("lives", 0) < 0:
            break
    
    if monitor:
        monitor.close()
        # size_kb = os.path.getsize(gif_path) / 1024
        # print(f"  GIF saved: {gif_path} ({size_kb:.1f} KB)")
    
    lives = last_info.get("lives", 0)
    
    # xscroll wraps to 0 on level completion, so use max
    won = max_xscroll >= 3072 and max_score >= 150
    
    # print(f"  {label}: max_xscroll={max_xscroll}, max_score={max_score}, lives={lives}, "
    #       f"frames={i+1}, {'WIN ✓' if won else 'LOSE ✗'}")
    
    return won, max_xscroll, max_score, lives


def main():
    parser = argparse.ArgumentParser(description="Learn VQ Action Codebook and Test Reconstructions")
    parser.add_argument("--save_gif", action="store_true", help="Save GIFs of the replays (can be slow and use lots of space)")
    parser.add_argument("--display", action="store_true", help="Display the learned codebook pattern distribution")
    parser.add_argument("--skip_test", action="store_true", help="Skip replaying the traces, only learn the codebook")
    args = parser.parse_args()

    # 1. Learn codebook from ALL traces
    print("Learning codebook from all human traces...")
    all_chunks = load_all_chunks(TRACE_DIR, WINDOW)
    codebook, centers, labels = learn_codebook(all_chunks, CODEBOOK_SIZE)
    print(f"Codebook: {CODEBOOK_SIZE} codes × {WINDOW} frames\n")
    
    if args.display:
        display_codebook(codebook, labels)
        
    if args.skip_test:
        return
        
    # 2. Find all traces
    trace_files = sorted(glob.glob(os.path.join(TRACE_DIR, "*.npz")))
    if not trace_files:
        print("No traces found!")
        return
    
    print(f"Found {len(trace_files)} traces to test.\n")
    print("=" * 60)
    
    total_traces = len(trace_files)
    orig_wins = 0
    quant_wins = 0
    orig_xscrolls = []
    quant_xscrolls = []
    orig_scores = []
    quant_scores = []
    accuracies = []
    
    # Create the environment once instead of re-creating per trace
    env = retro.make(
        game=GAME, state="Level1",
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    
    for f in tqdm(trace_files, desc="Evaluating Traces"):
        basename = os.path.basename(f)[:-4]
        data = np.load(f)
        # Skip the first dummy zero-action
        original_actions = data["actions"][1:] if len(data.get("actions", [])) > 1 else np.array([])
        original_useful = original_actions[:, USEFUL_COLS].astype(np.int8) if len(original_actions) > 0 else np.array([])
        
        # Test original first
        # print(f"\n--- {basename} ({len(original_actions)} frames) ---")
        # print("  [Original]")
        orig_gif = os.path.join(GIFS_DIR, f"codebook_orig_{basename}.gif") if args.save_gif else None
        orig_won, orig_xsc, orig_sc, _ = replay_and_check(env, original_actions, "Original", gif_path=orig_gif)
        if orig_won: orig_wins += 1
        orig_xscrolls.append(orig_xsc)
        orig_scores.append(orig_sc)
        
        # Encode and test quantized
        quantized_useful, code_indices = encode_trace(original_useful, codebook, centers)
        quantized_full = useful_to_full(quantized_useful)
        
        # Count bit differences
        min_len = min(len(original_actions), len(quantized_full))
        orig_useful_trimmed = original_actions[:min_len, :][:, USEFUL_COLS]
        quant_useful_trimmed = quantized_useful[:min_len]
        bit_errors = np.sum(orig_useful_trimmed != quant_useful_trimmed)
        total_bits = min_len * 6
        accuracy = (total_bits - bit_errors) / total_bits * 100
        accuracies.append(accuracy)
        # print(f"  Reconstruction: {accuracy:.1f}% ({bit_errors} bit errors / {total_bits} bits)")
        
        # print("  [Quantized]")
        quant_gif = os.path.join(GIFS_DIR, f"codebook_quant_{basename}.gif") if args.save_gif else None
        quant_won, quant_xsc, quant_sc, _ = replay_and_check(env, quantized_full, "Quantized", gif_path=quant_gif)
        if quant_won: quant_wins += 1
        quant_xscrolls.append(quant_xsc)
        quant_scores.append(quant_sc)
        
    
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Total Traces Evaluated: {total_traces}")
    print(f"Mean Reconstruction Acc : {np.mean(accuracies):.2f}%")
    print(f"Original Wins  : {orig_wins}/{total_traces} "
          f"(Avg xscroll: {np.mean(orig_xscrolls):.1f}, Avg score: {np.mean(orig_scores):.1f})")
    print(f"Quantized Wins : {quant_wins}/{total_traces} "
          f"(Avg xscroll: {np.mean(quant_xscrolls):.1f}, Avg score: {np.mean(quant_scores):.1f})")
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
