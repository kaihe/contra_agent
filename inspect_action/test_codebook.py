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
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import stable_retro as retro
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))
from contra_wrapper import Monitor

# =========================================================================
# CONFIG
# =========================================================================
WINDOW = 16
CODEBOOK_SIZE = 16
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
    flat = chunks.reshape(M, W * B)
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    km.fit(flat)
    codebook = km.cluster_centers_.reshape(k, W, B)
    codebook_bin = (codebook > 0.5).astype(np.int8)
    return codebook_bin, km


def encode_trace(actions_useful, codebook, km):
    """Encode a full trace into codebook indices, return quantized actions."""
    W = codebook.shape[1]
    B = codebook.shape[2]
    n_chunks = len(actions_useful) // W
    
    quantized = []
    code_indices = []
    
    for i in range(n_chunks):
        chunk = actions_useful[i * W : (i + 1) * W].reshape(1, -1).astype(np.float32)
        idx = km.predict(chunk)[0]
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


def replay_and_check(actions_full, label, gif_path=None):
    """Replay actions through the emulator from Level1, report outcome."""
    env = retro.make(
        game=GAME, state="Level1",
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
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
    
    if monitor:
        monitor.close()
        size_kb = os.path.getsize(gif_path) / 1024
        print(f"  GIF saved: {gif_path} ({size_kb:.1f} KB)")
    
    lives = last_info.get("lives", 0)
    
    # xscroll wraps to 0 on level completion, so use max
    won = max_xscroll >= 3072 and max_score >= 150
    
    print(f"  {label}: max_xscroll={max_xscroll}, max_score={max_score}, lives={lives}, "
          f"frames={i+1}, {'WIN ✓' if won else 'LOSE ✗'}")
    
    env.close()
    return won, max_xscroll, max_score, lives


def main():
    parser = argparse.ArgumentParser(description="Test Codebook Reconstruction on Winning Traces")
    parser.add_argument("--save_gif", action="store_true", help="Save GIFs of the replays (can be slow and use lots of space)")
    args = parser.parse_args()

    # 1. Learn codebook from ALL traces
    print("Learning codebook from all human traces...")
    all_chunks = load_all_chunks(TRACE_DIR, WINDOW)
    codebook, km = learn_codebook(all_chunks, CODEBOOK_SIZE)
    print(f"Codebook: {CODEBOOK_SIZE} codes × {WINDOW} frames\n")
    
    # 2. Find winning traces
    win_files = sorted(glob.glob(os.path.join(TRACE_DIR, "win_*.npz")))
    if not win_files:
        print("No winning traces found!")
        return
    
    print(f"Found {len(win_files)} winning traces to test.\n")
    print("=" * 60)
    
    total_traces = len(win_files)
    orig_wins = 0
    quant_wins = 0
    
    for f in win_files:
        basename = os.path.basename(f)[:-4]
        data = np.load(f)
        # Skip the first dummy zero-action
        original_actions = data["actions"][1:] if len(data.get("actions", [])) > 1 else np.array([])
        original_useful = original_actions[:, USEFUL_COLS].astype(np.int8) if len(original_actions) > 0 else np.array([])
        
        print(f"\n--- {basename} ({len(original_actions)} frames) ---")
        
        # Test original first
        print("  [Original]")
        orig_gif = os.path.join(GIFS_DIR, f"codebook_orig_{basename}.gif") if args.save_gif else None
        orig_won, _, _, _ = replay_and_check(original_actions, "Original", gif_path=orig_gif)
        if orig_won: orig_wins += 1
        
        # Encode and test quantized
        quantized_useful, code_indices = encode_trace(original_useful, codebook, km)
        quantized_full = useful_to_full(quantized_useful)
        
        # Count bit differences
        min_len = min(len(original_actions), len(quantized_full))
        orig_useful_trimmed = original_actions[:min_len, :][:, USEFUL_COLS]
        quant_useful_trimmed = quantized_useful[:min_len]
        bit_errors = np.sum(orig_useful_trimmed != quant_useful_trimmed)
        total_bits = min_len * 6
        accuracy = (total_bits - bit_errors) / total_bits * 100
        print(f"  Reconstruction: {accuracy:.1f}% ({bit_errors} bit errors / {total_bits} bits)")
        
        print("  [Quantized]")
        quant_gif = os.path.join(GIFS_DIR, f"codebook_quant_{basename}.gif") if args.save_gif else None
        quant_won, _, _, _ = replay_and_check(quantized_full, "Quantized", gif_path=quant_gif)
        if quant_won: quant_wins += 1
        
        status = "✓ BOTH WIN" if (orig_won and quant_won) else \
                 "✗ QUANTIZED LOST" if (orig_won and not quant_won) else \
                 "? ORIGINAL ALSO LOST" if not orig_won else "? UNEXPECTED"
        print(f"  Result: {status}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Total Traces Evaluated: {total_traces}")
    print(f"Original Wins: {orig_wins}/{total_traces}")
    print(f"Quantized Wins: {quant_wins}/{total_traces}")
    print("Done!")


if __name__ == "__main__":
    main()
