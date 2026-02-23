"""
Learn Action Codebook (VQ-style)
=================================

Treat human gameplay as a stream of 9-button NES inputs at 60 FPS.
Chunk into windows of W frames, then use k-means to learn K prototype
action sequences ("codes") that best reconstruct the human input.

Similar to the codebook learning step in VQ-VAE / VQ-GAN, but applied
directly to discrete controller inputs instead of image patches.

Usage:
    python learn_codebook.py
"""

import os
import glob
import numpy as np
from sklearn.cluster import KMeans

# =========================================================================
# CONFIG
# =========================================================================
WINDOW = 16        # frames per code
CODEBOOK_SIZE = 16 # number of discrete actions (codes)
TRACE_DIR = os.path.join(os.path.dirname(__file__), "human_trace")

BUTTON_NAMES = ["B", "NULL", "SEL", "STA", "U", "D", "L", "R", "A"]

# =========================================================================
# 1. LOAD & CHUNK
# =========================================================================
def load_chunks(trace_dir, window):
    """Load all human traces, chunk into (window, 9) patches."""
    npz_files = sorted(glob.glob(os.path.join(trace_dir, "*.npz")))
    print(f"Loading {len(npz_files)} trace files...")

    all_chunks = []
    for f in npz_files:
        # Skip the first dummy zero-action representation of env.reset() state
        actions = data["actions"][1:] if len(data.get("actions", [])) > 1 else np.array([])  # (N, 9)
        # Drop NULL, SELECT, START columns (indices 1, 2, 3) — always zero
        # Keep: B(0), U(4), D(5), L(6), R(7), A(8)
        actions_useful = actions[:, [0, 4, 5, 6, 7, 8]]

        n_chunks = len(actions_useful) // window
        for i in range(n_chunks):
            chunk = actions_useful[i * window : (i + 1) * window]  # (W, 6)
            all_chunks.append(chunk)

    all_chunks = np.array(all_chunks, dtype=np.float32)  # (M, W, 6)
    print(f"Total chunks: {all_chunks.shape[0]}  (window={window}, buttons=6)")
    return all_chunks


# =========================================================================
# 2. LEARN CODEBOOK
# =========================================================================
def learn_codebook(chunks, k):
    """Flatten each chunk to a vector and run k-means."""
    M, W, B = chunks.shape
    flat = chunks.reshape(M, W * B)  # (M, W*6)

    print(f"\nRunning k-means: {M} samples, {W*B}-dim vectors, k={k} ...")
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    km.fit(flat)

    # Codebook centroids: (k, W*6) -> (k, W, 6)
    codebook = km.cluster_centers_.reshape(k, W, B)
    labels = km.labels_
    inertia = km.inertia_

    # Binarize the codebook (round to 0/1 since inputs are binary)
    codebook_bin = (codebook > 0.5).astype(np.int8)

    return codebook_bin, labels, inertia


# =========================================================================
# 3. EVALUATE RECONSTRUCTION
# =========================================================================
def evaluate(chunks, codebook, labels):
    """Measure how well the binary codebook reconstructs the original."""
    M, W, B = chunks.shape
    total_bits = M * W * B
    errors = 0
    for i in range(M):
        code_idx = labels[i]
        errors += np.sum(chunks[i] != codebook[code_idx])

    accuracy = (total_bits - errors) / total_bits * 100
    print(f"\nReconstruction accuracy: {accuracy:.2f}% ({errors} bit errors / {total_bits} total bits)")
    return accuracy


# =========================================================================
# 4. DISPLAY CODEBOOK
# =========================================================================
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

        # Summarize: find the dominant pattern across the 16 frames
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


if __name__ == "__main__":
    chunks = load_chunks(TRACE_DIR, WINDOW)
    codebook, labels, inertia = learn_codebook(chunks, CODEBOOK_SIZE)
    evaluate(chunks, codebook, labels)
    display_codebook(codebook, labels)
