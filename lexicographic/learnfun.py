"""
learnfun — Discover Objectives from Recorded Gameplay
=======================================================

Loads RAM snapshots from a .npz recording (created by play_human.py) and
discovers which RAM byte orderings consistently increase. These become the
objectives that playfun uses to score candidate input sequences.

Based on Tom Murphy VII's learnfun algorithm.

Usage:
    python learnfun.py
"""

from __future__ import annotations

import os
import glob

import numpy as np

from objectives import WeightedObjectives

# =============================================================================
# CONFIG
# =============================================================================

RECORDING_DIR = os.path.join(os.path.dirname(__file__), "recordings")
MAX_OBJECTIVES = 50
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "contra.objectives")


# =============================================================================
# LOAD MEMORIES FROM RECORDING
# =============================================================================

def load_memories() -> list[np.ndarray]:
    """Load RAM snapshots from the most recent .npz recording."""
    pattern = os.path.join(RECORDING_DIR, "human_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No recordings found in {RECORDING_DIR}/")

    filepath = files[-1]  # most recent
    print(f"Loading recording: {filepath}")

    data = np.load(filepath)
    ram = data["ram"]  # (N, ram_size) uint8
    print(f"  {ram.shape[0]} RAM snapshots x {ram.shape[1]} bytes")

    return [ram[i] for i in range(ram.shape[0])]


# =============================================================================
# ENUMERATE OBJECTIVES
# =============================================================================

def enumerate_objectives(memories: list[np.ndarray]) -> list[list[int]]:
    """Find RAM byte orderings that consistently increase during gameplay.

    For each byte index, compute how often it increases vs decreases between
    consecutive frames. Keep bytes with positive trend, then build single-byte
    and two-byte orderings from the most reliable ones.
    """
    ram_size = memories[0].shape[0]
    n = len(memories)

    # Count increases and decreases for each byte
    increases = np.zeros(ram_size, dtype=np.int64)
    decreases = np.zeros(ram_size, dtype=np.int64)

    for t in range(1, n):
        prev = memories[t - 1].astype(np.int16)
        curr = memories[t].astype(np.int16)
        diff = curr - prev
        increases += (diff > 0).astype(np.int64)
        decreases += (diff < 0).astype(np.int64)

    # Trend score: net increase frequency
    trend = increases - decreases
    changes = increases + decreases

    # Filter: byte must have changed at least a few times and have positive trend
    min_changes = max(10, n // 100)
    candidates = []
    for idx in range(ram_size):
        if changes[idx] >= min_changes and trend[idx] > 0:
            reliability = increases[idx] / changes[idx] if changes[idx] > 0 else 0
            candidates.append((idx, trend[idx], reliability))

    # Sort by reliability (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    print(f"Found {len(candidates)} trending RAM bytes")

    # Build objectives: start with single-byte objectives from best candidates
    objectives: list[list[int]] = []

    for idx, _trend, reliability in candidates:
        if len(objectives) >= MAX_OBJECTIVES:
            break
        objectives.append([idx])

    # Add two-byte objectives from the top candidates (most-significant, least-significant)
    top = candidates[: min(15, len(candidates))]
    for i, (idx_a, _, _) in enumerate(top):
        for j, (idx_b, _, _) in enumerate(top):
            if i != j and len(objectives) < MAX_OBJECTIVES:
                pair = [idx_a, idx_b]
                if pair not in objectives:
                    objectives.append(pair)

    print(f"Generated {len(objectives)} candidate objectives")
    return objectives


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"learnfun: discovering objectives from human gameplay")
    print(f"  Recording dir:  {RECORDING_DIR}")
    print(f"  Max objectives: {MAX_OBJECTIVES}")
    print(f"  Output:         {OUTPUT_PATH}")

    # Phase 1: Load RAM snapshots from recording
    print("\n--- Loading memories ---")
    memories = load_memories()

    # Phase 2: Discover objectives
    print("\n--- Enumerating objectives ---")
    objectives = enumerate_objectives(memories)

    # Phase 3: Weight by examples — keep only objectives that actually improved
    initial_weights = [1.0] * len(objectives)
    wo = WeightedObjectives(objectives, initial_weights)
    wo.weight_by_examples(memories)

    active = wo.active_count()
    print(f"\n{active}/{len(objectives)} objectives survived weighting")

    # Save
    wo.save(OUTPUT_PATH)
    print(f"Saved objectives to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
