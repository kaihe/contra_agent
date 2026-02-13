"""
Lexicographic Objectives — Core Data Structures & Scoring
==========================================================

Implements the objective representation from Tom Murphy VII's learnfun/playfun.
An objective is an ordered tuple of RAM byte indices. Two RAM snapshots are
compared lexicographically over these indices to determine which is "better."
"""

from __future__ import annotations

import json
import numpy as np


def lex_compare(obj_indices: list[int], ram1: np.ndarray, ram2: np.ndarray) -> int:
    """Compare two RAM states over ordered byte indices.

    Returns +1 if ram2 > ram1, -1 if ram2 < ram1, 0 if equal.
    """
    for idx in obj_indices:
        v1 = int(ram1[idx])
        v2 = int(ram2[idx])
        if v2 > v1:
            return 1
        if v2 < v1:
            return -1
    return 0


class WeightedObjectives:
    """Collection of objectives with weights."""

    def __init__(self, objectives: list[list[int]], weights: list[float]):
        assert len(objectives) == len(weights)
        self.objectives = objectives
        self.weights = weights
        # Per-objective history of observed values (for value_frac scoring)
        # Each entry: sorted list of observed values at the objective's first byte
        self.observed: list[list[int]] = [[] for _ in objectives]

    def save(self, path: str) -> None:
        """Save objectives and weights to JSON file."""
        data = {
            "objectives": self.objectives,
            "weights": self.weights,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> WeightedObjectives:
        """Load objectives and weights from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["objectives"], data["weights"])

    def observe(self, ram: np.ndarray) -> None:
        """Record a RAM snapshot into per-objective history for value_frac."""
        for i, obj in enumerate(self.objectives):
            if obj:
                val = int(ram[obj[0]])
                self.observed[i].append(val)

    def evaluate(self, ram_before: np.ndarray, ram_after: np.ndarray) -> float:
        """Score a transition by lexicographic comparison on each objective.

        For each objective with nonzero weight:
          +weight if ram_after > ram_before (lexicographically)
          -weight if ram_after < ram_before
        Returns the sum.
        """
        score = 0.0
        for obj, w in zip(self.objectives, self.weights):
            if w == 0.0:
                continue
            cmp = lex_compare(obj, ram_before, ram_after)
            score += cmp * w
        return score

    def get_value_frac(self, objective_idx: int, ram: np.ndarray) -> float:
        """Where does current RAM value fall in observed history? (0.0 to 1.0).

        Used for integral scoring in playfun — higher means the objective
        is at a historically high value.
        """
        obj = self.objectives[objective_idx]
        if not obj:
            return 0.5
        val = int(ram[obj[0]])
        hist = self.observed[objective_idx]
        if not hist:
            return 0.5
        below = sum(1 for v in hist if v < val)
        return below / len(hist)

    def weight_by_examples(self, memories: list[np.ndarray]) -> None:
        """Set weight=1 for objectives that improved over the recording, else 0.

        An objective "improved" if its lexicographic value increased at least
        once between consecutive frames during the recording.
        """
        for i, obj in enumerate(self.objectives):
            improved = False
            for t in range(1, len(memories)):
                if lex_compare(obj, memories[t - 1], memories[t]) > 0:
                    improved = True
                    break
            self.weights[i] = 1.0 if improved else 0.0

    def active_count(self) -> int:
        """Number of objectives with nonzero weight."""
        return sum(1 for w in self.weights if w != 0.0)
