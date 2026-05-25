"""
RolloutBuffer — stores GRPO rollout data for one training iteration.

Layout
------
Each GRPO iteration collects N_groups × G episodes.  All G episodes in a
group start from the same emulator snapshot.  After collection the buffer:

  1. Computes per-episode discounted returns.
  2. Normalises returns within each group (G members) to produce advantages.
  3. Clips advantages to [-adv_clip, adv_clip] to dampen death-penalty spikes.
  4. Exposes a DataLoader-style batches() iterator for the policy update loop.

Memory layout
-------------
We keep everything as numpy arrays until batches() is called, which converts
to tensors and moves to device on the fly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch


# ── Per-transition data class ─────────────────────────────────────────────────

@dataclass
class Transition:
    images:       np.ndarray  # [2, 3, H, W] float32, ImageNet-normalised
    proprio:      np.ndarray  # [118] float32
    actions:      np.ndarray  # [T] int64, combined indices 0..35
    log_prob_old: float        # sum_{t} log π_old(a_t | s)  at collection time
    log_prob_ref: float        # sum_{t} log π_ref(a_t | s)  frozen BC model


@dataclass
class Episode:
    transitions:   list[Transition]
    group_id:      int
    raw_return:    float = 0.0   # Σ γ^k r_k across the rollout
    advantage:     float = 0.0   # set by compute_advantages()


# ── Buffer ────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Accumulates episodes grouped by their shared starting state.

    Usage
    -----
    buf = RolloutBuffer()

    # during rollout collection
    buf.begin_episode(group_id)
    buf.add_transition(...)       # one call per chunk step
    buf.end_episode(raw_return)

    # after collection
    buf.compute_advantages(adv_clip=5.0)

    # during policy update (multiple epochs)
    for batch in buf.batches(batch_size=24, device=device):
        ...  # dict of tensors

    buf.clear()
    """

    def __init__(self) -> None:
        self._episodes: list[Episode] = []
        self._current:  Episode | None = None

    # ── collection API ────────────────────────────────────────────────────────

    def begin_episode(self, group_id: int) -> None:
        self._current = Episode(transitions=[], group_id=group_id)

    def add_transition(
        self,
        images:       np.ndarray,
        proprio:      np.ndarray,
        actions:      np.ndarray,
        log_prob_old: float,
        log_prob_ref: float,
    ) -> None:
        assert self._current is not None, "Call begin_episode() first"
        self._current.transitions.append(Transition(
            images=images.copy(),
            proprio=proprio.copy(),
            actions=actions.copy(),
            log_prob_old=log_prob_old,
            log_prob_ref=log_prob_ref,
        ))

    def end_episode(self, raw_return: float) -> None:
        assert self._current is not None
        self._current.raw_return = raw_return
        self._episodes.append(self._current)
        self._current = None

    # ── post-collection ───────────────────────────────────────────────────────

    def compute_advantages(self, adv_clip: float = 5.0) -> None:
        """
        Group-normalise episode returns, clip, and assign to every transition
        in the episode.  Groups with only one surviving episode get advantage 0.
        """
        # collect returns per group
        from collections import defaultdict
        group_returns: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for ep_idx, ep in enumerate(self._episodes):
            group_returns[ep.group_id].append((ep_idx, ep.raw_return))

        for ep_idx_list in group_returns.values():
            rets = [r for _, r in ep_idx_list]
            mean_r = float(np.mean(rets))
            std_r  = float(np.std(rets)) + 1e-8

            for ep_idx, raw_r in ep_idx_list:
                adv = (raw_r - mean_r) / std_r
                adv = float(np.clip(adv, -adv_clip, adv_clip))
                self._episodes[ep_idx].advantage = adv

    # ── minibatch iterator ────────────────────────────────────────────────────

    def batches(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield shuffled minibatches of transitions for the policy update.

        Each batch dict contains:
            images       : [B, 2, 3, H, W] float32
            proprio      : [B, 118]         float32
            actions      : [B, T]           int64
            log_prob_old : [B]              float32
            log_prob_ref : [B]              float32
            advantages   : [B]              float32
        """
        # Flatten all transitions from all episodes
        all_images, all_proprio, all_actions = [], [], []
        all_lp_old, all_lp_ref, all_adv      = [], [], []

        for ep in self._episodes:
            for tr in ep.transitions:
                all_images.append(tr.images)
                all_proprio.append(tr.proprio)
                all_actions.append(tr.actions)
                all_lp_old.append(tr.log_prob_old)
                all_lp_ref.append(tr.log_prob_ref)
                all_adv.append(ep.advantage)   # episode-level advantage

        n = len(all_images)
        if n == 0:
            return

        # Shuffle once per epoch call
        indices = list(range(n))
        random.shuffle(indices)

        # Convert to numpy first, then to tensors in chunks
        images_arr  = np.stack(all_images)   # [N, 2, 3, H, W]
        proprio_arr = np.stack(all_proprio)  # [N, 118]
        actions_arr = np.stack(all_actions)  # [N, T]
        lp_old_arr  = np.array(all_lp_old, dtype=np.float32)
        lp_ref_arr  = np.array(all_lp_ref, dtype=np.float32)
        adv_arr     = np.array(all_adv,    dtype=np.float32)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "images":       torch.from_numpy(images_arr[idx]).to(device),
                "proprio":      torch.from_numpy(proprio_arr[idx]).to(device),
                "actions":      torch.from_numpy(actions_arr[idx]).long().to(device),
                "log_prob_old": torch.from_numpy(lp_old_arr[idx]).to(device),
                "log_prob_ref": torch.from_numpy(lp_ref_arr[idx]).to(device),
                "advantages":   torch.from_numpy(adv_arr[idx]).to(device),
            }

    # ── stats / utils ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._episodes.clear()
        self._current = None

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def n_transitions(self) -> int:
        return sum(len(ep.transitions) for ep in self._episodes)

    def return_stats(self) -> dict[str, float]:
        rets = [ep.raw_return for ep in self._episodes]
        if not rets:
            return {}
        return {
            "mean": float(np.mean(rets)),
            "std":  float(np.std(rets)),
            "min":  float(np.min(rets)),
            "max":  float(np.max(rets)),
        }
