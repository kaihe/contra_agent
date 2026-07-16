"""World-model training data generator — mixed batches, the default pipeline.

Encapsulates the two data sources the world model needs and hands back ready-to-train
batches, so train_wm (and later C6) just call `.sample(batch)`:

  * win-trace frames         — whole-level coverage (the agent never dies)
  * anchor-branched rollouts — biased-random rollouts branched from save-state
                               anchors along the traces (dreamer.anchor_gen), giving
                               DEATH / off-policy states across the WHOLE level

Each `.sample(batch)` returns a `trace_frac` mix of the two (episode-aware sequence
subsamples, on-device). `.refresh()` regenerates the random rollouts from new anchors
+ new randomness — call it periodically for an endlessly-fresh failure stream (the
"switch among seeds" behaviour), or just fill once.

    data = WMDataGenerator(size=256, seq_len=20, device="cuda")
    batch = data.sample(16)          # 8 trace seqs + 8 death-rollout seqs
    data.refresh()                   # new random rollouts into the ring
"""

from __future__ import annotations

import numpy as np
import torch

from dreamer.anchor_gen import branch_into_buffer, snapshot_anchors
from dreamer.buffer import ReplayBuffer
from dreamer.collect import fill_buffer_from_traces, trace_paths


class WMDataGenerator:
    """Mixed win-trace + anchor-branched-random batches for world-model training."""

    def __init__(self, level: int = 1, size: int = 256, seq_len: int = 20,
                 device: str = "cpu", *, train_traces: int = 8, trace_frac: float = 0.5,
                 anchor_traces: int = 16, anchor_stride: int = 20,
                 anchor_rollouts: int = 300, anchor_max_steps: int = 150,
                 rand_cap: int = 25000, seed: int = 0, verbose: bool = True):
        self.trace_frac = trace_frac
        self.anchor_rollouts = anchor_rollouts
        self.anchor_max_steps = anchor_max_steps
        self._seed = seed
        self._refresh_ctr = 0

        # source 1 — win-trace FRAMES (only one emulator at a time: this fills+closes,
        # then snapshot_anchors / branch_into_buffer each open+close their own).
        paths = trace_paths(level)[:train_traces]
        if not paths:
            raise SystemExit(f"no traces at tmp/mc_trace/level{level}/")
        tlen = sum(len(np.load(q, allow_pickle=True)["actions"]) for q in paths)
        self.trace_buf = ReplayBuffer(tlen + 64, (size, size, 3), 21, seq_len, device)
        fill_buffer_from_traces(self.trace_buf, paths, verbose=False)

        # source 2 — anchor-branched DEATH rollouts (cheap save-state seeds → generated
        # frames in a ring buffer, refreshable).
        self.anchors = snapshot_anchors(trace_paths(level)[:anchor_traces],
                                        stride=anchor_stride, verbose=verbose)
        self.rand_buf = ReplayBuffer(rand_cap, (size, size, 3), 21, seq_len, device)
        deaths = self.refresh()

        if verbose:
            print(f"[wmdata] trace_buf={self.trace_buf.size} ({len(paths)} traces)  "
                  f"rand_buf={self.rand_buf.size} ({deaths} deaths / {len(self.anchors)} anchors)"
                  f"  trace_frac={trace_frac}")

    def refresh(self, n_rollouts: int | None = None) -> int:
        """(Re)generate anchor-branched rollouts into the ring buffer. Returns deaths."""
        self._refresh_ctr += 1
        return branch_into_buffer(self.rand_buf, self.anchors,
                                  n_rollouts or self.anchor_rollouts,
                                  max_steps=self.anchor_max_steps,
                                  seed=self._seed + self._refresh_ctr)

    def sample(self, batch: int) -> dict[str, torch.Tensor]:
        """One mixed batch: `trace_frac` sequences from win traces, the rest random."""
        if not self.rand_buf.can_sample(1):
            return self.trace_buf.sample(batch)
        n_t = max(1, round(batch * self.trace_frac))
        bt = self.trace_buf.sample(n_t)
        br = self.rand_buf.sample(max(1, batch - n_t))
        return {k: torch.cat([bt[k], br[k]], 0) for k in bt}

    def batches(self, batch: int):
        """Infinite generator of mixed batches: `for b in data.batches(16): ...`."""
        while True:
            yield self.sample(batch)
