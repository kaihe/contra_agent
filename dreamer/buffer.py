"""Component 2 — episode replay buffer for Dreamer.

Stores a flat ring of steps and samples fixed-length subsequences for world-model
training. Subsequences MAY cross episode boundaries; each step carries `is_first`
so the RSSM resets its latent state at the boundary (the standard DreamerV3
trick). This matters for Contra, where deaths are frequent and episodes short —
forbidding boundary-crossing would waste most of the data.

What we store per step (see ReplayBuffer.add):
    image       uint8 (H,W,3)  — the observation BEFORE the action
    action      int            — discrete action index taken
    reward      float          — reward received after the action
    is_first    bool           — first step of an episode (reset RSSM here)
    is_terminal bool           — real MDP terminal (death/win), NOT timeout

`cont = 1 - is_terminal` is the world model's continue target. A timeout is
is_terminal=False (its future should be bootstrapped, not zeroed).

Verification gate (`python -m dreamer.buffer --smoke`):
  * fill from a real Contra rollout,
  * sample a batch and assert shapes / dtypes / ranges / one-hot / cont,
  * assert is_first lands exactly on reset steps,
  * dump a sampled sequence as a GIF that is contiguous gameplay, with the
    reset boundary (if any) visible as a scene cut.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, int, int],
                 num_actions: int, seq_len: int, device: str = "cpu"):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.device = device

        H, W, C = obs_shape
        self.image = np.zeros((capacity, H, W, C), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.is_first = np.zeros(capacity, dtype=bool)
        self.is_terminal = np.zeros(capacity, dtype=bool)

        self.ptr = 0      # next write index
        self.size = 0     # number of valid steps
        self.full = False

    def add(self, image, action, reward, is_first, is_terminal):
        i = self.ptr
        self.image[i] = image
        self.action[i] = action
        self.reward[i] = reward
        self.is_first[i] = is_first
        self.is_terminal[i] = is_terminal
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.full = self.full or self.ptr == 0

    def _valid_start(self, s: int) -> bool:
        """A length-`seq_len` window at physical start `s` is valid if it stays
        in stored data and does not straddle the ring write head (the only
        discontinuity in time order once the buffer has wrapped)."""
        if s + self.seq_len > self.capacity:
            return False
        if not self.full:
            return s + self.seq_len <= self.size
        # Wrapped: the gap sits between index ptr-1 (newest) and ptr (oldest).
        return not (s < self.ptr <= s + self.seq_len)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= self.seq_len + 1

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a batch of subsequences as tensors of shape (B, L, ...).

        image    float32 (B,L,H,W,C) in [0,1)
        action   float32 (B,L,A)      one-hot
        reward   float32 (B,L)
        is_first float32 (B,L)
        cont     float32 (B,L)        1 - is_terminal
        """
        hi = (self.capacity if self.full else self.size) - self.seq_len
        starts = []
        while len(starts) < batch_size:
            s = np.random.randint(0, max(hi, 1))
            if self._valid_start(s):
                starts.append(s)
        idx = np.stack([np.arange(s, s + self.seq_len) for s in starts])  # (B,L)

        img = torch.as_tensor(self.image[idx], dtype=torch.float32) / 255.0
        act_idx = torch.as_tensor(self.action[idx], dtype=torch.long)
        act = torch.nn.functional.one_hot(act_idx, self.num_actions).float()
        rew = torch.as_tensor(self.reward[idx], dtype=torch.float32)
        first = torch.as_tensor(self.is_first[idx], dtype=torch.float32)
        cont = 1.0 - torch.as_tensor(self.is_terminal[idx], dtype=torch.float32)

        out = {"image": img, "action": act, "reward": rew,
               "is_first": first, "cont": cont}
        return {k: v.to(self.device) for k, v in out.items()}


# ── Verification gate ────────────────────────────────────────────────────────

def _fill_from_env(buf: ReplayBuffer, steps: int, seed: int) -> int:
    """Drive Contra with a Right-biased random policy so episodes actually end."""
    from dreamer.envs import make_contra_env, ACTION_NAMES

    rng = np.random.default_rng(seed)
    RF = ACTION_NAMES.index("RF")
    n_actions = len(ACTION_NAMES)
    env = make_contra_env(level=1, size=buf.obs_shape[0])
    resets = 0
    try:
        obs, _ = env.reset(seed=seed)
        is_first = True
        carry_r, carry_term = 0.0, False
        for _ in range(steps):
            # Mostly hold Right+Fire so we make progress and actually reach
            # deaths (pure RF dies ~every 50 steps); occasional noise for variety.
            a = RF if rng.random() < 0.9 else int(rng.integers(n_actions))
            nobs, r, term, trunc, _ = env.step(a)
            # DreamerV3 convention: the reward/terminal stored at a step describe
            # the transition INTO that observation (so the latent state, which has
            # seen the action that produced it, can predict them).
            buf.add(obs, a, carry_r, is_first, carry_term)
            is_first = False
            if term or trunc:
                buf.add(nobs, 0, r, False, bool(term))   # the terminal observation
                obs, _ = env.reset()
                is_first = True
                carry_r, carry_term = 0.0, False
                resets += 1
            else:
                obs, carry_r, carry_term = nobs, r, False
    finally:
        env.close()
    return resets


def _smoke(steps: int, seq_len: int, batch_size: int, seed: int) -> None:
    import imageio

    H = 128
    buf = ReplayBuffer(capacity=steps + 10, obs_shape=(H, H, 3),
                       num_actions=21, seq_len=seq_len)
    resets = _fill_from_env(buf, steps, seed)
    print(f"[smoke] filled {buf.size} steps, {resets} episode resets, "
          f"{int(buf.is_terminal.sum())} terminals, {int(buf.is_first.sum())} firsts")
    assert resets > 0, "no episode ended — boundary logic would go untested"

    assert buf.can_sample(batch_size), "not enough data to sample"
    batch = buf.sample(batch_size)

    # shape / dtype / range invariants
    assert batch["image"].shape == (batch_size, seq_len, H, H, 3)
    assert batch["action"].shape == (batch_size, seq_len, 21)
    assert 0.0 <= batch["image"].min() and batch["image"].max() < 1.0 + 1e-6
    assert torch.allclose(batch["action"].sum(-1), torch.ones(batch_size, seq_len)), \
        "actions are not one-hot"
    assert set(batch["cont"].unique().tolist()) <= {0.0, 1.0}
    assert set(batch["is_first"].unique().tolist()) <= {0.0, 1.0}

    # is_first must align with terminals: a step right after a terminal is a first.
    # Check directly in the raw ring (sampling is random, so verify the source).
    term_idx = np.where(buf.is_terminal[:buf.size])[0]
    for t in term_idx:
        nxt = t + 1
        if nxt < buf.size:
            assert buf.is_first[nxt], f"step after terminal at {t} is not is_first"
    print("  invariants OK: shapes, one-hot, cont∈{0,1}, is_first follows terminals")

    # Visual gate: dump the batch row that contains the most resets, so the
    # boundary is visible as a scene cut, and confirm frames are contiguous.
    firsts_per_row = batch["is_first"].sum(1)
    row = int(torch.argmax(firsts_per_row).item())
    seq = (batch["image"][row].cpu().numpy() * 255).astype(np.uint8)
    from dreamer import out_path
    gif = out_path("buffer_sample.gif")
    imageio.mimsave(gif, list(seq), duration=80, loop=0)
    where_first = torch.nonzero(batch["is_first"][row]).flatten().tolist()
    print(f"  sampled row {row}: is_first at {where_first or 'none'}  → {gif}")
    print("  eyeball the GIF: contiguous play; any is_first index = a scene cut")


def main() -> None:
    p = argparse.ArgumentParser(description="Dreamer replay buffer")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.smoke:
        _smoke(args.steps, args.seq_len, args.batch_size, args.seed)
    else:
        p.error("nothing to do; pass --smoke")


if __name__ == "__main__":
    main()
