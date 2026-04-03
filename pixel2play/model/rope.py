from typing import Optional

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE). https://arxiv.org/abs/2104.09864"""

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        seq = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        freqs = torch.einsum("i,j->ij", seq, self.theta)          # (max_seq_len, head_dim/2)
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (..., 2)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, n_heads, head_dim)
        S = x.size(1)
        rope = self.cache[:S] if input_pos is None else self.cache[input_pos]
        rope = rope.view(-1, S, 1, x.size(-1) // 2, 2)     # broadcast over heads

        xr = x.float().reshape(*x.shape[:-1], -1, 2)        # (..., head_dim/2, 2)
        x_out = torch.stack([
            xr[..., 0] * rope[..., 0] - xr[..., 1] * rope[..., 1],
            xr[..., 1] * rope[..., 0] + xr[..., 0] * rope[..., 1],
        ], dim=-1).flatten(3)
        return x_out.type_as(x)
