"""
RAMTokenizer: maps a 2048-byte NES RAM snapshot to a fixed-size embedding vector.

The NES has 2 KB of RAM (2048 uint8 addresses). Values are normalised to [0, 1]
and projected to the transformer's embedding dimension with a single linear layer.
"""

import torch
import torch.nn as nn

RAM_SIZE = 2048


class RAMTokenizer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(RAM_SIZE, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, ram: torch.Tensor) -> torch.Tensor:
        # ram: (B, T, RAM_SIZE) uint8 or float
        x = ram.float() / 255.0
        out = self.proj(x)              # (B, T, embed_dim)
        return out.unsqueeze(2)         # (B, T, 1, embed_dim)

    def n_img_tokens(self) -> int:
        return 1
