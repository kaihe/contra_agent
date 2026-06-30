"""Shared Dreamer network modules.

Component 3a lives here: ConvEncoder / ConvDecoder. They are written to be reused
unchanged once the RSSM arrives — the encoder turns a frame into an embedding
that will condition the RSSM posterior, and the decoder reconstructs a frame from
a feature vector (in 3a that feature is the encoder embedding; from 3b on it will
be the RSSM state). The conv depth is derived from the image `size`, so 128 gives
five stride-2 layers (128→64→32→16→8→4, minres=4).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _num_layers(size: int, minres: int = 4) -> int:
    """How many stride-2 halvings take `size` down to `minres` (power-of-two)."""
    n = int(round(math.log2(size / minres)))
    assert minres * 2 ** n == size, f"size {size} is not minres*2^k (minres={minres})"
    return n


def _norm(channels: int) -> nn.Module:
    # GroupNorm ≈ DreamerV3's per-layer LayerNorm. Pick the largest group count
    # ≤32 that divides `channels` (depth need not be a power of two, e.g. 48).
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvEncoder(nn.Module):
    """Frame (B,3,size,size) in [0,1] → embedding (B, embed_dim)."""

    def __init__(self, size: int = 128, in_ch: int = 3, depth: int = 32,
                 embed_dim: int = 1024, minres: int = 4):
        super().__init__()
        n = _num_layers(size, minres)
        layers: list[nn.Module] = []
        ch = in_ch
        for i in range(n):
            out = depth * (2 ** i)
            layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), _norm(out), nn.SiLU()]
            ch = out
        self.convs = nn.Sequential(*layers)
        self.minres = minres
        self.conv_out_ch = ch                       # e.g. 512 at depth=32, n=5
        self.flat_dim = ch * minres * minres        # e.g. 8192
        self.head = nn.Linear(self.flat_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x)
        return self.head(h.flatten(1))


class ConvDecoder(nn.Module):
    """Feature (B, feat_dim) → reconstructed frame (B,3,size,size) in [0,1]."""

    def __init__(self, size: int = 128, out_ch: int = 3, depth: int = 32,
                 feat_dim: int = 1024, minres: int = 4):
        super().__init__()
        n = _num_layers(size, minres)
        self.minres = minres
        self.start_ch = depth * (2 ** (n - 1))      # mirror of encoder's last ch
        self.fc = nn.Linear(feat_dim, self.start_ch * minres * minres)

        layers: list[nn.Module] = []
        ch = self.start_ch
        for i in reversed(range(n)):
            last = i == 0
            out = out_ch if last else depth * (2 ** (i - 1))
            layers += [nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1)]
            if not last:
                layers += [_norm(out), nn.SiLU()]
            ch = out
        self.deconvs = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.fc(feat).view(-1, self.start_ch, self.minres, self.minres)
        return torch.sigmoid(self.deconvs(h))
