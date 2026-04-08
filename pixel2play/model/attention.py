"""
Core transformer blocks:
  SelfAttention      – GQA + RoPE + flex_attention
  SwiGLUFFN          – packed SwiGLU feed-forward
  TransformerLayer   – pre-norm residual block
  Transformer        – stack of TransformerLayers
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention as fa
from torch.nn.attention.flex_attention import BlockMask

from pixel2play.model.norm import RMSNorm
from pixel2play.model.rope import RotaryEmbedding


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        rope: RotaryEmbedding,
        is_causal: bool = False,
    ):
        super().__init__()
        assert dim % n_q_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_q_heads
        self.q_dim = dim
        self.kv_dim = n_kv_heads * self.head_dim
        self.is_causal = is_causal
        self.rope = rope

        self.qkv = nn.Linear(dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # QK-norm stabilises training
        self.q_norm = RMSNorm(self.q_dim)
        self.k_norm = RMSNorm(self.kv_dim)

        # Set by PolicyCausalTransformer after construction
        self.block_mask: Optional[BlockMask] = None

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = v.to(q.dtype)   # ensure all three match after norm dtype promotion

        # (B, S, n_heads, head_dim) → (B, n_heads, S, head_dim)
        q = self.rope(q.view(B, S, self.n_q_heads, self.head_dim), input_pos).transpose(1, 2)
        k = self.rope(k.view(B, S, self.n_kv_heads, self.head_dim), input_pos).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        gqa = self.n_q_heads != self.n_kv_heads

        if self.block_mask is not None:
            y = fa.flex_attention(q, k, v, block_mask=self.block_mask, enable_gqa=gqa)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, enable_gqa=gqa)

        y = y.transpose(1, 2).reshape(B, S, self.q_dim)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """Packed SwiGLU: fuses the gate and up-projection into one linear."""

    def __init__(self, dim: int, multiple_of: int = 8):
        super().__init__()
        hidden = int(2 * 4 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w_gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    def __init__(self, dim: int, attn: SelfAttention, ffn: SwiGLUFFN, dropout: float = 0.0):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), input_pos))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Transformer stack
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        is_causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = dim // n_q_heads
        rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim=dim,
                attn=SelfAttention(dim, n_q_heads, n_kv_heads, rope, is_causal),
                ffn=SwiGLUFFN(dim),
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, input_pos)
        return self.norm(x)
