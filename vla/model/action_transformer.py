"""Causal action transformer for discrete action chunk prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), is_causal)
        x = x + self.mlp(self.norm2(x))
        return x


class CausalActionTransformer(nn.Module):
    """
    Predicts T discrete action logits given VLM features and structured state.

    Sequence layout (causal attention, SimVLA-style):
        [vlm_token_0, ..., vlm_token_{T_vlm-1}, proprio_token, query_0, ..., query_{T-1}]
         ^^^^^^^^^^^^^ VLM prefix (T_vlm) ^^^^^^^^^^^^^^^^^^^^^  action queries (T) ^^^^^^

    Full VLM feature sequence is kept (no mean-pooling). Token i can attend to
    tokens 0..i via is_causal=True. Logits are read from the last T positions.
    """

    def __init__(
        self,
        hidden_size: int,
        vlm_hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        action_dim: int,
        num_actions: int,
        proprio_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        self.vlm_proj     = nn.Linear(vlm_hidden_size, hidden_size)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_size)
        self.action_queries = nn.Parameter(torch.zeros(1, num_actions, hidden_size))
        nn.init.normal_(self.action_queries, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vlm_features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_features: [B, T_vlm, D_vlm]  — full fused VLM output (all tokens kept)
            proprio:      [B, proprio_dim]    — 118-dim structured state
        Returns:
            logits: [B, T, action_dim]
        """
        B, T_vlm, _ = vlm_features.shape

        vlm_tokens    = self.vlm_proj(vlm_features)             # [B, T_vlm, H]
        proprio_token = self.proprio_proj(proprio).unsqueeze(1) # [B, 1, H]
        queries       = self.action_queries.expand(B, -1, -1)   # [B, T, H]

        # [B, T_vlm + 1 + T, H]
        x = torch.cat([vlm_tokens, proprio_token, queries], dim=1)

        for block in self.blocks:
            x = block(x, is_causal=True)

        x = self.norm(x[:, T_vlm + 1:])  # [B, T, H]  — last T positions
        return self.head(x)               # [B, T, action_dim]
