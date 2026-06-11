"""Causal action transformer used by ContraVLA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden size {dim} must be divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        sdpa_mask = None
        sdpa_is_causal = is_causal
        if attention_mask is not None:
            key_mask = attention_mask.to(device=x.device, dtype=torch.bool).view(
                batch_size, 1, 1, seq_len
            )
            sdpa_mask = key_mask
            if is_causal:
                causal_mask = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=x.device
                ).tril()
                sdpa_mask = key_mask & causal_mask.view(1, 1, seq_len, seq_len)
                sdpa_is_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            is_causal=sdpa_is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), is_causal=is_causal, attention_mask=attention_mask
        )
        x = x + self.mlp(self.norm2(x))
        return x


class CausalActionTransformer(nn.Module):
    """Predict discrete NES action tokens from VLM features and RAM state.

    Sequence layout:
        [vlm_tokens..., state_token, action_query_0, ...]

    Causal self-attention lets action queries read the VLM prefix and state while
    preserving support for future multi-action chunks.
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
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.num_actions = num_actions

        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_size)
        self.action_queries = nn.Parameter(torch.empty(1, num_actions, hidden_size))
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.action_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vlm_features: torch.Tensor,
        proprio: torch.Tensor,
        vlm_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if proprio.ndim != 2:
            raise ValueError(f"proprio must be [B, D], got {tuple(proprio.shape)}")
        if vlm_features.ndim != 3:
            raise ValueError(
                f"vlm_features must be [B, T, D], got {tuple(vlm_features.shape)}"
            )

        batch_size, vlm_len, _ = vlm_features.shape
        if vlm_attention_mask is not None and vlm_attention_mask.shape != (batch_size, vlm_len):
            raise ValueError(
                "vlm_attention_mask must be "
                f"{(batch_size, vlm_len)}, got {tuple(vlm_attention_mask.shape)}"
            )

        vlm_tokens = self.vlm_proj(vlm_features)
        state_token = self.proprio_proj(proprio).unsqueeze(1)
        action_queries = self.action_queries.expand(batch_size, -1, -1)

        x = torch.cat([vlm_tokens, state_token, action_queries], dim=1)
        attention_mask = None
        if vlm_attention_mask is not None:
            suffix_mask = torch.ones(
                batch_size,
                1 + self.num_actions,
                dtype=torch.bool,
                device=vlm_attention_mask.device,
            )
            attention_mask = torch.cat(
                [vlm_attention_mask.to(torch.bool), suffix_mask], dim=1
            )

        for block in self.blocks:
            x = block(x, is_causal=True, attention_mask=attention_mask)

        action_states = self.norm(x[:, vlm_len + 1 :])
        return self.head(action_states)
