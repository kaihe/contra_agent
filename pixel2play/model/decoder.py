"""
ActionDecoder: small causal transformer that maps a single "action_out" token
from the backbone into an autoregressive sequence of N action tokens.

Forward (training):
  input_action_token : (B, T, backbone_dim)   – action_out embedding per timestep
  action_embeddings_in : (B, T, N-1, dec_dim) – teacher-forced action embeddings
  output              : (B, T, N-1, dec_dim)  – predicted embeddings for each action slot
"""

import torch
import torch.nn as nn

from pixel2play.model.attention import Transformer


class ActionDecoder(nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        dim: int,
        n_action_tokens: int,   # total slots = N-1 real + 1 start → N
        n_layers: int = 3,
        n_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        # n_action_tokens here is N (start token + N-1 real tokens)
        self.n_slots = n_action_tokens

        # Project backbone action_out token into decoder dim
        self.input_proj = nn.Linear(backbone_dim, dim, bias=False)

        # Learned position token per slot
        self.pos_tokens = nn.Parameter(
            torch.empty(n_action_tokens, dim, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.pos_tokens, std=3.0)

        self.transformer = Transformer(
            dim=dim,
            n_layers=n_layers,
            n_q_heads=n_heads,
            n_kv_heads=n_heads,
            max_seq_len=n_action_tokens,
            is_causal=True,
        )

    def forward(self, input_action_token: torch.Tensor, action_embeddings_in: torch.Tensor) -> torch.Tensor:
        """
        input_action_token  : (B, T, backbone_dim)
        action_embeddings_in: (B, T, N-1, dim)   – previous action embeddings (teacher-forced)
        returns             : (B, T, N-1, dim)   – decoded embeddings for each action slot
        """
        B, T, _ = input_action_token.shape
        N = self.n_slots  # = (N-1) + 1

        # Project start token and prepend to action embeddings
        start = self.input_proj(input_action_token).unsqueeze(2)          # (B, T, 1, dim)
        tokens = torch.cat([start, action_embeddings_in], dim=2)          # (B, T, N, dim)
        tokens = tokens + self.pos_tokens.unsqueeze(0).unsqueeze(0)       # add positional tokens

        # Run causal transformer over the action-slot dimension
        BT = B * T
        tokens = tokens.view(BT, N, self.dim)
        out = self.transformer(tokens)                                     # (BT, N, dim)
        out = out.view(B, T, N, self.dim)

        # Discard the last slot (it has no next-token target)
        return out[:, :, :-1, :]                                          # (B, T, N-1, dim)
