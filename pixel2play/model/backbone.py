"""
PolicyCausalTransformer: the core backbone for pixel-to-play.

Per-timestep token layout (one_step tokens):
  [img_0] [think_0] [action_out] [action_0 .. action_{N-1}]
   ─────── ─────────  ─────────── ──────────────────────────
   n_img   n_think       1                  N (real actions)

T timesteps are concatenated, giving a sequence of T * one_step tokens.
The flex_attention block mask implements the causal cross-step and within-step rules:
  - All tokens see all non-action_out tokens from strictly past steps.
  - Within the same step:
      img/think  → attend to each other (fully bidirectional)
      action_out → attend to img/think + itself
      real_action → attend to img/think + all prior real_action tokens
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn.attention import flex_attention as fa

from pixel2play.model.attention import Transformer
from pixel2play.model.decoder import ActionDecoder
from pixel2play.model.tokenizer import RAMTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    # Transformer
    dim: int = 1024
    n_layers: int = 10
    n_q_heads: int = 16
    n_kv_heads: int = 16
    # Sequence
    n_steps: int = 200
    n_thinking_tokens: int = 1
    n_action_tokens: int = 2        # NES: dpad + buttons
    # RAM
    ram_size: int = 2048
    # Flex-attention block size
    mask_block_size: int = 128
    # Attention history per layer (length must equal n_layers)
    attention_history_len: List[int] = None
    # Dropout
    dropout: float = 0.1
    # Action decoder
    dec_n_layers: int = 3
    dec_n_heads: int = 8

    def __post_init__(self):
        if self.attention_history_len is None:
            self.attention_history_len = [self.n_steps] * self.n_layers


# ---------------------------------------------------------------------------
# Causal attention mask
# ---------------------------------------------------------------------------

def _build_causal_mask_fn(
    n_img: int,
    n_think: int,
    n_action: int,
    history_steps: int,
):
    """
    Returns a flex_attention mask_mod function for one layer.

    Token layout per step (length = one_step):
      [img × n_img | think × n_think | action_out × 1 | real_action × n_action]

    Indices within one step:
      img/think:   [0, token_to_action_out)
      action_out:  [token_to_action_out]
      real_action: (token_to_action_out, one_step)
    """
    one_step = n_img + n_think + 1 + n_action
    token_to_action_out = n_img + n_think

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx % one_step
        k_pos = kv_idx % one_step
        q_step = q_idx // one_step
        k_step = kv_idx // one_step

        q_is_context = q_pos < token_to_action_out
        q_is_action_out = q_pos == token_to_action_out
        q_is_real_action = q_pos > token_to_action_out

        k_is_context = k_pos < token_to_action_out
        k_is_action_out = k_pos == token_to_action_out

        past = k_step < q_step
        same = k_step == q_step
        in_history = (q_step - k_step) <= history_steps

        from_past = past & ~k_is_action_out & in_history

        same_context_to_context = same & q_is_context & k_is_context
        same_action_out_to_context_or_self = same & q_is_action_out & (k_is_context | k_is_action_out)
        same_real_action_to_non_action_out = same & q_is_real_action & ~k_is_action_out

        return from_past | same_context_to_context | same_action_out_to_context_or_self | same_real_action_to_non_action_out

    return mask_mod


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class PolicyCausalTransformer(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim

        # RAM tokenizer
        self.ram_tokenizer = RAMTokenizer(embed_dim=D)
        self.n_img = self.ram_tokenizer.n_img_tokens()

        # Sequence geometry
        self.one_step = self.n_img + cfg.n_thinking_tokens + 1 + cfg.n_action_tokens
        self.max_seq_len = self.one_step * cfg.n_steps

        # Backbone transformer
        self.transformer = Transformer(
            dim=D,
            n_layers=cfg.n_layers,
            n_q_heads=cfg.n_q_heads,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=self.max_seq_len,
            dropout=cfg.dropout,
        )

        # Action decoder (small causal transformer)
        self.action_decoder = ActionDecoder(
            backbone_dim=D,
            dim=D,
            n_action_tokens=cfg.n_action_tokens + 1,   # +1 for the start token
            n_layers=cfg.dec_n_layers,
            n_heads=cfg.dec_n_heads,
        )

        # Learned positional tokens (added to each token type)
        std = 0.05
        self.img_pos          = nn.Parameter(torch.empty(1, self.n_img,            D, dtype=torch.bfloat16))
        self.think_pos        = nn.Parameter(torch.empty(1, cfg.n_thinking_tokens, D, dtype=torch.bfloat16))
        self.action_out_token = nn.Parameter(torch.empty(1, 1,                     D, dtype=torch.bfloat16))
        self.action_pos       = nn.Parameter(torch.empty(1, cfg.n_action_tokens,   D, dtype=torch.bfloat16))
        for p in [self.img_pos, self.think_pos, self.action_out_token, self.action_pos]:
            nn.init.normal_(p, std=std)

        # Pre-compute which positions in the flat sequence are action_out tokens
        action_out_offset = self.n_img + cfg.n_thinking_tokens  # within one step
        action_out_idx = torch.tensor([
            i * self.one_step + action_out_offset for i in range(cfg.n_steps)
        ], dtype=torch.long)
        self.register_buffer("action_out_idx", action_out_idx, persistent=False)

        # Build per-layer flex_attention block masks
        self._build_block_masks()

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _build_block_masks(self):
        cfg = self.cfg
        for i, layer in enumerate(self.transformer.layers):
            history = cfg.attention_history_len[i]
            mask_fn = _build_causal_mask_fn(
                n_img=self.n_img,
                n_think=cfg.n_thinking_tokens,
                n_action=cfg.n_action_tokens,
                history_steps=history,
            )
            layer.attn.block_mask = fa.create_block_mask(
                mask_fn,
                B=None, H=None,
                Q_LEN=self.max_seq_len,
                KV_LEN=self.max_seq_len,
                BLOCK_SIZE=cfg.mask_block_size,
                device=next(self.parameters()).device,
            )

    def block_masks_to(self, device: torch.device):
        for layer in self.transformer.layers:
            if layer.attn.block_mask is not None:
                layer.attn.block_mask = layer.attn.block_mask.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _build_sequence(
        self,
        img: torch.Tensor,          # (B, T, n_img, D)
        action_in: torch.Tensor,    # (B, T, n_action, D)
    ) -> torch.Tensor:
        B = img.size(0)
        think = self.think_pos.expand(B, -1, -1)              # (B, n_think, D)
        action_out = self.action_out_token.expand(B, -1, -1)  # (B, 1, D)

        steps = []
        for t in range(self.cfg.n_steps):
            step = torch.cat([
                img[:, t]  + self.img_pos,
                think,
                action_out,
                action_in[:, t] + self.action_pos,
            ], dim=1)                                         # (B, one_step, D)
            steps.append(step)

        return torch.cat(steps, dim=1)                        # (B, max_seq_len, D)

    def _encode(
        self,
        ram: torch.Tensor,                  # (B, T, 2048)
        action_embeddings_in: torch.Tensor, # (B, T, n_action, D)
    ) -> torch.Tensor:
        """Run RAM encoding + backbone transformer. Returns action_out_tokens (B, T, D)."""
        img_tokens = self.ram_tokenizer(ram)  # (B, T, n_img, D)
        x = self._build_sequence(img_tokens, action_embeddings_in)
        y = self.transformer(x)
        return y[:, self.action_out_idx, :]   # (B, T, D)

    def forward(
        self,
        ram: torch.Tensor,                   # (B, T, 2048)
        action_embeddings_in: torch.Tensor,  # (B, T, N, D)
    ) -> torch.Tensor:
        """Returns action_out: (B, T, N, D)."""
        action_out_tokens = self._encode(ram, action_embeddings_in)
        return self.action_decoder(action_out_tokens, action_embeddings_in)
