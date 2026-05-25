"""
PolicyCausalTransformer: the core backbone for pixel-to-play.

Per-timestep token layout (one_step tokens):
  [img_0 .. img_{n_img-1}] [action_0 .. action_{N-1}]
   ────────────────────────  ────────────────────────
         n_img                     N (actions)

T timesteps are concatenated, giving a sequence of T * one_step tokens.
The flex_attention block mask implements the causal cross-step and within-step rules:
  - All tokens see all tokens from strictly past steps (within history).
  - Within the same step:
      img     → attend to itself (fully bidirectional within img group)
      action  → attend to img + all prior action tokens in the step
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.attention import flex_attention as fa

from pixel2play.model.attention import Transformer
from pixel2play.model.tokenizer import RAMTokenizer
from pixel2play.model.vision_encoder import SmallResNetEncoder


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
    n_action_tokens: int = 1        # NES: single combined action
    # Vision / RAM
    use_vision: bool = False
    in_channels: int = 1          # 1 for grayscale, 3 for RGB
    grid_size: int = 2            # Output spatial grid size per frame for vision encoder
    vision_depth: int = 4         # 3 for old 128-channel, 4 for new 256-channel
    ram_size: int = 2048
    # Flex-attention block size
    mask_block_size: int = 128
    # Attention history per layer (length must equal n_layers)
    attention_history_len: List[int] = None
    # Dropout
    dropout: float = 0.1
    # Token ablation for experiments: zero-out embeddings of listed token types
    # e.g. ["img"]  →  img tokens are zeroed out in the sequence
    ablate: Optional[List[str]] = None

    def __post_init__(self):
        if self.attention_history_len is None:
            self.attention_history_len = [self.n_steps] * self.n_layers


# ---------------------------------------------------------------------------
# Causal attention mask
# ---------------------------------------------------------------------------

def _build_causal_mask_fn(
    n_img: int,
    n_action: int,
    history_steps: int,
):
    """
    Returns a flex_attention mask_mod function for one layer.

    Token layout per step (length = one_step):
      [img × n_img | action × n_action]

    Indices within one step:
      img:    [0, n_img)
      action: [n_img, one_step)
    """
    one_step = n_img + n_action

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx % one_step
        k_pos = kv_idx % one_step
        q_step = q_idx // one_step
        k_step = kv_idx // one_step

        q_is_img = q_pos < n_img
        k_is_img = k_pos < n_img

        past = k_step < q_step
        same = k_step == q_step
        in_history = (q_step - k_step) <= history_steps

        # All tokens from past steps (within history window)
        from_past = past & in_history

        # Within the same step:
        # img tokens are fully bidirectional among themselves
        same_img_to_img = same & q_is_img & k_is_img
        # action tokens attend to img and to earlier action tokens in the step
        k_is_prior_action = (k_pos >= n_img) & (k_pos < q_pos)
        same_action_to_img_or_prior_action = same & ~q_is_img & (k_is_img | k_is_prior_action)

        return from_past | same_img_to_img | same_action_to_img_or_prior_action

    return mask_mod


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class PolicyCausalTransformer(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim

        # Image tokenizer (vision or RAM)
        if cfg.use_vision:
            self.img_tokenizer = SmallResNetEncoder(embed_dim=D, in_channels=cfg.in_channels, grid_size=cfg.grid_size, depth=cfg.vision_depth)
        else:
            self.img_tokenizer = RAMTokenizer(embed_dim=D)
        self.n_img = self.img_tokenizer.n_img_tokens()

        # Sequence geometry
        self.one_step = self.n_img + cfg.n_action_tokens
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

        # Learned positional tokens (added to each token type)
        std = 0.05
        self.img_pos     = nn.Parameter(torch.empty(1, self.n_img,          D, dtype=torch.bfloat16))
        self.action_pos  = nn.Parameter(torch.empty(1, cfg.n_action_tokens, D, dtype=torch.bfloat16))
        for p in [self.img_pos, self.action_pos]:
            nn.init.normal_(p, std=std)

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

        ablate = set(self.cfg.ablate or [])

        steps = []
        for t in range(self.cfg.n_steps):
            img_t    = (img[:, t] + self.img_pos)      if "img"    not in ablate else torch.zeros_like(img[:, t])
            action_t = (action_in[:, t] + self.action_pos) if "action" not in ablate else torch.zeros_like(action_in[:, t])

            step = torch.cat([img_t, action_t], dim=1)  # (B, one_step, D)
            steps.append(step)

        return torch.cat(steps, dim=1)                        # (B, max_seq_len, D)

    def _encode(
        self,
        obs: torch.Tensor,                  # (B, T, 2048) RAM or (B, T, C, H, W) frames
        action_embeddings_in: torch.Tensor, # (B, T, n_action, D)
    ) -> torch.Tensor:
        """Run observation encoding + backbone transformer. Returns per-timestep representation (B, T, D)."""
        D = self.cfg.dim
        img_tokens = self.img_tokenizer(obs)  # (B, T, n_img, D)
        B, T = img_tokens.shape[:2]
        x = self._build_sequence(img_tokens, action_embeddings_in)
        
        # All tokens in timestep t get RoPE temporal position t
        input_pos = torch.arange(T, device=x.device).repeat_interleave(self.one_step)
        
        y = self.transformer(x, input_pos=input_pos)        # (B, T * one_step, D)
        y = y.view(B, T, self.one_step, D)                  # (B, T, one_step, D)
        # Average img tokens per timestep to produce a single step representation
        return y[:, :, :self.n_img, :].mean(dim=2)          # (B, T, D)

    def forward(
        self,
        obs: torch.Tensor,                   # (B, T, 2048) RAM or (B, T, C, H, W) frames
        action_embeddings_in: torch.Tensor,  # (B, T, n_action, D)
    ) -> torch.Tensor:
        """Returns per-timestep representation: (B, T, D)."""
        return self._encode(obs, action_embeddings_in)
