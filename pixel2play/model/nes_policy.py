"""
NESPolicyModel: backbone + NES-specific action heads.

Input  (per step):
  frames  (B, T, 3, H, W)  – normalised pixels
  dpad    (B, T)            – ground-truth dpad class (teacher forcing)
  button  (B, T)            – ground-truth button class (teacher forcing)
  text    (B, T, 1, 768)    – Gemma text embedding (zeros if absent)

Output:
  dpad_logits   (B, T, N_DPAD)    – 9 class scores
  button_logits (B, T, N_BUTTONS) – 4 class scores
  loss          scalar            – normalised cross-entropy
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel2play.model.backbone import BackboneConfig, PolicyCausalTransformer
from pixel2play.model.nes_actions import N_BUTTONS, N_DPAD


class NESPolicyModel(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        D = cfg.dim
        assert cfg.n_action_tokens == 2, "NES uses 2 action tokens: dpad + button"

        self.backbone = PolicyCausalTransformer(cfg)

        # Action input embeddings (teacher forcing)
        self.dpad_embed   = nn.Embedding(N_DPAD,    D, dtype=torch.bfloat16)
        self.button_embed = nn.Embedding(N_BUTTONS, D, dtype=torch.bfloat16)
        nn.init.normal_(self.dpad_embed.weight,   std=0.1)
        nn.init.normal_(self.button_embed.weight, std=0.1)

        # Output heads
        self.dpad_head   = nn.Linear(D, N_DPAD)
        self.button_head = nn.Linear(D, N_BUTTONS)

    def forward(
        self,
        frames: torch.Tensor,   # (B, T, 3, H, W)
        dpad: torch.Tensor,     # (B, T)  int64
        button: torch.Tensor,   # (B, T)  int64
        text: torch.Tensor,     # (B, T, 1, 768)
    ):
        # Build teacher-forced action embeddings → (B, T, 2, D)
        action_in = torch.stack([
            self.dpad_embed(dpad),
            self.button_embed(button),
        ], dim=2)

        # Backbone → (B, T, 2, D)
        action_out = self.backbone(frames, action_in, text)

        # Project to logits
        dpad_logits   = self.dpad_head(action_out[:, :, 0, :])   # (B, T, N_DPAD)
        button_logits = self.button_head(action_out[:, :, 1, :]) # (B, T, N_BUTTONS)

        return dpad_logits, button_logits

    def loss(
        self,
        dpad_logits: torch.Tensor,    # (B, T, N_DPAD)
        button_logits: torch.Tensor,  # (B, T, N_BUTTONS)
        dpad_labels: torch.Tensor,    # (B, T)
        button_labels: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        dpad_ce   = F.cross_entropy(dpad_logits.flatten(0, 1),   dpad_labels.flatten())
        button_ce = F.cross_entropy(button_logits.flatten(0, 1), button_labels.flatten())
        # Normalise by max entropy so both terms are in [0, 1]
        return dpad_ce / math.log(N_DPAD) + button_ce / math.log(N_BUTTONS)
