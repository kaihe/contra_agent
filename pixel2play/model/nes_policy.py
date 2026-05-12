"""
NESPolicyModel: backbone + single combined action head.

Input  (per step):
  obs    (B, T, 2048)        – NES RAM snapshot, OR
         (B, T, C, H, W)     – frame stack when use_vision=True
  action (B, T)              – ground-truth combined action class (teacher forcing)
                               action = dpad * N_BUTTONS + button  ∈ [0, N_ACTIONS)

Output:
  action_logits (B, T, N_ACTIONS) – 36 class scores
  loss          scalar             – normalised cross-entropy
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel2play.model.backbone import BackboneConfig, PolicyCausalTransformer
from pixel2play.model.nes_actions import N_ACTIONS


class NESPolicyModel(nn.Module):
    def __init__(
        self,
        cfg: BackboneConfig,
        focal_gamma: float = 0.0,
        focal_alpha: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        D = cfg.dim
        self.backbone = PolicyCausalTransformer(cfg)

        # Single combined action embedding (teacher forcing input)
        self.action_embed = nn.Embedding(N_ACTIONS, D, dtype=torch.bfloat16)
        nn.init.normal_(self.action_embed.weight, std=0.1)

        # Single output head
        self.action_head = nn.Linear(D, N_ACTIONS)

        # Loss hyperparameters
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha  # (N_ACTIONS,) tensor or None
        self.class_weights = class_weights  # (N_ACTIONS,) tensor or None

    def _action_in(self, action: torch.Tensor) -> torch.Tensor:
        # (B, T) → (B, T, 1, D)
        return self.action_embed(action).unsqueeze(2)

    def encode(
        self,
        obs: torch.Tensor,    # (B, T, 2048) or (B, T, C, H, W)
        action: torch.Tensor, # (B, T)  int64
    ) -> torch.Tensor:
        """Run the backbone transformer only. Returns action_out_tokens (B, T, D)."""
        return self.backbone._encode(obs, self._action_in(action))

    def forward(
        self,
        obs: torch.Tensor,    # (B, T, 2048) or (B, T, C, H, W)
        action: torch.Tensor, # (B, T)  int64
    ) -> torch.Tensor:
        """Returns action_logits (B, T, N_ACTIONS)."""
        action_out = self.backbone._encode(obs, self._action_in(action))
        return self.action_head(action_out)

    def loss(
        self,
        action_logits: torch.Tensor,  # (B, T, N_ACTIONS)
        action_labels: torch.Tensor,  # (B, T)
        valid_mask: torch.Tensor,     # (B, T) bool
    ) -> torch.Tensor:
        mask = valid_mask.flatten()
        weight = self.class_weights.to(action_logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(
            action_logits.flatten(0, 1), action_labels.flatten(), reduction="none", weight=weight
        )

        if self.focal_gamma > 0.0:
            pt = torch.exp(-ce)                       # p_t for the ground-truth class
            focal_weight = (1.0 - pt) ** self.focal_gamma
            if self.focal_alpha is not None:
                alpha_t = self.focal_alpha.to(ce.device)[action_labels.flatten()]
                focal_weight = focal_weight * alpha_t
            ce = focal_weight * ce

        return ce[mask].mean() / math.log(N_ACTIONS)
