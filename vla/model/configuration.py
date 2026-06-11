"""Configuration for the Contra VLA model."""

from __future__ import annotations

from transformers import PretrainedConfig


class ContraVLAConfig(PretrainedConfig):
    """Small config object matching the behavior-cloning contract in vla/readme.md."""

    model_type = "contravla"

    def __init__(
        self,
        vlm_model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_dim: int = 36,
        num_actions: int = 1,
        proprio_dim: int = 118,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vlm_model_name = vlm_model_name
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.proprio_dim = proprio_dim
