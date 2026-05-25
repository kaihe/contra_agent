from transformers import PretrainedConfig


class ContraVLAConfig(PretrainedConfig):
    model_type = "contravla"

    def __init__(
        self,
        vlm_model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        # action transformer
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # action space
        action_dim: int = 36,   # 9 D-pad × 4 button combos
        num_actions: int = 2,   # chunk size T
        # structured state
        proprio_dim: int = 118, # contra.game_state.STATE_DIM
        **kwargs,
    ):
        self.vlm_model_name = vlm_model_name
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.proprio_dim = proprio_dim
        super().__init__(**kwargs)
