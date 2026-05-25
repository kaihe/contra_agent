"""ContraVLA: Vision-Language-Action model for NES Contra."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForImageTextToText

from .configuration import ContraVLAConfig
from .action_transformer import CausalActionTransformer


class ContraVLA(PreTrainedModel):
    """
    SmolVLM backbone + causal action transformer for discrete action chunk prediction.

    Forward pass (training):
        images + text → VLM features → action transformer → cross-entropy over 36 classes

    Inference:
        model.generate_actions(...) → action indices [B, T]  (argmax)
    """

    config_class = ContraVLAConfig
    base_model_prefix = "contravla"
    supports_gradient_checkpointing = True

    def __init__(self, config: ContraVLAConfig) -> None:
        super().__init__(config)

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        vlm_hidden_size = self.vlm.config.text_config.hidden_size

        self.action_transformer = CausalActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            action_dim=config.action_dim,
            num_actions=config.num_actions,
            proprio_dim=config.proprio_dim,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # VLM encoding
    # ------------------------------------------------------------------

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,  # [B, V, C, H, W]  V=2 frames, always valid
        input_ids: torch.LongTensor,      # [B, L]
    ) -> torch.Tensor:
        """
        Efficient training forward: vision encoder → connector → LM in one shot.
        Returns fused VLM features [B, T_enc, D].
        """
        B, V, C, H, W = pixel_values.shape
        device, dtype = pixel_values.device, pixel_values.dtype

        # ---- vision encoder: encode all V frames at once ----
        flat_images = pixel_values.flatten(0, 1)            # [B*V, C, H, W]

        vis_out   = self.vlm.model.vision_model(pixel_values=flat_images, return_dict=True)
        img_feats = vis_out.last_hidden_state                # [B*V, n_patches, D_vis]

        connector = getattr(self.vlm.model, "connector", None) \
                    or self.vlm.model.multi_modal_projector
        img_feats = connector(img_feats)                     # [B*V, n_patches, D_lm]

        # reshape and flatten frames into one token sequence per sample
        _, n_patches, D = img_feats.shape
        img_feats = img_feats.view(B, V * n_patches, D)      # [B, V*n_patches, D]

        # ---- text embeddings ----
        lm          = getattr(self.vlm.model, "text_model", None) or self.vlm.model.language_model
        text_embeds = lm.get_input_embeddings()(input_ids)   # [B, L, D]

        # ---- fuse: [img_tokens, text_tokens] → LM ----
        inputs_embeds = torch.cat([img_feats, text_embeds], dim=1)  # [B, V*n_patches+L, D]

        lm_out = lm(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        return lm_out.last_hidden_state  # [B, T_enc, D]

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,       # [B, L]
        images: torch.FloatTensor,         # [B, V, C, H, W]
        proprio: torch.FloatTensor,        # [B, 118]
        actions: torch.LongTensor,         # [B, T]  class indices 0..35
    ) -> dict[str, torch.Tensor]:
        vlm_features = self.forward_vlm_efficient(images, input_ids)
        logits = self.action_transformer(vlm_features, proprio)  # [B, T, 36]
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), actions.reshape(B * T))
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        proprio: torch.FloatTensor,
    ) -> torch.LongTensor:
        """Returns greedy action indices [B, T] in range [0, 35]."""
        vlm_features = self.forward_vlm_efficient(images, input_ids)
        logits = self.action_transformer(vlm_features, proprio)
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def freeze_vlm(self) -> None:
        for p in self.vlm.parameters():
            p.requires_grad_(False)

    def unfreeze_vlm(self) -> None:
        for p in self.vlm.parameters():
            p.requires_grad_(True)
