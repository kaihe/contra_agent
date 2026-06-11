"""ContraVLA model: SmolVLM features plus a causal action transformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, PreTrainedModel

from .action_transformer import CausalActionTransformer
from .configuration import ContraVLAConfig


def _first_attr(obj, names: tuple[str, ...]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of: {', '.join(names)}")


class ContraVLA(PreTrainedModel):
    """Vision-language-action next-action classifier for NES Contra."""

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

    def _vision_backbone(self):
        model = self.vlm.model
        return _first_attr(model, ("vision_model",))

    def _image_projector(self):
        model = self.vlm.model
        return _first_attr(model, ("connector", "multi_modal_projector"))

    def _language_model(self):
        model = self.vlm.model
        return _first_attr(model, ("text_model", "language_model"))

    def _text_attention_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            return attention_mask.to(device=input_ids.device, dtype=torch.bool)

        pad_token_id = getattr(self.vlm.config, "pad_token_id", None)
        text_config = getattr(self.vlm.config, "text_config", None)
        if pad_token_id is None and text_config is not None:
            pad_token_id = getattr(text_config, "pad_token_id", None)
        if pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids.ne(pad_token_id)

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode images and text as one VLM token sequence.

        Args:
            pixel_values: [B, V, 3, H, W]. The BC dataset uses V=1; rollout
                wrappers may pass a longer frame window.
            input_ids: [B, L] tokenized goal text.
            attention_mask: optional text mask [B, L], where 1/True is valid.
            return_attention_mask: return the fused image/text valid-token mask
                alongside hidden states.

        Returns:
            VLM hidden states [B, T, D_lm], plus a matching valid-token mask
            [B, T] when ``return_attention_mask`` is true.
        """
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B, V, C, H, W], got {tuple(pixel_values.shape)}")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask must be {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}"
            )

        batch_size, n_frames, channels, height, width = pixel_values.shape
        flat_images = pixel_values.reshape(batch_size * n_frames, channels, height, width)

        vision_out = self._vision_backbone()(pixel_values=flat_images, return_dict=True)
        image_features = self._image_projector()(vision_out.last_hidden_state)
        _, n_image_tokens, hidden = image_features.shape
        image_features = image_features.reshape(
            batch_size, n_frames * n_image_tokens, hidden
        )

        lm = self._language_model()
        text_embeds = lm.get_input_embeddings()(input_ids)
        image_features = image_features.to(
            device=text_embeds.device, dtype=text_embeds.dtype
        )
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)
        image_attention_mask = torch.ones(
            batch_size,
            image_features.size(1),
            dtype=torch.bool,
            device=text_embeds.device,
        )
        text_attention_mask = self._text_attention_mask(input_ids, attention_mask)
        vlm_attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)

        lm_out = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=vlm_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        if return_attention_mask:
            return lm_out.last_hidden_state, vlm_attention_mask
        return lm_out.last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        proprio: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        actions: torch.LongTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        vlm_features, vlm_attention_mask = self.forward_vlm_efficient(
            images,
            input_ids,
            attention_mask=attention_mask,
            return_attention_mask=True,
        )
        logits = self.action_transformer(
            vlm_features, proprio, vlm_attention_mask=vlm_attention_mask
        )

        out = {"logits": logits}
        if actions is not None:
            if actions.ndim == 1:
                actions = actions.unsqueeze(1)
            if actions.shape != logits.shape[:2]:
                raise ValueError(
                    f"actions must be {tuple(logits.shape[:2])}, got {tuple(actions.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        proprio: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Return greedy action ids [B, num_actions]."""
        return self(
            input_ids=input_ids,
            images=images,
            proprio=proprio,
            attention_mask=attention_mask,
        )["logits"].argmax(-1)

    def freeze_vlm(self) -> None:
        for param in self.vlm.parameters():
            param.requires_grad_(False)

    def unfreeze_vlm(self) -> None:
        for param in self.vlm.parameters():
            param.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
