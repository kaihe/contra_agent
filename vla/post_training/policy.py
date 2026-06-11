"""Policy helpers for ContraVLA post-training."""

from __future__ import annotations

import torch
from torch.distributions import Categorical
from transformers import AutoTokenizer

from vla.datasets.dataset import LEVEL_TEXTS
from vla.model import ContraVLA, ContraVLAConfig


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("._orig_mod.", "."): value for key, value in state_dict.items()}


def load_bc_policy(
    checkpoint_path: str,
    device: torch.device,
    *,
    dropout: float = 0.0,
    freeze_vlm: bool = True,
) -> ContraVLA:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ContraVLA(ContraVLAConfig(dropout=dropout))
    model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=True)
    if freeze_vlm:
        model.freeze_vlm()
    return model.to(device)


class VLAPolicy:
    def __init__(
        self,
        model: ContraVLA,
        device: torch.device,
        *,
        level_id: int = 0,
        max_text_len: int = 32,
    ) -> None:
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model.config.vlm_model_name)
        enc = self.tokenizer(
            LEVEL_TEXTS[level_id],
            return_tensors="pt",
            padding="max_length",
            max_length=max_text_len,
            truncation=True,
        )
        self.input_ids = enc["input_ids"].to(device)
        self.attention_mask = enc["attention_mask"].to(device)

    def batch_tensors(self, obs_batch: list[dict]) -> dict[str, torch.Tensor]:
        batch_size = len(obs_batch)
        return {
            "input_ids": self.input_ids.expand(batch_size, -1),
            "attention_mask": self.attention_mask.expand(batch_size, -1),
            "images": torch.stack([obs["images"] for obs in obs_batch]).to(self.device),
            "proprio": torch.stack([obs["proprio"] for obs in obs_batch]).to(self.device),
        }

    @torch.no_grad()
    def sample_batch(
        self,
        obs_batch: list[dict],
        *,
        temperature: float = 1.0,
    ) -> tuple[list[int], list[float]]:
        self.model.eval()
        batch = self.batch_tensors(obs_batch)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(**batch)["logits"][:, 0]
        behavior_dist = Categorical(logits=logits / max(temperature, 1e-6))
        actions = behavior_dist.sample()
        # GRPO/PPO ratios are computed against the raw policy distribution.
        # Temperature is only a rollout exploration device.
        policy_dist = Categorical(logits=logits)
        logprobs = policy_dist.log_prob(actions)
        return actions.cpu().tolist(), logprobs.float().cpu().tolist()

    def logits_for(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = images.size(0)
        return self.model(
            input_ids=self.input_ids.expand(batch_size, -1),
            attention_mask=self.attention_mask.expand(batch_size, -1),
            images=images.to(self.device),
            proprio=proprio.to(self.device),
        )["logits"][:, 0]
