"""
DPO training loop for the NES policy model.

Loads a pre-trained BC checkpoint as both the initial policy and the frozen
reference model, then fine-tunes with Direct Preference Optimization on
graph-generated (chosen, rejected) pairs.
"""

import argparse
import logging
import math
import os
import random

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from pixel2play.dpo_dataset import DPODataset
from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_actions import N_ACTIONS
from pixel2play.model.nes_policy import NESPolicyModel
from pixel2play.train import NESLightningModule, _load_config

CHECKPOINT_DIR = "tmp/checkpoints/nes_policy_dpo"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "nes_dpo.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(checkpoint_path: str, backbone_cfg: BackboneConfig) -> NESPolicyModel:
    """Load a NESPolicyModel from a Lightning checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    # torch.compile() adds '_orig_mod.' to keys — strip it if present
    state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}

    module = NESLightningModule(backbone_cfg, lr=1e-4)
    module.load_state_dict(state_dict, strict=True)
    return module.model


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DPOLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg: BackboneConfig,
        ref_model: NESPolicyModel,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        average_logprob: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "ref_model"])
        cfg.dropout = 0.0  # no dropout for DPO fine-tuning
        self.policy_model = NESPolicyModel(cfg)
        self.ref_model = ref_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.average_logprob = average_logprob

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def on_fit_start(self):
        self.policy_model.backbone._build_block_masks()
        self.ref_model.backbone._build_block_masks()

    def _compute_logprobs(
        self,
        model: NESPolicyModel,
        ram: torch.Tensor,     # (B, T, 2048)
        action: torch.Tensor,  # (B, T) int64
        mask: torch.Tensor,    # (B, T) bool
    ) -> torch.Tensor:
        """Compute summed (or averaged) log-probability of the taken actions.

        Returns shape (B,).
        """
        logits = model(ram, action)  # (B, T, N_ACTIONS)
        logprobs = F.log_softmax(logits, dim=-1)
        # Gather log-prob of the taken action at each step
        action_logprobs = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Mask and aggregate
        action_logprobs = action_logprobs * mask  # zero out invalid positions
        if self.average_logprob:
            # Average over valid tokens per sequence
            token_count = mask.sum(dim=-1).clamp(min=1)
            return action_logprobs.sum(dim=-1) / token_count
        else:
            return action_logprobs.sum(dim=-1)

    def _step(self, batch: dict):
        chosen_ram = batch["chosen_ram"]
        chosen_action = batch["chosen_action"]
        rejected_ram = batch["rejected_ram"]
        rejected_action = batch["rejected_action"]
        pivot = batch["pivot"]  # (B,)

        B, T = chosen_action.shape
        device = chosen_action.device
        arange_t = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        # Mask: only positions >= pivot are considered
        mask_chosen = arange_t >= pivot.unsqueeze(1)    # (B, T)
        mask_rejected = mask_chosen.clone()

        # If lengths are available, also mask beyond actual trace length
        if "chosen_len" in batch:
            mask_chosen = mask_chosen & (arange_t < batch["chosen_len"].unsqueeze(1))
        if "rejected_len" in batch:
            mask_rejected = mask_rejected & (arange_t < batch["rejected_len"].unsqueeze(1))

        # Guard against empty masks (shouldn't happen with reasonable data)
        if mask_chosen.sum() == 0 or mask_rejected.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {}, {}

        # Policy log-probs
        policy_chosen_logps = self._compute_logprobs(
            self.policy_model, chosen_ram, chosen_action, mask_chosen
        )
        policy_rejected_logps = self._compute_logprobs(
            self.policy_model, rejected_ram, rejected_action, mask_rejected
        )

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._compute_logprobs(
                self.ref_model, chosen_ram, chosen_action, mask_chosen
            )
            ref_rejected_logps = self._compute_logprobs(
                self.ref_model, rejected_ram, rejected_action, mask_rejected
            )

        # DPO loss
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = self.beta * (policy_logratios - ref_logratios)

        if self.label_smoothing > 0:
            loss = -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing
            loss = loss.mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            margins = chosen_rewards - rejected_rewards

        metrics = {
            "loss": loss,
            "rewards/chosen": chosen_rewards.mean(),
            "rewards/rejected": rejected_rewards.mean(),
            "rewards/margin": margins.mean(),
            "rewards/accuracy": accuracy,
            "logps/policy_chosen": policy_chosen_logps.mean(),
            "logps/policy_rejected": policy_rejected_logps.mean(),
            "logps/ref_chosen": ref_chosen_logps.mean(),
            "logps/ref_rejected": ref_rejected_logps.mean(),
        }
        return loss, metrics

    def training_step(self, batch: dict, _):
        loss, metrics = self._step(batch)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"))
        return loss

    def validation_step(self, batch: dict, _):
        loss, metrics = self._step(batch)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k == "loss"))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_epochs = self.trainer.max_epochs
        warmup_epochs = max(1, total_epochs // 20)
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=self.lr * 0.05
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to YAML config file")
    parser.add_argument("--exp_name", default=None, help="Experiment name (subdirectory for checkpoints)")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run a single batch for sanity checking")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile() — faster startup, slower throughput")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    shared = cfg["shared"]
    pm = cfg.get("policy_model", {})
    dpo_cfg = cfg.get("dpo", {})

    pl.seed_everything(dpo_cfg.get("seed", 42), workers=True)
    torch.set_float32_matmul_precision("high")

    n_steps = shared["n_seq_timesteps"]
    backbone_cfg = BackboneConfig(
        n_steps=n_steps,
        dim=pm.get("transformer_dim", 1024),
        n_layers=pm.get("n_transformer_layers", 10),
        n_q_heads=pm.get("n_q_head", 16),
        n_kv_heads=pm.get("n_kv_head", 16),
        n_thinking_tokens=pm.get("n_thinking_tokens", 1),
        mask_block_size=pm.get("mask_block_size", 128),
        attention_history_len=pm.get("attention_history_len", None),
        dropout=0.0,
    )

    ref_ckpt = dpo_cfg.get("ref_checkpoint")
    if ref_ckpt is None:
        raise ValueError("ref_checkpoint is required in the config file (path to the BC checkpoint)")

    # -----------------------------------------------------------------------
    # Load reference model (frozen) and initialize policy from same checkpoint
    # -----------------------------------------------------------------------
    logging.info(f"Loading reference model from {ref_ckpt}")
    ref_model = load_model_from_checkpoint(ref_ckpt, backbone_cfg)
    ref_model = ref_model.to(torch.bfloat16)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    data_folder = dpo_cfg.get("data_folder", "annotate/bc_data/level_all_ram_dpo")
    kind_filter = dpo_cfg.get("kind_filter", None)
    full_dataset = DPODataset(data_folder, kind_filter=kind_filter)
    val_fraction = dpo_cfg.get("val_fraction", 0.1)
    n_val = int(len(full_dataset) * val_fraction)
    n_train = len(full_dataset) - n_val
    seed = dpo_cfg.get("seed", 42)
    if n_val > 0:
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    logging.info(f"Train: {len(train_dataset)}  Val: {len(val_dataset) if val_dataset else 0}")

    batch_size = dpo_cfg.get("batch_size", 8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    module = DPOLightningModule(
        backbone_cfg,
        ref_model=ref_model,
        lr=dpo_cfg.get("lr", 1e-5),
        weight_decay=dpo_cfg.get("weight_decay", 1e-4),
        beta=dpo_cfg.get("beta", 0.1),
        label_smoothing=dpo_cfg.get("label_smoothing", 0.0),
        average_logprob=dpo_cfg.get("average_logprob", True),
    )
    # Initialize policy from the same checkpoint as reference
    logging.info("Initializing policy model from reference checkpoint")
    policy_state = load_model_from_checkpoint(ref_ckpt, backbone_cfg).state_dict()
    module.policy_model.load_state_dict(policy_state, strict=True)

    if not args.no_compile:
        module.policy_model = torch.compile(module.policy_model)
        # Note: ref_model stays un-compiled since it's frozen and only used in no_grad

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger

    checkpoint_dir = dpo_cfg.get("checkpoint_dir", CHECKPOINT_DIR)
    ckpt_dir = os.path.join(checkpoint_dir, args.exp_name) if args.exp_name else checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss" if val_dataset is not None else None,
        save_top_k=1,
        mode="min",
        enable_version_counter=False,
    )
    callbacks = [checkpoint_cb]

    if val_dataset is not None:
        early_stop_cb = EarlyStopping(
            monitor="val/loss",
            patience=3,
            mode="min",
        )
        callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=dpo_cfg.get("max_epochs", 5),
        limit_train_batches=dpo_cfg.get("limit_train_batches", None),
        limit_val_batches=dpo_cfg.get("limit_val_batches", None),
        check_val_every_n_epoch=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        default_root_dir="tmp",
        logger=TensorBoardLogger("tmp/lightning_logs", name=args.exp_name or "dpo_default", version=0),
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
