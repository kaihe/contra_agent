import argparse
import logging
import os

import random
import time

import numpy as np

import lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from pixel2play.dataset import NESDataset
from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_actions import N_BUTTONS
from pixel2play.model.nes_policy import NESPolicyModel

CHECKPOINT_DIR = "tmp/checkpoints/nes_policy"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "nes_10M.yaml")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class NESLightningModule(pl.LightningModule):
    def __init__(self, cfg: BackboneConfig, lr: float = 1e-4, dropout: float = 0.0, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        cfg.dropout = dropout
        self.model = NESPolicyModel(cfg)
        self.lr = lr
        self.weight_decay = weight_decay

    def on_fit_start(self):
        # Block masks are built on CPU at __init__ time; rebuild on the actual device.
        self.model.backbone._build_block_masks()
        self._epoch_end_time = None

    def on_train_epoch_end(self):
        self._epoch_end_time = time.perf_counter()

    def on_train_epoch_start(self):
        if self._epoch_end_time is not None:
            gap = time.perf_counter() - self._epoch_end_time
            logging.info(f"[timing] inter-epoch gap: {gap:.2f}s")


    def _step(self, batch):
        ram, dpad, button, valid_mask = batch
        action = dpad * N_BUTTONS + button
        action_logits = self.model(ram, action)
        return self.model.loss(action_logits, action, valid_mask)

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_epochs  = self.trainer.max_epochs
        warmup_epochs = max(1, total_epochs // 20)   # 5% warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=self.lr * 0.05
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Lightning data module
# ---------------------------------------------------------------------------

class NESDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, n_steps: int, batch_size: int, n_val_recordings: int = 50,
                 max_train_recordings: int = 0):
        super().__init__()
        self.data_root = data_root
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_val_recordings = n_val_recordings
        self.max_train_recordings = max_train_recordings  # 0 = use all

    def setup(self, stage=None):
        self._epoch_counter = 0
        full = NESDataset(self.data_root, n_steps=self.n_steps)

        rng = random.Random(42)
        recs = full.recordings[:]
        rng.shuffle(recs)
        n_val = min(self.n_val_recordings, len(recs) - 1) if self.n_val_recordings > 0 else 0
        val_recs   = recs[:n_val]
        train_recs = recs[n_val:]

        if self.max_train_recordings > 0:
            train_recs = train_recs[:self.max_train_recordings]
        print(f"Dataset: {len(train_recs)} train recordings, {len(val_recs)} val recordings")

        self.train_set = NESDataset.from_recordings(train_recs, self.n_steps)
        self.val_set   = NESDataset.from_recordings(val_recs,   self.n_steps)

    def on_train_epoch_start(self):
        rng = np.random.RandomState(self._epoch_counter)
        self.train_set.resample(rng)
        self._epoch_counter += 1

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, force=True)

    # Pre-parse --config so all other defaults can come from the chosen YAML.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=CONFIG_FILE)
    pre_args, _ = pre.parse_known_args()

    cfg    = _load_config(pre_args.config)
    shared = cfg["shared"]
    stage3 = cfg["stage3_finetune"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default=pre_args.config, help="Path to YAML config file")
    parser.add_argument("--exp_name",       default=None, help="Experiment name, used as checkpoint subdirectory")
    parser.add_argument("--fast_dev_run",            action="store_true")
    parser.add_argument("--no_compile",              action="store_true",
                        help="Skip torch.compile() — faster startup, slower throughput. Useful for quick overfit experiments.")
    parser.add_argument("--max_train_recordings",    type=int, default=0,
                        help="Limit training set size (0 = use all). Use a small number to find the overfit point.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    n_steps                = shared["n_seq_timesteps"]
    batch_size             = stage3["training_dataset"]["batch_size"]
    lr                     = stage3["optim"]["learning_rate"]
    max_epochs             = stage3.get("n_epochs", 5)
    accumulate_grad_batches = stage3.get("accumulate_grad_batches", 1)

    pm      = cfg.get("policy_model", {})
    dropout = pm.get("dropout", 0.0)
    backbone_cfg = BackboneConfig(
        n_steps=n_steps,
        dim=pm.get("transformer_dim", 1024),
        n_layers=pm.get("n_transformer_layers", 10),
        n_q_heads=pm.get("n_q_head", 16),
        n_kv_heads=pm.get("n_kv_head", 16),
        n_thinking_tokens=pm.get("n_thinking_tokens", 1),
        mask_block_size=pm.get("mask_block_size", 128),
        attention_history_len=pm.get("attention_history_len", None),
        dropout=pm.get("dropout", 0.0),
    )

    data_folder = stage3["training_dataset"]["data_folder"]
    checkpoint_dir = stage3.get("checkpoint_dir", CHECKPOINT_DIR)

    datamodule = NESDataModule(
        data_root=data_folder,
        n_steps=n_steps,
        batch_size=batch_size,
        n_val_recordings=stage3.get("n_val_recordings", 50),
        max_train_recordings=args.max_train_recordings,
    )

    weight_decay = stage3["optim"]["weight_decay"]
    module = NESLightningModule(backbone_cfg, lr=lr, dropout=dropout, weight_decay=weight_decay)
    if not args.no_compile:
        module.model = torch.compile(module.model)

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ThroughputMonitor
    from lightning.pytorch.loggers import TensorBoardLogger

    ckpt_dir = os.path.join(checkpoint_dir, args.exp_name) if args.exp_name else checkpoint_dir
    has_val = datamodule.n_val_recordings > 0
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss" if has_val else None,
        save_top_k=1,
        mode="min",
        enable_version_counter=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=stage3.get("early_stop_patience", 5),
        mode="min",
    )
    # batch: (ram, dpad, button, valid_mask); ram shape: (B, T, 2048)
    throughput_cb = ThroughputMonitor(
        batch_size_fn=lambda batch: batch[0].shape[0],
        length_fn=lambda batch: batch[0].shape[0] * batch[0].shape[1],
    )

    callbacks = [checkpoint_cb, throughput_cb]
    if has_val:
        callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        default_root_dir="tmp",
        logger=TensorBoardLogger("tmp/lightning_logs", name=args.exp_name or "default", version=0),
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
