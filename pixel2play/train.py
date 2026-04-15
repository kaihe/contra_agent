import argparse
import logging
import os
import re
import random
import time

import numpy as np

import lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from pixel2play.dataset import NESDataset
from pixel2play.model.backbone import BackboneConfig
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
        ram, dpad, button, text, valid_mask = batch
        dpad_logits, button_logits = self.model(ram, dpad, button, text)
        return self.model.loss(dpad_logits, button_logits, dpad, button, valid_mask)

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
    def __init__(self, data_root: str, n_steps: int, batch_size: int, val_fraction: float = 0.1, n_text_tokens: int = 1):
        super().__init__()
        self.data_root = data_root
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.n_text_tokens = n_text_tokens

    def setup(self, stage=None):
        self._epoch_counter = 0
        full = NESDataset(self.data_root, n_steps=self.n_steps, n_text_tokens=self.n_text_tokens)

        # Group recordings by level, then pick 1 val recording per level (stratified).
        by_level: dict[str, list] = {}
        for rec in full.recordings:
            level = re.search(r"level\d+", os.path.basename(rec.path), re.IGNORECASE)
            key = level.group(0).lower() if level else "unknown"
            by_level.setdefault(key, []).append(rec)

        rng = random.Random(42)
        val_recs, train_recs = [], []
        for level_key, recs in by_level.items():
            shuffled = recs[:]
            rng.shuffle(shuffled)
            if level_key == "level1":
                n_val = min(10, len(shuffled) - 1)
            else:
                n_val = 0
            val_recs.extend(shuffled[:n_val])
            train_recs.extend(shuffled[n_val:])

        self.train_set = NESDataset.from_recordings(train_recs, self.n_steps)
        self.val_set   = NESDataset.from_recordings(val_recs,   self.n_steps)

    def on_train_epoch_start(self):
        rng = np.random.RandomState(self._epoch_counter)
        self.train_set.resample(rng)
        self._epoch_counter += 1

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


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
    parser.add_argument("--data_folder",    default=stage3["training_dataset"]["data_folder"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--exp_name",       default=None, help="Experiment name, used as checkpoint subdirectory")
    parser.add_argument("--fast_dev_run",   action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    n_steps                = shared["n_seq_timesteps"]
    batch_size             = stage3["training_dataset"]["batch_size"]
    lr                     = stage3["optim"]["learning_rate"]
    max_epochs             = stage3.get("n_epochs", 5)
    accumulate_grad_batches = stage3.get("accumulate_grad_batches", 1)

    n_text_tokens = shared.get("text_tokenizer_config", {}).get("text_embedding_shape", [1, 768])[0]

    pm      = cfg.get("policy_model", {})
    dec     = pm.get("action_decoder", {})
    dropout = pm.get("dropout", 0.0)
    backbone_cfg = BackboneConfig(
        n_steps=n_steps,
        n_text_tokens=n_text_tokens,
        dim=pm.get("transformer_dim", 1024),
        n_layers=pm.get("n_transformer_layers", 10),
        n_q_heads=pm.get("n_q_head", 16),
        n_kv_heads=pm.get("n_kv_head", 16),
        n_thinking_tokens=pm.get("n_thinking_tokens", 1),
        mask_block_size=pm.get("mask_block_size", 128),
        attention_history_len=pm.get("attention_history_len", None),
        dropout=pm.get("dropout", 0.0),
        dec_n_layers=dec.get("n_layers", 3),
        dec_n_heads=dec.get("n_heads", 8),
    )

    datamodule = NESDataModule(
        data_root=args.data_folder,
        n_steps=n_steps,
        batch_size=batch_size,
        n_text_tokens=n_text_tokens,
    )

    weight_decay = stage3["optim"]["weight_decay"]
    module = NESLightningModule(backbone_cfg, lr=lr, dropout=dropout, weight_decay=weight_decay)
    module.model = torch.compile(module.model)

    from lightning.pytorch.callbacks import ModelCheckpoint, ThroughputMonitor

    ckpt_dir = os.path.join(args.checkpoint_dir, args.exp_name) if args.exp_name else args.checkpoint_dir
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_top_k=1,
        enable_version_counter=False,
    )
    # batch: (ram, dpad, button, text, valid_mask); ram shape: (B, T, 2048)
    throughput_cb = ThroughputMonitor(
        batch_size_fn=lambda batch: batch[0].shape[0],
        length_fn=lambda batch: batch[0].shape[0] * batch[0].shape[1],
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=1,
        precision="bf16-mixed",
        callbacks=[checkpoint_cb, throughput_cb],
        fast_dev_run=args.fast_dev_run,
        default_root_dir="tmp",
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
