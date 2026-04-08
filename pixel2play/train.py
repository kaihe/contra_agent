import argparse
import logging
import os
import re
import random

import lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from pixel2play.dataset import NESDataset
from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_policy import NESPolicyModel

DATA_FOLDER = "annotate/bc_data/Contra-Nes"
CHECKPOINT_DIR = "tmp/checkpoints/nes_policy"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "nes_150M.yaml")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class NESLightningModule(pl.LightningModule):
    def __init__(self, cfg: BackboneConfig, lr: float = 1e-4, dropout: float = 0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        cfg.dropout = dropout
        self.model = NESPolicyModel(cfg)
        self.lr = lr

    def on_fit_start(self):
        # Block masks are built on CPU at __init__ time; rebuild on the actual device.
        self.model.backbone._build_block_masks()


    def _step(self, batch):
        frames, dpad, button, text, valid_mask = batch
        dpad_logits, button_logits = self.model(frames, dpad, button, text)
        return self.model.loss(dpad_logits, button_logits, dpad, button, valid_mask)

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


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
        full = NESDataset(self.data_root, n_steps=self.n_steps, n_text_tokens=self.n_text_tokens)

        # Group recordings by level, then pick 1 val recording per level (stratified).
        by_level: dict[str, list] = {}
        for rec in full.recordings:
            level = re.search(r"level\d+", os.path.basename(rec.path), re.IGNORECASE)
            key = level.group(0).lower() if level else "unknown"
            by_level.setdefault(key, []).append(rec)

        rng = random.Random(42)
        val_recs, train_recs = [], []
        n_val_per_level = 2
        for recs in by_level.values():
            shuffled = recs[:]
            rng.shuffle(shuffled)
            n_val = min(n_val_per_level, len(shuffled) - 1)
            val_recs.extend(shuffled[:n_val])
            train_recs.extend(shuffled[n_val:])

        self.train_set = NESDataset.from_recordings(train_recs, self.n_steps)
        self.val_set   = NESDataset.from_recordings(val_recs,   self.n_steps)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, force=True)

    cfg = _load_config(CONFIG_FILE)
    shared   = cfg["shared"]
    stage3   = cfg["stage3_finetune"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder",    default=DATA_FOLDER)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--fast_dev_run",   action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    n_steps    = shared["n_seq_timesteps"]
    batch_size = stage3["training_dataset"]["batch_size"]
    lr         = stage3["optim"]["learning_rate"]
    max_steps  = stage3["n_training_steps"]

    n_text_tokens = shared.get("text_tokenizer_config", {}).get("text_embedding_shape", [1, 768])[0]
    dropout       = cfg.get("policy_model", {}).get("dropout", 0.0)

    backbone_cfg = BackboneConfig(n_steps=n_steps, n_text_tokens=n_text_tokens, dropout=dropout)

    datamodule = NESDataModule(
        data_root=args.data_folder,
        n_steps=n_steps,
        batch_size=batch_size,
        n_text_tokens=n_text_tokens,
    )

    module = NESLightningModule(backbone_cfg, lr=lr, dropout=dropout)
    module.model = torch.compile(module.model)

    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        every_n_epochs=5,
        filename="ckpt-{epoch:04d}-{val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=max_steps,
        check_val_every_n_epoch=5,
        precision="bf16-mixed",
        callbacks=[checkpoint_cb],
        fast_dev_run=args.fast_dev_run,
        default_root_dir="tmp",
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
