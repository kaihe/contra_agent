"""
Train ContraVLA on pre-processed shard data.

Usage:
    python -m vla.train \
        --data_dir vla/data/level1_action2 \
        --out      vla/checkpoints/level1 \
        --epochs    30 \
        --batch_size 32 \
        --lr        1e-4
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from vla.datasets import ContraVLADataset, build_collate_fn
from vla.model import ContraVLA, ContraVLAConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_loader(shard_dir: str, collate_fn, batch_size: int,
                 workers: int, shuffle: bool) -> DataLoader:
    ds = ContraVLADataset(shard_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _to(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _count_params(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# Train / val loops
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, scheduler, scaler,
                 device, grad_clip: float, epoch: int, writer: SummaryWriter | None,
                 global_step: int) -> tuple[float, int, float]:
    model.train()
    total_loss, n_steps, total_tokens = 0.0, 0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch = _to(batch, device)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out  = model(input_ids=batch["input_ids"], images=batch["images"],
                         proprio=batch["proprio"],    actions=batch["actions"])
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        total_loss   += loss.item()
        total_tokens += batch["input_ids"].numel() + batch["actions"].numel()
        n_steps      += 1
        global_step  += 1

        if writer is not None:
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)

        if (step + 1) % 50 == 0:
            elapsed    = time.time() - t0
            tok_s      = total_tokens / max(elapsed, 1e-6)
            clip_ratio = grad_norm / grad_clip
            print(f"  epoch {epoch}  step {step+1}/{len(loader)}"
                  f"  loss {total_loss/n_steps:.4f}"
                  f"  lr {scheduler.get_last_lr()[0]:.2e}"
                  f"  grad {grad_norm:.2f} ({clip_ratio:.2f}x)"
                  f"  {tok_s/1e3:.1f}k tok/s"
                  f"  {elapsed:.0f}s")

    elapsed = time.time() - t0
    return total_loss / max(n_steps, 1), global_step, total_tokens / max(elapsed, 1e-6)


@torch.no_grad()
def _val_epoch(model, loader, device, epoch: int) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total, n_steps = 0.0, 0, 0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch  = _to(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out    = model(input_ids=batch["input_ids"], images=batch["images"],
                           proprio=batch["proprio"],    actions=batch["actions"])
        total_loss += out["loss"].item()
        preds      = out["logits"].argmax(-1)   # [B, T]
        correct   += (preds == batch["actions"]).sum().item()
        total     += batch["actions"].numel()
        n_steps   += 1

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  val epoch {epoch}  step {step+1}/{len(loader)}"
                  f"  loss {total_loss/n_steps:.4f}"
                  f"  acc {correct/max(total,1)*100:.1f}%"
                  f"  {elapsed:.0f}s")

    return total_loss / max(n_steps, 1), correct / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # ── model ──
    config = ContraVLAConfig(dropout=args.dropout)
    model  = ContraVLA(config).to(device)
    if args.freeze_vlm:
        model.freeze_vlm()
        print("VLM frozen — training action transformer only")
    if args.grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    if getattr(args, "compile", False):
        model.action_transformer = torch.compile(model.action_transformer, dynamic=False)
        print("torch.compile applied to action transformer")

    total, trainable = _count_params(model)
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # ── tokenizer + data ──
    tokenizer  = AutoTokenizer.from_pretrained(config.vlm_model_name)
    collate_fn = build_collate_fn(tokenizer)
    train_loader = _make_loader(os.path.join(args.data_dir, "train"), collate_fn,
                                args.batch_size, args.workers, shuffle=True)
    val_loader   = _make_loader(os.path.join(args.data_dir, "val"),   collate_fn,
                                args.batch_size, args.workers, shuffle=False)
    print(f"Train: {len(train_loader.dataset)} samples  "
          f"Val: {len(val_loader.dataset)} samples")

    # ── optimiser + schedule ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = min(args.warmup_steps, total_steps // 10)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ── tensorboard ──
    writer = SummaryWriter(args.log_dir)

    # ── training loop ──
    best_val_loss = float("inf")
    global_step = 0
    epoch_times: list[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        train_loss, global_step, tok_per_sec = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, args.grad_clip, epoch, writer, global_step,
        )
        val_loss, val_acc = _val_epoch(model, val_loader, device, epoch)

        epoch_elapsed = time.time() - epoch_t0
        epoch_times.append(epoch_elapsed)
        remaining = args.epochs - epoch
        eta_s = remaining * (sum(epoch_times) / len(epoch_times))
        eta_str = f"{eta_s/3600:.1f}h" if eta_s >= 3600 else f"{eta_s/60:.0f}m"

        print(f"[epoch {epoch:3d}/{args.epochs}]  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  acc {val_acc*100:.1f}%  "
              f"{tok_per_sec/1e3:.1f}k tok/s  "
              f"eta {eta_str}")

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)

        ckpt = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_loss,
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  ↳ best val loss {best_val_loss:.4f} — saved")

    writer.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default=None, help="Path to YAML config file")
    p.add_argument("--name",         default=None, help="Experiment name (sets out and log_dir)")
    p.add_argument("--data_dir",     default="vla/data/level1_action2")
    p.add_argument("--out",          default=None)
    p.add_argument("--log_dir",      default=None)
    p.add_argument("--epochs",       type=int,   default=4)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--freeze_vlm",   action="store_true", default=True)
    p.add_argument("--no_freeze_vlm", dest="freeze_vlm", action="store_false")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--dropout",       type=float, default=0.1,
                   help="Dropout rate for action transformer attention and MLP")
    p.add_argument("--grad_checkpoint", action="store_true", default=False,
                   help="Enable gradient checkpointing to save memory")
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile the action transformer (first batch is slow)")

    # First pass: parse --config only so we can load YAML defaults
    partial_args, _ = p.parse_known_args()
    if partial_args.config is not None and os.path.isfile(partial_args.config):
        with open(partial_args.config) as f:
            yaml_defaults = yaml.safe_load(f)
        if yaml_defaults:
            p.set_defaults(**yaml_defaults)

    args = p.parse_args()

    # Resolve output directories based on --name if not explicitly provided
    if args.out is None:
        args.out = f"tmp/vla/{args.name}" if args.name else "tmp/vla"
    if args.log_dir is None:
        args.log_dir = f"tmp/tf_logs/{args.name}" if args.name else "tmp/tensorboard_logs"

    train(args)


if __name__ == "__main__":
    main()
