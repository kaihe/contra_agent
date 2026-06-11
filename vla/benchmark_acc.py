"""
vla/benchmark_acc.py — Offline button accuracy for a trained ContraVLA checkpoint.

Loads random samples from the val split and compares model predictions (greedy
argmax) against ground-truth actions from the BC dataset.  Reports per-timestep
jump / fire accuracy, precision, recall and F1, plus a focused confusion matrix
for the top validation actions.

Usage:
    python -m vla.benchmark_acc \
        --config     vla/behavior_clone.yaml \
        --n_samples  1000
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import torch
import yaml
from transformers import AutoTokenizer

from vla.datasets.dataset import LEVEL_TEXTS, ContraVLADataset
from vla.model import ContraVLA, ContraVLAConfig

DEFAULT_CONFIG = "vla/behavior_clone.yaml"


# ── Config loading ────────────────────────────────────────────────────────────

def _load_yaml_defaults(config_path: str) -> dict:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        defaults = yaml.safe_load(f) or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    # CLI aliases such as --no_freeze_vlm are parser actions, not benchmark args.
    defaults.pop("no_freeze_vlm", None)
    return defaults


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device) -> ContraVLA:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]
    sd   = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    cfg  = ContraVLAConfig(dropout=0.0)
    mdl  = ContraVLA(cfg)
    mdl.load_state_dict(sd, strict=True)
    return mdl.to(device).eval()


def _make_input_ids(model: ContraVLA, level_id: int, device: torch.device) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained(model.config.vlm_model_name)
    enc = tok(LEVEL_TEXTS[level_id], return_tensors="pt",
              padding="max_length", max_length=32, truncation=True)
    return enc["input_ids"].to(device)  # [1, L]


def _has_jump(action: int) -> bool:
    return (action % 4) in (1, 3)

def _has_fire(action: int) -> bool:
    return (action % 4) in (2, 3)


def _action_label(action: int) -> str:
    dpad_labels = (
        "neutral",
        "left",
        "right",
        "up",
        "down",
        "up-left",
        "up-right",
        "down-left",
        "down-right",
    )
    button_labels = ("none", "A", "B", "A+B")
    dpad_id, button_id = divmod(action, len(button_labels))
    if 0 <= dpad_id < len(dpad_labels) and 0 <= button_id < len(button_labels):
        return f"{action}:{dpad_labels[dpad_id]}+{button_labels[button_id]}"
    return str(action)


def _print_top_action_confusion(confusion: torch.Tensor, top_k: int) -> None:
    support = confusion.sum(dim=1)
    active = torch.nonzero(support > 0, as_tuple=False).flatten()
    if active.numel() == 0:
        print("\nTop action confusion matrix: no actions observed")
        return

    top_k = min(top_k, active.numel())
    top_actions = torch.topk(support, k=top_k).indices.tolist()
    top_set = set(top_actions)

    print(f"\nTop {top_k} action confusion matrix")
    print("Rows are ground truth; columns are predictions. Counts only.")
    print("Top actions by validation support:")
    for action in top_actions:
        correct = int(confusion[action, action].item())
        total = int(support[action].item())
        acc = correct / max(total, 1)
        print(f"  {_action_label(action):18s}  support={total:5d}  acc={acc:.3f}")

    col_width = max(8, max(len(str(a)) for a in top_actions) + 2)
    label_width = 18
    axis_label = "gt \\ pred"
    header = f"{axis_label:>{label_width}}" + "".join(
        f"{action:>{col_width}d}" for action in top_actions
    ) + f"{'other':>{col_width}}"
    print(header)
    print("-" * len(header))

    for gt in top_actions:
        row_counts = [int(confusion[gt, pred].item()) for pred in top_actions]
        other = int(sum(confusion[gt, pred].item() for pred in range(confusion.size(1)) if pred not in top_set))
        row = f"{_action_label(gt):>{label_width}}" + "".join(
            f"{count:>{col_width}d}" for count in row_counts
        ) + f"{other:>{col_width}d}"
        print(row)


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"checkpoint : {args.checkpoint}  device={device}")

    model     = _load_model(args.checkpoint, device)
    input_ids = _make_input_ids(model, level_id=0, device=device)

    val_dir = os.path.join(args.data_dir, "val")
    print(f"val dir    : {val_dir}")
    dataset = ContraVLADataset(val_dir, level_id=0)
    n_total = len(dataset)
    n       = min(args.n_samples, n_total)
    print(f"samples    : {n} / {n_total}\n")

    indices = random.sample(range(n_total), n)

    stats: dict[int, dict] = {t: defaultdict(int) for t in range(model.config.num_actions)}
    confusion = torch.zeros(
        model.config.action_dim,
        model.config.action_dim,
        dtype=torch.long,
    )
    overall_correct = 0
    overall_total   = 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample  = dataset[idx]
            images  = sample["images"].unsqueeze(0).to(device)   # [1, 2, 3, H, W]
            proprio = sample["proprio"].unsqueeze(0).to(device)  # [1, 118]
            gt_acts = sample["actions"]                           # [T] int64

            preds = model.generate_actions(input_ids, images, proprio)[0].cpu()  # [T]

            for t in range(model.config.num_actions):
                gt, pr = int(gt_acts[t]), int(preds[t])
                confusion[gt, pr] += 1

                for btn, fn in (("jump", _has_jump), ("fire", _has_fire)):
                    g, p = fn(gt), fn(pr)
                    if g and p:
                        stats[t][f"tp_{btn}"] += 1
                    elif g and not p:
                        stats[t][f"fn_{btn}"] += 1
                    elif not g and p:
                        stats[t][f"fp_{btn}"] += 1
                    else:
                        stats[t][f"tn_{btn}"] += 1

                overall_correct += int(gt == pr)
                overall_total   += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{n}]", flush=True)

    print("\n── Backtest Results ─────────────────────────────────────────────")
    print(f"Exact action accuracy: {overall_correct}/{overall_total}"
          f"  ({100*overall_correct/max(overall_total,1):.1f}%)\n")

    for t in range(model.config.num_actions):
        s = stats[t]
        print(f"  Action chunk t={t}:")
        for btn in ("jump", "fire"):
            tp = s[f"tp_{btn}"]; fp = s[f"fp_{btn}"]
            fn = s[f"fn_{btn}"]; tn = s[f"tn_{btn}"]
            total   = tp + fp + fn + tn
            correct = tp + tn
            acc  = correct / max(total, 1)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"    {btn:4s}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}"
                  f"  (TP={tp} FP={fp} FN={fn} TN={tn})")

    _print_top_action_confusion(confusion, args.confusion_top_k)

    print("─────────────────────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ContraVLA offline accuracy benchmark")
    p.add_argument("--config",     default=DEFAULT_CONFIG, help="Path to YAML config file")
    p.add_argument("--name",       default=None, help="Experiment name used to derive checkpoint path")
    p.add_argument("--data_dir",   default="vla/data/level1_action2")
    p.add_argument("--out",        default=None, help="Training output dir; defaults from --name")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint to evaluate; defaults to <out>/best.pt")
    p.add_argument("--n_samples",  type=int, default=1000)
    p.add_argument("--confusion_top_k", type=int, default=5,
                   help="Number of most common validation actions to show in the confusion matrix")
    p.add_argument("--device",     default="cuda")

    partial_args, _ = p.parse_known_args()
    yaml_defaults = _load_yaml_defaults(partial_args.config)
    if yaml_defaults:
        p.set_defaults(**yaml_defaults)

    args = p.parse_args()

    if args.out is None:
        args.out = f"tmp/vla/{args.name}" if args.name else "tmp/vla"
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.out, "best.pt")

    backtest(args)


if __name__ == "__main__":
    main()
