"""
Compute confusion matrix for a trained NESPolicyModel checkpoint.

Focuses on button-level accuracy (fire / jump / none / A+B) since those are
the critical gameplay actions.

Usage:
    python pixel2play/confusion_matrix.py \
        --checkpoint tmp/checkpoints/nes_policy/best.ckpt \
        --config pixel2play/nes_10M_168.yaml \
        --data_folder annotate/bc_data/level1_grey_168 \
        --n_val_recordings 50
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from pixel2play.benchmark import load_model
from pixel2play.dataset import NESDataset
from pixel2play.model.nes_actions import decode_combined, N_BUTTONS


def _load_val_data(data_folder: str, n_steps: int, n_val_recordings: int):
    full = NESDataset(data_folder, n_steps=n_steps)
    recs = full.recordings[:]
    n_val = min(n_val_recordings, len(recs) - 1) if n_val_recordings > 0 else 0
    val_recs = recs[:n_val]
    val_set = NESDataset.from_recordings(val_recs, n_steps)
    return val_set


def _decode_button(action: int) -> int:
    """Extract button class (0=none, 1=jump/A, 2=fire/B, 3=A+B) from combined action."""
    _, button = decode_combined(action)
    return button


def _button_name(button: int) -> str:
    names = {0: "none", 1: "jump", 2: "fire", 3: "jump+fire"}
    return names.get(button, f"?{button}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", default=None, help="Path to YAML config (optional)")
    parser.add_argument("--data_folder", required=True, help="Validation data folder")
    parser.add_argument("--n_steps", type=int, default=128, help="Sequence length")
    parser.add_argument("--n_val_recordings", type=int, default=50, help="Num validation recordings")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_full", default=None, help="Path to save full 36x36 confusion matrix (npy)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, torch.device(args.device), config_path=args.config)
    model.eval()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading validation data from: {args.data_folder}")
    val_set = _load_val_data(args.data_folder, args.n_steps, args.n_val_recordings)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Validation samples: {len(val_set)}")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    all_pred = []
    all_true = []
    all_valid = []

    with torch.no_grad():
        for batch in loader:
            obs, dpad, button, valid_mask = batch
            action_true = (dpad * N_BUTTONS + button).to(args.device)
            valid_mask = valid_mask.to(args.device)

            # Teacher-forced: feed ground-truth actions as history
            action_logits = model(obs.to(args.device), action_true)
            action_pred = action_logits.argmax(dim=-1)

            all_pred.append(action_pred.cpu().numpy())
            all_true.append(action_true.cpu().numpy())
            all_valid.append(valid_mask.cpu().numpy())

    pred = np.concatenate(all_pred).flatten()
    true = np.concatenate(all_true).flatten()
    valid = np.concatenate(all_valid).flatten()

    # Keep only valid (non-padded) frames
    pred = pred[valid]
    true = true[valid]

    n_total = len(pred)
    overall_acc = (pred == true).mean()
    print(f"\n{'='*60}")
    print(f"Total valid frames: {n_total}")
    print(f"Overall accuracy:   {overall_acc:.4f}")

    # ------------------------------------------------------------------
    # Button-level confusion matrix (4 x 4)
    # ------------------------------------------------------------------
    pred_btn = np.array([_decode_button(a) for a in pred])
    true_btn = np.array([_decode_button(a) for a in true])

    btn_labels = [0, 1, 2, 3]
    btn_names = [_button_name(b) for b in btn_labels]

    cm_btn = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(true_btn, pred_btn):
        cm_btn[t, p] += 1

    print(f"\n{'='*60}")
    print("Button-level Confusion Matrix")
    print(f"{' '*12}", end="")
    for name in btn_names:
        print(f"{name:>12}", end="")
    print()
    for i, name in enumerate(btn_names):
        print(f"{name:>12}", end="")
        for j in range(4):
            print(f"{cm_btn[i, j]:>12,}", end="")
        print()

    # ------------------------------------------------------------------
    # Per-button metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Per-button metrics")
    print(f"{' '*12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for btn in btn_labels:
        tp = cm_btn[btn, btn]
        fp = cm_btn[:, btn].sum() - tp
        fn = cm_btn[btn, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm_btn[btn, :].sum()
        print(f"{_button_name(btn):>12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10,}")

    # ------------------------------------------------------------------
    # Fire vs Jump specific analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Fire vs Jump confusion")

    # Fire (button=2) analysis
    fire_true_mask = true_btn == 2
    fire_pred_mask = pred_btn == 2
    jump_pred_mask = pred_btn == 1

    fire_support = fire_true_mask.sum()
    fire_tp = (fire_true_mask & fire_pred_mask).sum()
    fire_fn = (fire_true_mask & ~fire_pred_mask).sum()
    fire_fp = (~fire_true_mask & fire_pred_mask).sum()
    fire_missed_as_jump = (fire_true_mask & jump_pred_mask).sum()

    print(f"\nFire (button=B):")
    print(f"  Support:            {fire_support:,}")
    print(f"  Correct:            {fire_tp:,}  ({100*fire_tp/fire_support:.2f}%)")
    print(f"  Missed (any):       {fire_fn:,}")
    print(f"  Missed as Jump:     {fire_missed_as_jump:,}  ({100*fire_missed_as_jump/fire_support:.2f}% of fire frames)")

    # Jump (button=1) analysis
    jump_true_mask = true_btn == 1
    jump_support = jump_true_mask.sum()
    jump_tp = (jump_true_mask & jump_pred_mask).sum()
    jump_fn = (jump_true_mask & ~jump_pred_mask).sum()
    jump_missed_as_fire = (jump_true_mask & fire_pred_mask).sum()

    print(f"\nJump (button=A):")
    print(f"  Support:            {jump_support:,}")
    print(f"  Correct:            {jump_tp:,}  ({100*jump_tp/jump_support:.2f}%)")
    print(f"  Missed (any):       {jump_fn:,}")
    print(f"  Missed as Fire:     {jump_missed_as_fire:,}  ({100*jump_missed_as_fire/jump_support:.2f}% of jump frames)")

    # ------------------------------------------------------------------
    # Full 36x36 combined-action confusion matrix (optional save)
    # ------------------------------------------------------------------
    if args.save_full:
        cm_full = np.zeros((36, 36), dtype=np.int64)
        for t, p in zip(true, pred):
            cm_full[t, p] += 1
        np.save(args.save_full, cm_full)
        print(f"\nFull 36x36 confusion matrix saved to: {args.save_full}")

    # ------------------------------------------------------------------
    # Most confused action pairs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Top confused action pairs (true -> pred)")
    errors = []
    for t in range(36):
        for p in range(36):
            if t != p and cm_full[t, p] > 0:
                errors.append((cm_full[t, p], t, p))
    errors.sort(reverse=True)
    for count, t, p in errors[:10]:
        td, tb = decode_combined(t)
        pd, pb = decode_combined(p)
        print(f"  {count:>6,}  dpad={td},btn={tb} -> dpad={pd},btn={pb}")


if __name__ == "__main__":
    main()
