"""
vla/benchmark.py — Evaluate a trained ContraVLA checkpoint.

Two evaluation modes:

  backtest   Load random episodes from the val split and compute
             jump / fire button accuracy against ground-truth actions.

  realtime   Run the model live in the Contra emulator and accumulate
             reward computed by contra.events.compute_reward().

Usage:
    # Offline button accuracy (5 random episodes ≈ 1000 samples)
    python -m vla.benchmark backtest \
        --checkpoint tmp/vla/unfrozen_400/best.pt \
        --data_dir   vla/data/level1_action_600 \
        --n_samples  1000

    # Live emulator reward
    python -m vla.benchmark realtime \
        --checkpoint tmp/vla/unfrozen_400/best.pt \
        --n_episodes 3
"""

from __future__ import annotations

import argparse
import io
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

import contra  # registers the Contra-Nes integration
import stable_retro as retro

from contra.events import (
    EV_LEVELUP, EV_PLAYER_DIE, compute_reward,
)
from contra.game_state import state_from_ram
from vla.datasets.dataset import LEVEL_TEXTS, _preprocess_frames, ContraVLADataset
from vla.model import ContraVLA, ContraVLAConfig

# ── Constants ─────────────────────────────────────────────────────────────────

GAME      = "Contra-Nes"
EMU_SKIP  = 3          # env steps per agent step (20 Hz agent @ 60 Hz NES)
IMG_SIZE  = 192        # must match training preprocessing

# NES action layout: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Combined action index: dpad * 4 + button
# D-pad combos (9): none, L, R, U, D, UL, UR, DL, DR
# Button combos (4): none, Jump(A), Fire(B), Fire+Jump
_DPAD_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1: L
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2: R
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: U
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: D
    [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 5: UL
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 6: UR
    [0, 0, 0, 0, 0, 1, 1, 0, 0],  # 7: DL
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 8: DR
], dtype=np.int8)

_BUTTON_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1: Jump (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: Fire (B)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: Fire+Jump
], dtype=np.int8)

# ImageNet normalisation (matches training in vla/datasets/dataset.py)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_jump(action: int) -> bool:
    return (action % 4) in (1, 3)

def _has_fire(action: int) -> bool:
    return (action % 4) in (2, 3)

def _decode_nes(action: int) -> np.ndarray:
    dpad, button = action // 4, action % 4
    return np.clip(_DPAD_TABLE[dpad] + _BUTTON_TABLE[button], 0, 1).astype(np.int8)

def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) → resized (IMG_SIZE, IMG_SIZE, 3) uint8."""
    return np.array(Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> ContraVLA:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    # strip torch.compile prefix if present
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

    config = ContraVLAConfig(dropout=0.0)
    model  = ContraVLA(config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def _make_tokenizer(config: ContraVLAConfig, max_len: int = 32):
    tok = AutoTokenizer.from_pretrained(config.vlm_model_name)
    return tok, max_len


def _tokenize_level(tokenizer, max_len: int, level_id: int, device: torch.device) -> torch.Tensor:
    text = LEVEL_TEXTS[level_id]
    enc  = tokenizer(text, return_tensors="pt", padding="max_length",
                     max_length=max_len, truncation=True)
    return enc["input_ids"].to(device)  # (1, L)


def _frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """frames: [2, H, W, 3] uint8 → [1, 2, 3, H, W] float32 normalised."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)  # [2, 3, H, W]
    t = (t - _MEAN) / _STD
    return t.unsqueeze(0).to(device)                                        # [1, 2, 3, H, W]


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(args: argparse.Namespace) -> None:
    """Offline accuracy: load val samples, run model, compare to ground truth."""
    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}  device={device}")
    model = load_model(args.checkpoint, device)

    tokenizer, max_len = _make_tokenizer(model.config)
    input_ids = _tokenize_level(tokenizer, max_len, level_id=0, device=device)  # Level1

    val_dir = os.path.join(args.data_dir, "val")
    print(f"Loading val dataset from: {val_dir}")
    dataset = ContraVLADataset(val_dir, level_id=0)
    n_total = len(dataset)
    print(f"  {n_total} total val samples")

    n = min(args.n_samples, n_total)
    indices = random.sample(range(n_total), n)
    print(f"  Evaluating {n} random samples\n")

    # Accumulators per time-step in the chunk
    stats: dict[str, dict] = {}
    for t in range(model.config.num_actions):
        stats[t] = defaultdict(int)  # tp_jump, fp_jump, fn_jump, tn_jump, tp_fire, ...

    overall_correct = 0
    overall_total   = 0

    with torch.no_grad():
        for sample_i, idx in enumerate(indices):
            sample = dataset[idx]
            images  = sample["images"].unsqueeze(0).to(device)   # [1, 2, 3, H, W]
            proprio = sample["proprio"].unsqueeze(0).to(device)   # [1, 118]
            gt_acts = sample["actions"]                           # [T] int64

            preds = model.generate_actions(input_ids, images, proprio)  # [1, T]
            preds = preds[0].cpu()                                        # [T]

            for t in range(model.config.num_actions):
                gt, pr = int(gt_acts[t]), int(preds[t])

                # jump accuracy
                gt_jump = _has_jump(gt)
                pr_jump = _has_jump(pr)
                if gt_jump and pr_jump:
                    stats[t]["tp_jump"] += 1
                elif gt_jump and not pr_jump:
                    stats[t]["fn_jump"] += 1
                elif not gt_jump and pr_jump:
                    stats[t]["fp_jump"] += 1
                else:
                    stats[t]["tn_jump"] += 1

                # fire accuracy
                gt_fire = _has_fire(gt)
                pr_fire = _has_fire(pr)
                if gt_fire and pr_fire:
                    stats[t]["tp_fire"] += 1
                elif gt_fire and not pr_fire:
                    stats[t]["fn_fire"] += 1
                elif not gt_fire and pr_fire:
                    stats[t]["fp_fire"] += 1
                else:
                    stats[t]["tn_fire"] += 1

                # exact action accuracy
                overall_correct += int(gt == pr)
                overall_total   += 1

            if (sample_i + 1) % 100 == 0:
                print(f"  [{sample_i + 1}/{n}] ...", flush=True)

    # ── Print results ──
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
            acc     = correct / max(total, 1)
            prec    = tp / max(tp + fp, 1)
            rec     = tp / max(tp + fn, 1)
            f1      = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"    {btn:4s}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}"
                  f"  (TP={tp} FP={fp} FN={fn} TN={tn})")
    print("─────────────────────────────────────────────────────────────────")


# ── Realtime ──────────────────────────────────────────────────────────────────

def _make_env(level: str = "Level1"):
    import gzip
    state_path = os.path.join(
        os.path.dirname(__file__), "..", "contra", "integration", "Contra-Nes",
        f"{level}.state",
    )
    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()

    state_bytes = None
    for opener in (gzip.open, open):
        try:
            with opener(state_path, "rb") as f:
                state_bytes = f.read()
            break
        except OSError:
            continue
    if state_bytes is None:
        raise FileNotFoundError(f"Could not load state: {state_path}")

    env.em.set_state(state_bytes)
    env.data.update_ram()
    return env


def _run_episode(
    model: ContraVLA,
    input_ids: torch.Tensor,
    device: torch.device,
    n_steps: int = 3000,
) -> dict:
    env = _make_env("Level1")

    prev_frame = _preprocess_frame(env.em.get_screen().copy())
    curr_frame = prev_frame.copy()

    total_reward = 0.0
    level_up     = False
    died         = False

    for step in range(n_steps):
        frames = np.stack([prev_frame, curr_frame])                        # [2, H, W, 3]
        images  = _frames_to_tensor(frames, device)                        # [1, 2, 3, H, W]
        proprio = torch.from_numpy(
            state_from_ram(env.unwrapped.get_ram())
        ).unsqueeze(0).to(device)                                          # [1, 118]

        with torch.no_grad():
            act_chunk = model.generate_actions(input_ids, images, proprio) # [1, T]
        act_chunk = act_chunk[0].cpu().tolist()                            # list of T ints

        # Execute all T actions in the chunk before re-querying the model
        for action_idx in act_chunk:
            nes_action = _decode_nes(action_idx)
            pre_ram    = env.unwrapped.get_ram().copy()

            terminated = truncated = False
            for _ in range(EMU_SKIP):
                _, _, terminated, truncated, _ = env.step(nes_action.copy())
                if terminated or truncated:
                    break

            curr_ram      = env.unwrapped.get_ram()
            total_reward += compute_reward(pre_ram, curr_ram)

            if EV_LEVELUP.trigger(pre_ram, curr_ram):
                level_up = True
            if EV_PLAYER_DIE.trigger(pre_ram, curr_ram):
                died = True

            if level_up or died or terminated or truncated:
                break

        prev_frame = curr_frame
        curr_frame = _preprocess_frame(env.em.get_screen().copy())

        if level_up or died or terminated or truncated:
            break

    env.close()
    return {
        "steps":        step + 1,
        "total_reward": total_reward,
        "level_up":     level_up,
        "died":         died,
    }


def realtime(args: argparse.Namespace) -> None:
    """Live emulator evaluation: accumulate reward using compute_reward()."""
    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}  device={device}")
    model = load_model(args.checkpoint, device)

    tokenizer, max_len = _make_tokenizer(model.config)
    input_ids = _tokenize_level(tokenizer, max_len, level_id=0, device=device)

    results = []
    for ep in range(args.n_episodes):
        result = _run_episode(model, input_ids, device, n_steps=args.n_steps)
        results.append(result)
        status = "CLEAR" if result["level_up"] else ("DEAD" if result["died"] else "END")
        print(
            f"Ep {ep + 1:02d} | {status} | steps={result['steps']:4d} | "
            f"reward={result['total_reward']:8.1f}"
        )

    if args.n_episodes > 1:
        successes  = sum(1 for r in results if r["level_up"])
        avg_steps  = sum(r["steps"]        for r in results) / len(results)
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print("─" * 60)
        print(f"Episodes  : {args.n_episodes}")
        print(f"Cleared   : {successes} ({100*successes/args.n_episodes:.0f}%)")
        print(f"Avg steps : {avg_steps:.0f}")
        print(f"Avg reward: {avg_reward:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ContraVLA benchmark")
    sub = p.add_subparsers(dest="mode", required=True)

    # shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--checkpoint", default="tmp/vla/unfrozen_400/best.pt")
    shared.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")

    # backtest sub-command
    bt = sub.add_parser("backtest", parents=[shared])
    bt.add_argument("--data_dir",  default="vla/data/level1_action_600")
    bt.add_argument("--n_samples", type=int, default=1000,
                    help="Random samples to draw from val split (~200/episode)")

    # realtime sub-command
    rt = sub.add_parser("realtime", parents=[shared])
    rt.add_argument("--n_episodes", type=int, default=3)
    rt.add_argument("--n_steps",    type=int, default=3000)

    args = p.parse_args()

    if args.mode == "backtest":
        backtest(args)
    else:
        realtime(args)


if __name__ == "__main__":
    main()
