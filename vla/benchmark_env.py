"""
vla/benchmark_env.py — Live emulator evaluation for a trained ContraVLA checkpoint.

Runs N episodes in the Contra emulator using temperature sampling (default T=1.0)
so each episode explores a different trajectory.  Use --temperature 0 for greedy
argmax (deterministic, same as generate_actions).

Usage:
    python -m vla.benchmark_env \
        --config vla/behavior_clone.yaml \
        --temperature 1.0
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.distributions import Categorical
from transformers import AutoTokenizer

from vla.datasets.dataset import LEVEL_TEXTS
from vla.model import ContraVLA, ContraVLAConfig
from vla.post_training.env_wrappers import VLAEnv

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


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device) -> ContraVLA:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]
    sd   = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    cfg  = ContraVLAConfig(dropout=0.0)
    mdl  = ContraVLA(cfg)
    mdl.load_state_dict(sd, strict=True)
    return mdl.to(device).eval()


def _make_inputs(model: ContraVLA, level_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tok = AutoTokenizer.from_pretrained(model.config.vlm_model_name)
    enc = tok(LEVEL_TEXTS[level_id], return_tensors="pt",
              padding="max_length", max_length=32, truncation=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)  # [1, L]


@torch.no_grad()
def _sample_actions(
    model:       ContraVLA,
    input_ids:   torch.Tensor,   # [1, L]
    attention_mask: torch.Tensor, # [1, L]
    images:      torch.Tensor,   # [1, 2, 3, H, W]
    proprio:     torch.Tensor,   # [1, 118]
    temperature: float = 1.0,
) -> list[int]:
    """Return one action chunk (T ints) sampled at the given temperature."""
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        proprio=proprio,
    )["logits"]
    if temperature == 0.0:
        return logits.argmax(dim=-1)[0].cpu().tolist()
    return Categorical(logits=logits[0] / temperature).sample().cpu().tolist()


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(
    model:       ContraVLA,
    input_ids:   torch.Tensor,
    attention_mask: torch.Tensor,
    device:      torch.device,
    temperature: float = 1.0,
    max_chunks:  int   = 1500,
    level:       str   = "Level1",
) -> dict:
    """
    Run one episode.  Each step queries the model for a T-action chunk, which
    VLAEnv steps through with per-frame death/levelup detection.

    Returns a dict with steps, total_reward, status, xscroll.
    """
    env = VLAEnv(level=level)
    obs = env.reset()

    total_reward = 0.0
    status       = VLAEnv.RUNNING
    chunk        = 0

    for chunk in range(max_chunks):
        images  = obs["images"].unsqueeze(0).to(device)   # [1, 2, 3, H, W]
        proprio = obs["proprio"].unsqueeze(0).to(device)  # [1, 118]

        actions        = _sample_actions(model, input_ids, attention_mask, images, proprio, temperature)
        obs, reward, status, _ = env.step(actions)
        total_reward  += reward

        if status != VLAEnv.RUNNING:
            break

    xscroll = int(obs["proprio"][0].item())
    env.close()

    return {
        "chunks":       chunk + 1,
        "total_reward": total_reward,
        "status":       status,
        "xscroll":      xscroll,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ContraVLA live emulator benchmark")
    p.add_argument("--config",       default=DEFAULT_CONFIG, help="Path to YAML config file")
    p.add_argument("--name",         default=None, help="Experiment name used to derive checkpoint path")
    p.add_argument("--out",          default=None, help="Training output dir; defaults from --name")
    p.add_argument("--checkpoint",   default=None,
                   help="Checkpoint to evaluate; defaults to <out>/best.pt")
    p.add_argument("--n_episodes",   type=int,   default=10)
    p.add_argument("--temperature",  type=float, default=1.0,
                   help="Sampling temperature. 0 = greedy argmax (deterministic).")
    p.add_argument("--max_chunks",   type=int,   default=1500,
                   help="Max model queries per episode (each = T actions × EMU_SKIP frames).")
    p.add_argument("--level",        default="Level1")
    p.add_argument("--device",       default="cuda")

    partial_args, _ = p.parse_known_args()
    yaml_defaults = _load_yaml_defaults(partial_args.config)
    if yaml_defaults:
        p.set_defaults(**yaml_defaults)

    args = p.parse_args()

    if args.out is None:
        args.out = f"tmp/vla/{args.name}" if args.name else "tmp/vla"
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.out, "best.pt")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"checkpoint : {args.checkpoint}")
    print(f"episodes   : {args.n_episodes}")
    print(f"device     : {device}  temperature={args.temperature}  max_chunks={args.max_chunks}\n")

    model     = _load_model(args.checkpoint, device)
    input_ids, attention_mask = _make_inputs(model, level_id=0, device=device)

    results = []
    for ep in range(args.n_episodes):
        r = _run_episode(model, input_ids, attention_mask, device,
                         temperature=args.temperature,
                         max_chunks=args.max_chunks,
                         level=args.level)
        results.append(r)
        status_str = {"DEAD": "DEAD", "DONE": "CLEAR", "RUNNING": "END"}.get(r["status"], r["status"])
        print(
            f"Ep {ep + 1:02d} | {status_str:5s} | "
            f"chunks={r['chunks']:4d} | xscroll={r['xscroll']:5d} | "
            f"reward={r['total_reward']:9.1f}"
        )

    if args.n_episodes > 1:
        clears     = sum(1 for r in results if r["status"] == VLAEnv.DONE)
        deaths     = sum(1 for r in results if r["status"] == VLAEnv.DEAD)
        avg_chunks = sum(r["chunks"]       for r in results) / len(results)
        avg_xscroll= sum(r["xscroll"]      for r in results) / len(results)
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print("─" * 60)
        print(f"Episodes   : {args.n_episodes}")
        print(f"Cleared    : {clears}  ({100*clears/args.n_episodes:.0f}%)")
        print(f"Died       : {deaths}  ({100*deaths/args.n_episodes:.0f}%)")
        print(f"Avg chunks : {avg_chunks:.0f}")
        print(f"Avg xscroll: {avg_xscroll:.0f}")
        print(f"Avg reward : {avg_reward:.1f}")


if __name__ == "__main__":
    main()
