"""
benchmark.py — Run one or many episodes with a trained NESPolicyModel.

Auto-detects BC and DPO checkpoints. Supports single-episode recording
(like the old play.py) and multi-episode benchmarking.

Usage:
    # Single episode + save NPZ for video rendering
    python pixel2play/benchmark.py --checkpoint tmp/checkpoints/nes_policy/last.ckpt --out agent.npz

    # Multi-episode benchmark
    python pixel2play/benchmark.py --checkpoint tmp/checkpoints/nes_policy_dpo/best.ckpt \
        --level 2 --n_episodes 20 --temperature 0.0
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
from typing import Optional

import contra  # registers custom ROM integration
import cv2
import numpy as np
import stable_retro as retro
import torch

from contra.events import (
    EV_LEVELUP, EV_PLAYER_DIE, EV_ENEMY_HIT, ADDR_XSCROLL,
    compute_reward,
)
from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from contra.replay import replay_actions, save_video
from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_actions import decode_combined
from pixel2play.model.nes_policy import NESPolicyModel
from pixel2play.train import NESLightningModule, _load_config

# ── Constants ─────────────────────────────────────────────────────────────────

GAME              = "Contra-Nes"
SPREAD_STATES_DIR = os.path.join(os.path.dirname(__file__), "../contra/integration/Contra-Nes/spread_gun_state")

RAM_SIZE = 2048  # NES RAM bytes
EMU_SKIP = 3     # env steps per agent step (20 Hz agent @ 60 Hz NES)

_DPAD_NES   = np.array(DPAD_TABLE,   dtype=np.int8)   # (9, 9)
_BUTTON_NES = np.array(BUTTON_TABLE, dtype=np.int8)   # (4, 9)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0.0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))


def _process_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """RGB → grayscale size×size."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def _decode_action(dpad: int, button: int) -> np.ndarray:
    return np.clip(_DPAD_NES[dpad] + _BUTTON_NES[button], 0, 1).astype(np.int8)


def _xscroll(ram: np.ndarray) -> int:
    return (int(ram[ADDR_XSCROLL]) << 8) | int(ram[ADDR_XSCROLL + 1])


# ── Environment setup ─────────────────────────────────────────────────────────

def make_env(level: str):
    # Normalize e.g. "5" → "Level5", "level5" → "Level5"
    level = level.strip()
    if level.isdigit():
        level = f"Level{level}"
    elif level and level[0].islower():
        level = level.capitalize()

    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()

    if level == "Level1":
        state_path = os.path.join(
            os.path.dirname(__file__), "..", "contra", "integration", "Contra-Nes", "Level1.state"
        )
    else:
        state_path = os.path.join(SPREAD_STATES_DIR, f"{level}.state")

    state_bytes = None
    for opener in (gzip.open, open):
        try:
            with opener(state_path, "rb") as f:
                state_bytes = f.read()
            break
        except OSError:
            continue
    if state_bytes is None:
        raise FileNotFoundError(f"Could not load state file: {state_path}")

    env.em.set_state(state_bytes)
    env.data.update_ram()
    obs = env.em.get_screen().copy()

    initial_state = env.em.get_state()
    return env, obs, initial_state


# ── Model loading ─────────────────────────────────────────────────────────────

def _infer_backbone_cfg(state_dict: dict) -> BackboneConfig:
    """Infer BackboneConfig from checkpoint state dict keys/shapes."""
    embed_key = "action_embed.weight"
    if embed_key not in state_dict:
        embed_key = "model.action_embed.weight"
    dim = state_dict[embed_key].shape[1]

    layer_re = re.compile(r"(?:model\.)?backbone\.transformer\.layers\.(\d+)\.")
    n_layers = max(
        (int(m.group(1)) for k in state_dict for m in [layer_re.match(k)] if m),
        default=9,
    ) + 1

    # Detect vision encoder from ResNet stem key
    use_vision = any("backbone.img_tokenizer.stem" in k for k in state_dict)

    return BackboneConfig(dim=dim, n_layers=n_layers, n_steps=200, dropout=0.0, use_vision=use_vision)


def load_model(checkpoint_path: str, device: torch.device, config_path: Optional[str] = None) -> NESPolicyModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Detect DPO checkpoint (has policy_model.* keys) vs BC checkpoint (has model.* keys)
    is_dpo = any(k.startswith("policy_model") for k in state_dict)
    if is_dpo:
        state_dict = {k.replace("policy_model._orig_mod.", "policy_model."): v for k, v in state_dict.items()}
        state_dict = {k[len("policy_model."):]: v for k, v in state_dict.items() if k.startswith("policy_model.")}
    else:
        state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
        state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}

    # Infer grid_size and vision_depth from checkpoint state_dict directly
    grid_size = 2
    vision_depth = 4
    if "backbone.img_pos" in state_dict:
        img_pos_shape = state_dict["backbone.img_pos"].shape
        grid_size = int(img_pos_shape[1] ** 0.5)
    elif "model.backbone.img_pos" in state_dict:
        img_pos_shape = state_dict["model.backbone.img_pos"].shape
        grid_size = int(img_pos_shape[1] ** 0.5)
        
    proj_weight_key = None
    if "backbone.img_tokenizer.proj.0.weight" in state_dict:
        proj_weight_key = "backbone.img_tokenizer.proj.0.weight"
    elif "model.backbone.img_tokenizer.proj.0.weight" in state_dict:
        proj_weight_key = "model.backbone.img_tokenizer.proj.0.weight"
        
    if proj_weight_key:
        proj_shape = state_dict[proj_weight_key].shape
        if proj_shape[1] == 128:
            vision_depth = 3
        elif proj_shape[1] == 256:
            vision_depth = 4

    if config_path is not None:
        raw_cfg = _load_config(config_path)
        shared  = raw_cfg["shared"]
        pm      = raw_cfg.get("policy_model", {})
        backbone_cfg = BackboneConfig(
            n_steps=shared["n_seq_timesteps"],
            dim=pm.get("transformer_dim", 1024),
            n_layers=pm.get("n_transformer_layers", 10),
            n_q_heads=pm.get("n_q_head", 16),
            n_kv_heads=pm.get("n_kv_head", 16),
            n_action_tokens=pm.get("n_action_tokens", 1),
            mask_block_size=pm.get("mask_block_size", 128),
            attention_history_len=pm.get("attention_history_len", None),
            dropout=0.0,
            use_vision=pm.get("use_vision", False),
            in_channels=pm.get("in_channels", 1),
            grid_size=grid_size,
            vision_depth=vision_depth,
        )
    else:
        backbone_cfg = _infer_backbone_cfg(state_dict)
        backbone_cfg.grid_size = grid_size
        backbone_cfg.vision_depth = vision_depth
        print(f"  Inferred config: dim={backbone_cfg.dim}  n_layers={backbone_cfg.n_layers}"
              f"  n_action_tokens={backbone_cfg.n_action_tokens}")

    ablate = None
    if config_path is not None:
        pm = raw_cfg.get("policy_model", {})
        ablate = pm.get("ablate", None)

    model = NESPolicyModel(backbone_cfg)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    if ablate is not None:
        model.backbone.cfg.ablate = ablate
    model.backbone._build_block_masks()
    return model


# ── Core episode runner ───────────────────────────────────────────────────────

def run_episode(
    model: NESPolicyModel,
    level: str = "1",
    n_steps: int = 2000,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    frame_size: int = 84,
) -> dict:
    """Run one episode and return gameplay metrics + recorded actions."""
    if device is None:
        device = next(model.parameters()).device

    ctx_len = model.backbone.cfg.n_steps
    use_vision = model.backbone.cfg.use_vision
    env, _, initial_state = make_env(level)

    if use_vision:
        obs_buf    = torch.zeros(ctx_len, 1, frame_size, frame_size, dtype=torch.uint8)
    else:
        obs_buf    = torch.zeros(ctx_len, RAM_SIZE, dtype=torch.uint8)
    action_buf = torch.zeros(ctx_len, dtype=torch.long)
    buf_len    = 0

    recorded_actions: list[np.ndarray] = []
    max_xscroll  = 0
    enemies_hit  = 0.0
    total_reward = 0.0
    level_up     = False

    for step in range(n_steps):
        if use_vision:
            screen = env.em.get_screen().copy()  # (H, W, 3) RGB
            frame = torch.from_numpy(_process_frame(screen, size=frame_size)).unsqueeze(0)  # (1, size, size)
        else:
            ram = torch.from_numpy(env.unwrapped.get_ram().copy())   # (2048,) uint8

        if buf_len < ctx_len:
            obs_buf[buf_len] = frame if use_vision else ram
            pos     = buf_len
            buf_len += 1
        else:
            obs_buf    = torch.roll(obs_buf,    -1, dims=0)
            action_buf = torch.roll(action_buf, -1, dims=0)
            obs_buf[-1]    = frame if use_vision else ram
            action_buf[-1] = 0
            pos = ctx_len - 1

        with torch.no_grad():
            action_logits = model(
                obs_buf.unsqueeze(0).to(device),
                action_buf.unsqueeze(0).to(device),
            )                                                     # (1, ctx_len, N_ACTIONS)

        pred_action = _sample(action_logits[0, pos], temperature)
        action_buf[pos] = pred_action

        pred_dpad, pred_button = decode_combined(pred_action)
        action = _decode_action(pred_dpad, pred_button)
        recorded_actions.append(action)

        pre_ram = env.unwrapped.get_ram().copy()
        terminated = truncated = False
        for i in range(EMU_SKIP):
            _, _, terminated, truncated, _ = env.step(action.copy())
            if terminated or truncated:
                break
        curr_ram = env.unwrapped.get_ram()

        max_xscroll  = max(max_xscroll, _xscroll(curr_ram))
        enemies_hit += EV_ENEMY_HIT.trigger(pre_ram, curr_ram)
        total_reward += compute_reward(pre_ram, curr_ram)

        if EV_LEVELUP.trigger(pre_ram, curr_ram):
            level_up = True
            break

        if EV_PLAYER_DIE.trigger(pre_ram, curr_ram):
            break

        if terminated or truncated:
            break

    env.close()

    return {
        "actions":       np.stack(recorded_actions).astype(np.int8),
        "initial_state": initial_state,
        "steps":         len(recorded_actions),
        "xscroll":       max_xscroll,
        "enemies_hit":   enemies_hit,
        "total_reward":  total_reward,
        "level_up":      level_up,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint file, or directory containing best.ckpt + config.yaml")
    parser.add_argument("--config",      default=None, help="YAML config used to train the checkpoint (optional if checkpoint is a dir)")
    parser.add_argument("--frame_size",  type=int,   default=None, help="Input frame size (84 or 168). Overrides config if set.")
    parser.add_argument("--level",       default="1")
    parser.add_argument("--n_episodes",  type=int,   default=10)
    parser.add_argument("--n_steps",     type=int,   default=3000)
    parser.add_argument("--out",         default=None, help="Save single-episode actions to NPZ")
    parser.add_argument("--out_dir",     default='tmp', help="Save multi-episode actions to directory")
    parser.add_argument("--save_best_video", default=None, help="Save the best episode as an MP4 video to this path")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. 0=argmax (deterministic)")
    parser.add_argument("--ablate",      default=None,
                        help="Comma-separated token types to zero-out for ablation study. "
                             "E.g. 'img' = action-only; 'action' = image-only.")
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    config_path = args.config
    if os.path.isdir(checkpoint_path):
        if config_path is None:
            config_path = os.path.join(checkpoint_path, "config.yaml")
        checkpoint_path = os.path.join(checkpoint_path, "best.ckpt")

    device = torch.device(args.device)
    print(f"Loading checkpoint: {checkpoint_path}  device={device}")
    model = load_model(checkpoint_path, device, config_path=config_path)

    # Apply ablation for inference experiments
    if args.ablate is not None:
        ablate_list = [t.strip() for t in args.ablate.split(",") if t.strip()]
        model.backbone.cfg.ablate = ablate_list
        print(f"  Ablating tokens: {ablate_list}")

    # Determine frame size: CLI arg > config > default 84
    frame_size = 84
    if config_path is not None and os.path.exists(config_path):
        raw_cfg = _load_config(config_path)
        pm = raw_cfg.get("policy_model", {})
        frame_size = pm.get("frame_size", 84)
    if args.frame_size is not None:
        frame_size = args.frame_size
    print(f"  Frame size: {frame_size}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for ep in range(args.n_episodes):
        result = run_episode(
            model,
            level=args.level,
            n_steps=args.n_steps,
            temperature=args.temperature,
            device=device,
            frame_size=frame_size,
        )
        results.append(result)

        status = "CLEAR" if result["level_up"] else "DEAD"
        print(
            f"Ep {ep + 1:02d} | {status} | steps={result['steps']:4d} | "
            f"reward={result['total_reward']:8.1f} | xscroll={result['xscroll']:5d} | "
            f"enemies={result['enemies_hit']:4.0f}"
        )

        if args.out_dir:
            out_path = os.path.join(args.out_dir, f"ep_{ep + 1:02d}.npz")
            np.savez_compressed(
                out_path,
                actions       = result["actions"],
                initial_state = np.frombuffer(result["initial_state"], dtype=np.uint8),
                level         = np.array(args.level),
                fps           = np.array(20),
                game          = np.array(GAME),
            )

    # ── Aggregate stats (only printed for multi-episode) ────────────────────
    if args.n_episodes > 1:
        successes = sum(1 for r in results if r["level_up"])
        avg_steps   = sum(r["steps"]        for r in results) / len(results)
        avg_reward  = sum(r["total_reward"] for r in results) / len(results)
        avg_xscroll = sum(r["xscroll"]      for r in results) / len(results)
        avg_enemies = sum(r["enemies_hit"]  for r in results) / len(results)

        print("-" * 70)
        print(f"Episodes   : {args.n_episodes}")
        print(f"Success    : {successes} ({successes / args.n_episodes * 100:.1f}%)")
        print(f"Avg steps  : {avg_steps:.1f}")
        print(f"Avg reward : {avg_reward:.1f}")
        print(f"Avg scroll : {avg_xscroll:.1f}")
        print(f"Avg enemies: {avg_enemies:.1f}")

    # ── Single-episode save ─────────────────────────────────────────────────
    if args.out:
        result = results[0]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(
            args.out,
            actions       = result["actions"],
            initial_state = np.frombuffer(result["initial_state"], dtype=np.uint8),
            level         = np.array(args.level),
            fps           = np.array(20),
            game          = np.array(GAME),
        )
        print(f"Saved {result['steps']} actions → {args.out}")
        print(f"Render video:  python contra/replay.py {args.out} --video out.mp4")

    if args.out_dir:
        print(f"Saved recordings → {args.out_dir}")

    # ── Save best episode video ─────────────────────────────────────────────
    if args.save_best_video:
        # Best = level_up > xscroll > total_reward
        def _score(r):
            return (1 if r["level_up"] else 0, r["xscroll"], r["total_reward"])

        best_idx = max(range(len(results)), key=lambda i: _score(results[i]))
        best = results[best_idx]
        print(f"\nBest episode: Ep {best_idx + 1:02d} | "
              f"level_up={best['level_up']} | xscroll={best['xscroll']} | reward={best['total_reward']:.1f}")

        print("Replaying best episode for video capture...")
        replay = replay_actions(
            best["actions"],
            initial_state=best["initial_state"],
            want_video=True,
            verbose=False,
        )
        save_video(replay["video"], args.save_best_video)
        print(f"Best episode video saved → {args.save_best_video}")


if __name__ == "__main__":
    main()
