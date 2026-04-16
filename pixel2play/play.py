"""
play.py — Generate an action sequence using a trained NESPolicyModel checkpoint.

The model observes a growing context window:
  steps 0..199  → context grows from 1 to 200 (prediction at the last real position)
  steps 200+    → sliding window of the last 200 frames

The output is an NPZ file compatible with contra.replay.replay_actions:

    python contra/replay.py out.npz --video out.mp4

Usage:
    python pixel2play/play.py --checkpoint tmp/checkpoints/nes_policy/last.ckpt
    python pixel2play/play.py --checkpoint tmp/checkpoints/nes_policy/last.ckpt \\
        --level Level2 --n_steps 1000 --out tmp/agent_run.npz
"""

from __future__ import annotations

import argparse
import gzip
import os
from typing import Optional

import contra  # registers custom ROM integration
import numpy as np
import stable_retro as retro
import torch

from contra.events import EV_PLAYER_DIE, ADDR_XSCROLL, EV_ENEMY_HIT, EV_LEVELUP
from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_policy import NESPolicyModel
from pixel2play.train import NESLightningModule

# ── Constants ─────────────────────────────────────────────────────────────────

GAME              = "Contra-Nes"
SPREAD_STATES_DIR = os.path.join(os.path.dirname(__file__), "../contra/integration/Contra-Nes/spread_gun_state")

RAM_SIZE = 2048  # NES RAM bytes
EMU_SKIP = 3     # env steps per agent step (20 Hz agent @ 60 Hz NES)

_DPAD_NES   = np.array(DPAD_TABLE,   dtype=np.int8)   # (9, 9)
_BUTTON_NES = np.array(BUTTON_TABLE, dtype=np.int8)   # (4, 9)


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0.0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))


def _decode_action(dpad: int, button: int) -> np.ndarray:
    return np.clip(_DPAD_NES[dpad] + _BUTTON_NES[button], 0, 1).astype(np.int8)


def _xscroll(ram: np.ndarray) -> int:
    return (int(ram[ADDR_XSCROLL]) << 8) | int(ram[ADDR_XSCROLL + 1])


# ── Model loading ─────────────────────────────────────────────────────────────

def _infer_backbone_cfg(state_dict: dict) -> BackboneConfig:
    """Infer BackboneConfig from checkpoint state dict keys/shapes."""
    import re as _re

    dim = state_dict["model.dpad_embed.weight"].shape[1]

    layer_re = _re.compile(r"model\.backbone\.transformer\.layers\.(\d+)\.")
    n_layers = max(
        (int(m.group(1)) for k in state_dict for m in [layer_re.match(k)] if m),
        default=9,
    ) + 1

    dec_re = _re.compile(r"model\.backbone\.action_decoder\.transformer\.layers\.(\d+)\.")
    dec_n_layers = max(
        (int(m.group(1)) for k in state_dict for m in [dec_re.match(k)] if m),
        default=2,
    ) + 1

    return BackboneConfig(
        dim=dim,
        n_layers=n_layers,
        dec_n_layers=dec_n_layers,
        n_steps=200,
        dropout=0.0,
    )


def load_model(checkpoint_path: str, device: torch.device, config_path: Optional[str] = None) -> NESPolicyModel:
    if config_path is not None:
        import yaml
        from pixel2play.train import _load_config
        raw_cfg = _load_config(config_path)
        shared  = raw_cfg["shared"]
        pm      = raw_cfg.get("policy_model", {})
        dec     = pm.get("action_decoder", {})
        backbone_cfg = BackboneConfig(
            n_steps=shared["n_seq_timesteps"],
            n_text_tokens=shared.get("text_tokenizer_config", {}).get("text_embedding_shape", [1, 768])[0],
            dim=pm.get("transformer_dim", 1024),
            n_layers=pm.get("n_transformer_layers", 10),
            n_q_heads=pm.get("n_q_head", 16),
            n_kv_heads=pm.get("n_kv_head", 16),
            n_thinking_tokens=pm.get("n_thinking_tokens", 1),
            mask_block_size=pm.get("mask_block_size", 128),
            attention_history_len=pm.get("attention_history_len", None),
            dropout=0.0,
            dec_n_layers=dec.get("n_layers", 3),
            dec_n_heads=dec.get("n_heads", 8),
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        # torch.compile() adds '_orig_mod.' to keys — strip it if present
        state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
        backbone_cfg = _infer_backbone_cfg(state_dict)
        print(f"  Inferred config: dim={backbone_cfg.dim}  n_layers={backbone_cfg.n_layers}"
              f"  dec_n_layers={backbone_cfg.dec_n_layers}  n_action_tokens={backbone_cfg.n_action_tokens}")

    module = NESLightningModule(backbone_cfg, lr=1e-4)

    # torch.compile() adds '_orig_mod.' to keys — strip it if present
    fixed = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    module.load_state_dict(fixed, strict=True)

    model = module.model.to(device).eval()
    model.backbone._build_block_masks()
    return model


# ── Environment setup ─────────────────────────────────────────────────────────

def make_env(level: str):
    # Normalize e.g. "level5" → "Level5"
    if level and level[0].islower():
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


# ── Core episode runner ───────────────────────────────────────────────────────

def run_episode(
    model: NESPolicyModel,
    level: str = "Level1",
    n_steps: int = 2000,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Run one episode and return gameplay metrics + recorded actions.

    Returns dict with:
        actions      : np.ndarray (N, 9) int8
        initial_state: bytes
        steps        : int   — steps survived before death / n_steps
        xscroll      : int   — max horizontal scroll reached
        enemies_hit  : float — total enemy HP damage dealt
        level_up     : bool
    """
    if device is None:
        device = next(model.parameters()).device

    ctx_len  = model.backbone.cfg.n_steps

    env, _, initial_state = make_env(level)

    ram_buf    = torch.zeros(ctx_len, RAM_SIZE, dtype=torch.uint8)
    dpad_buf   = torch.zeros(ctx_len, dtype=torch.long)
    button_buf = torch.zeros(ctx_len, dtype=torch.long)
    buf_len    = 0

    recorded_actions: list[np.ndarray] = []
    max_xscroll  = 0
    enemies_hit  = 0.0
    level_up     = False

    for step in range(n_steps):
        ram = torch.from_numpy(env.unwrapped.get_ram().copy())   # (2048,) uint8

        if buf_len < ctx_len:
            ram_buf[buf_len] = ram
            pos     = buf_len
            buf_len += 1
        else:
            ram_buf    = torch.roll(ram_buf,    -1, dims=0)
            dpad_buf   = torch.roll(dpad_buf,   -1, dims=0)
            button_buf = torch.roll(button_buf, -1, dims=0)
            ram_buf[-1]    = ram
            dpad_buf[-1]   = 0
            button_buf[-1] = 0
            pos = ctx_len - 1

        ram_d = ram_buf.unsqueeze(0).to(device)                  # (1, ctx_len, 2048)

        # Backbone: one pass
        with torch.no_grad():
            action_out_tokens = model.encode(
                ram_d,
                dpad_buf.unsqueeze(0).to(device),
                button_buf.unsqueeze(0).to(device),
            )

        # ActionDecoder pass 1: dpad
        with torch.no_grad():
            dpad_logits, _ = model.decode(
                action_out_tokens,
                dpad_buf.unsqueeze(0).to(device),
                button_buf.unsqueeze(0).to(device),
            )
        pred_dpad = _sample(dpad_logits[0, pos], temperature)

        # ActionDecoder pass 2: button conditioned on pred_dpad
        dpad_buf[pos] = pred_dpad
        with torch.no_grad():
            _, button_logits = model.decode(
                action_out_tokens,
                dpad_buf.unsqueeze(0).to(device),
                button_buf.unsqueeze(0).to(device),
            )
        pred_button = _sample(button_logits[0, pos], temperature)
        button_buf[pos] = pred_button

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
        "level_up":      level_up,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",      default=None,  help="YAML config used to train the checkpoint")
    parser.add_argument("--level",       default="Level1")
    parser.add_argument("--n_steps",     type=int,   default=2000)
    parser.add_argument("--out",         default="tmp/agent_run.npz")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. 0=argmax (deterministic)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}  device={device}")
    model = load_model(args.checkpoint, device, config_path=args.config)

    result = run_episode(model, level=args.level, n_steps=args.n_steps,
                         temperature=args.temperature, device=device)

    status = "LEVEL UP" if result["level_up"] else "FAILED"
    print(f"{status} | steps={result['steps']} | xscroll={result['xscroll']} | enemies_hit={result['enemies_hit']:.0f}")

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


if __name__ == "__main__":
    main()
