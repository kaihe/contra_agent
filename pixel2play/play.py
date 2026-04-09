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
import torch.nn.functional as F

from contra.events import EV_PLAYER_DIE, ADDR_XSCROLL, EV_ENEMY_HIT, EV_LEVELUP
from contra.inputs import DPAD_TABLE, BUTTON_TABLE
from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_policy import NESPolicyModel
from pixel2play.train import NESLightningModule

# ── Constants ─────────────────────────────────────────────────────────────────

GAME              = "Contra-Nes"
SPREAD_STATES_DIR = os.path.join(os.path.dirname(__file__), "../contra/spread_gun_states")

N_STEPS   = 200       # model context window
FRAME_H   = FRAME_W = 192
EMBED_DIM = 1024
EMU_SKIP  = 3         # env steps per agent step (20 Hz agent @ 60 Hz NES)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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

def load_model(checkpoint_path: str, device: torch.device) -> NESPolicyModel:
    cfg = BackboneConfig(n_steps=N_STEPS, n_text_tokens=0, dropout=0.0)
    module = NESLightningModule(cfg, lr=1e-4)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    # torch.compile() adds '_orig_mod.' to keys — strip it if present
    fixed = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    module.load_state_dict(fixed, strict=True)

    model = module.model.to(device).eval()
    model.backbone._build_block_masks()
    return model


# ── Environment setup ─────────────────────────────────────────────────────────

def make_env(level: str):
    env = retro.make(
        game=GAME, state=level,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    obs, _ = env.reset()

    if level != "Level1":
        path = os.path.join(SPREAD_STATES_DIR, f"{level}.state")
        for opener in (gzip.open, open):
            try:
                with opener(path, "rb") as f:
                    state_bytes = f.read()
                env.initial_state = state_bytes
                obs, _ = env.reset()
                break
            except OSError:
                continue

    initial_state = env.em.get_state()
    return env, obs, initial_state


# ── Frame tokenisation ────────────────────────────────────────────────────────

def tokenize(obs: np.ndarray, tokenizer: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, EMBED_DIM) float32 on CPU."""
    frame = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0   # (3, H, W)
    frame = (frame.unsqueeze(0) - _MEAN) / _STD                       # (1, 3, H, W)
    frame = F.interpolate(frame, size=(FRAME_H, FRAME_W), mode="bilinear", align_corners=False)
    with torch.no_grad():
        feat = tokenizer(frame.unsqueeze(1).to(device))                # (1, 1, 1, D)
    return feat[0, 0].cpu().float()                                     # (1, D)


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

    tokenizer = model.backbone.image_tokenizer
    env, obs, initial_state = make_env(level)

    img_buf    = torch.zeros(N_STEPS, 1, EMBED_DIM)
    dpad_buf   = torch.zeros(N_STEPS, dtype=torch.long)
    button_buf = torch.zeros(N_STEPS, dtype=torch.long)
    text_buf   = torch.zeros(N_STEPS, 0, 768)
    buf_len    = 0

    recorded_actions: list[np.ndarray] = []
    max_xscroll  = 0
    enemies_hit  = 0.0
    level_up     = False

    for step in range(n_steps):
        feat = tokenize(obs, tokenizer, device)

        if buf_len < N_STEPS:
            img_buf[buf_len] = feat
            pos     = buf_len
            buf_len += 1
        else:
            img_buf    = torch.roll(img_buf,    -1, dims=0)
            dpad_buf   = torch.roll(dpad_buf,   -1, dims=0)
            button_buf = torch.roll(button_buf, -1, dims=0)
            img_buf[-1]    = feat
            dpad_buf[-1]   = 0
            button_buf[-1] = 0
            pos = N_STEPS - 1

        frames_d = img_buf.unsqueeze(0).to(device)
        text_d   = text_buf.unsqueeze(0).to(device)

        # Backbone: one pass
        with torch.no_grad():
            action_out_tokens = model.encode(
                frames_d,
                dpad_buf.unsqueeze(0).to(device),
                button_buf.unsqueeze(0).to(device),
                text_d,
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
        for i in range(EMU_SKIP):
            frame, _, terminated, truncated, _ = env.step(action.copy())
            if i == 0:
                obs = frame
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
    parser.add_argument("--level",       default="Level1")
    parser.add_argument("--n_steps",     type=int,   default=2000)
    parser.add_argument("--out",         default="tmp/agent_run.npz")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. 0=argmax (deterministic)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}  device={device}")
    model = load_model(args.checkpoint, device)

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
