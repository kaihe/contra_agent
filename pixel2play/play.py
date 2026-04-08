"""
pixel2play agent play — run the trained policy live in the emulator.

The agent plays until it loses a life (or the level is cleared / game is won).
The action sequence is saved to a .npz file and replayed to confirm the outcome
and optionally render a video.

Usage:
    python pixel2play/play.py --checkpoint tmp/checkpoints/nes_policy/ckpt-00001000.ckpt
    python pixel2play/play.py --checkpoint <ckpt> --level 2 --video out/play.mp4
    python pixel2play/play.py --checkpoint <ckpt> --max-steps 3000
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", message=".*Gym.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stable_retro as retro
from tqdm import tqdm

from contra.replay import rewind_state, replay_actions, save_video, SKIP, FPS, GAME
from contra.events import EV_PLAYER_DIE, EV_LEVELUP, EV_GAME_CLEAR, scan_events

from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_policy import NESPolicyModel
from pixel2play.train import NESLightningModule

import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "nes_150M.yaml")
STATE_DIR   = "contra/integration/Contra-Nes"

# NES button order: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Matches DPAD_ENCODING and BUTTON_ENCODING in nes_actions.py
_DPAD_NES = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1: left
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2: right
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: down
    [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 5: up-left
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 6: up-right
    [0, 0, 0, 0, 0, 1, 1, 0, 0],  # 7: down-left
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 8: down-right
], dtype=np.uint8)
_BUTTON_NES = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1: A (jump)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: B (fire)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: A+B
], dtype=np.uint8)

_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess_frame(frame: np.ndarray, size: int = 192) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 1, 3, size, size) float32 normalised, ready for tokenizer."""
    img = frame.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3, H, W)
    img = (img - _IMG_MEAN) / _IMG_STD
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)         # (1, 1, 3, H, W)
    if t.shape[-2:] != (size, size):
        t = torch.nn.functional.interpolate(
            t.squeeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).unsqueeze(0)
    return t


def _load_initial_state(level: int) -> bytes:
    level_label = f"Level{level}"
    env = retro.make(
        game=GAME, state=level_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    spread = os.path.join(STATE_DIR, f"{level_label}.state")
    if level > 1 and os.path.exists(spread):
        with open(spread, "rb") as f:
            env.initial_state = f.read()
    env.reset()
    state = env.em.get_state()
    env.close()
    return state


@torch.inference_mode()
def play(
    model: NESPolicyModel,
    tokenizer,
    initial_state: bytes,
    level: int,
    device: torch.device,
    T: int = 200,
    n_text_tokens: int = 0,
    max_steps: int = 3_000,
    temperature: float = 0.0,
) -> tuple[np.ndarray, bytes, str]:
    """Run the agent live in the emulator.

    Returns (actions (N, 9) uint8, initial_state bytes, outcome str).
    Each logical step = SKIP=3 NES frames; the model sees the first frame.
    The first T-1=199 steps use zero-padded image and action history.
    """
    level_label = f"Level{level}"
    env = retro.make(
        game=GAME, state=level_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial_state)

    # Capture the very first frame (nes_0, before any action)
    first_frame = env.em.get_screen().copy()

    # Rolling feature + action buffers (zero-padded, T=200 history)
    feat_dim   = tokenizer.embed_dim
    n_img      = tokenizer.n_tokens
    feat_buf   = torch.zeros(T, n_img, feat_dim, device=device)
    text_buf   = torch.zeros(T, max(n_text_tokens, 0), 768, device=device) if n_text_tokens > 0 else torch.zeros(T, 0, 768, device=device)
    dpad_buf   = torch.zeros(T, dtype=torch.long, device=device)
    button_buf = torch.zeros(T, dtype=torch.long, device=device)

    all_actions: list[np.ndarray] = []
    outcome      = "lose"

    # pending_frame holds the frame to be placed at the start of each step,
    # matching infer.py: roll → place frame → predict → store pred → step env.
    pending_frame = first_frame

    for step in tqdm(range(max_steps), desc=f"Level{level}", unit="step"):
        # 1. Roll buffers; new T-1 slot starts zeroed for dpad/button
        feat_buf   = torch.roll(feat_buf,   -1, dims=0)
        text_buf   = torch.roll(text_buf,   -1, dims=0)
        dpad_buf   = torch.roll(dpad_buf,   -1, dims=0)
        button_buf = torch.roll(button_buf, -1, dims=0)

        # 2. Place current frame at T-1; action slot stays 0 (unknown, to be predicted)
        frame_t = _preprocess_frame(pending_frame).to(device)
        feat_buf[-1]   = tokenizer(frame_t).squeeze(0).squeeze(0)
        dpad_buf[-1]   = 0
        button_buf[-1] = 0

        # 3. Predict from buffer
        dpad_logits, button_logits = model(
            feat_buf.unsqueeze(0),
            dpad_buf.unsqueeze(0),
            button_buf.unsqueeze(0),
            text_buf.unsqueeze(0),
        )
        dpad_logits_step = dpad_logits[0, -1]
        button_logits_step = button_logits[0, -1]
        
        if temperature > 0.0:
            dpad_pred = torch.distributions.Categorical(logits=dpad_logits_step / temperature).sample().item()
            button_pred = torch.distributions.Categorical(logits=button_logits_step / temperature).sample().item()
        else:
            dpad_pred   = dpad_logits_step.argmax().item()
            button_pred = button_logits_step.argmax().item()

        # 4. Store prediction at T-1 — becomes past context on next roll
        dpad_buf[-1]   = dpad_pred
        button_buf[-1] = button_pred

        nes_action = (_DPAD_NES[dpad_pred] | _BUTTON_NES[button_pred]).astype(np.uint8)
        all_actions.append(nes_action)

        # 5. Step env SKIP=3 NES frames; first frame is next model input
        pre_ram = env.unwrapped.get_ram().copy()
        for i in range(SKIP):
            obs, _, _, _, _ = env.step(nes_action.copy())
            if i == 0:
                pending_frame = obs.copy()
        curr_ram = env.unwrapped.get_ram().copy()

        # Check terminal events
        if EV_PLAYER_DIE.trigger(pre_ram, curr_ram):
            print(f"  step {step}  PLAYER_DIE  lives {int(pre_ram[50])} → {int(curr_ram[50])}")
            outcome = "lose"
            break
        if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
            game_cleared = True
            outcome = "game_clear"
            break
        if EV_LEVELUP.trigger(pre_ram, curr_ram):
            leveled_up = True
            outcome = "level_up"
            break

    env.close()

    actions = np.stack(all_actions)  # (N, 9) uint8
    return actions, outcome


def main():
    parser = argparse.ArgumentParser(description="pixel2play agent play")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--level",      type=int, default=1, help="Level to play (default: 1)")
    parser.add_argument("--max-steps",  type=int, default=1_000, help="Max logical steps")
    parser.add_argument("--video",      default=None, help="Save replay MP4 to this path")
    parser.add_argument("--out",        default="tmp/play_actions.npz", help="Save actions to this .npz")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling actions (0 for argmax)")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg           = yaml.safe_load(open(CONFIG_FILE))
    n_steps       = cfg["shared"]["n_seq_timesteps"]
    lr            = cfg["stage3_finetune"]["optim"]["learning_rate"]
    n_text_tokens = cfg["shared"].get("text_tokenizer_config", {}).get("text_embedding_shape", [1, 768])[0]

    backbone_cfg = BackboneConfig(n_steps=n_steps, n_text_tokens=n_text_tokens)
    module       = NESLightningModule(backbone_cfg, lr=lr)

    ckpt       = torch.load(args.checkpoint, map_location=device)
    state_dict = {
        k.replace("model._orig_mod.", "model."): v
        for k, v in ckpt["state_dict"].items()
    }
    module.load_state_dict(state_dict)
    raw_model = module.model.eval().to(device)
    raw_model.backbone._build_block_masks()
    tokenizer = raw_model.backbone.image_tokenizer.eval().to(device)
    model     = torch.compile(raw_model)

    # Warmup
    print("Warming up...", flush=True)
    dummy_feat = torch.zeros(1, n_steps, 1, 1024, device=device)
    dummy_act  = torch.zeros(1, n_steps, dtype=torch.long, device=device)
    dummy_text = torch.zeros(1, n_steps, n_text_tokens, 768, device=device)
    model(dummy_feat, dummy_act, dummy_act, dummy_text)
    print("Warmup done.\n", flush=True)

    initial_state = _load_initial_state(args.level)

    t0 = time.perf_counter()
    actions, outcome = play(
        model, tokenizer, initial_state,
        level=args.level,
        device=device,
        T=n_steps,
        n_text_tokens=n_text_tokens,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - t0

    n = len(actions)
    print(f"\n  outcome={outcome}  steps={n}  "
          f"elapsed={elapsed:.1f}s  ({n / FPS:.1f}s game-time)  "
          f"{elapsed / n * 1000:.1f}ms/step")

    # Save action sequence
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, actions=actions, initial_state=np.frombuffer(initial_state, dtype=np.uint8),
             level=f"Level{args.level}", outcome=outcome)
    print(f"  actions saved → {args.out}")

    # Replay to confirm outcome and optionally render video
    print("\nReplaying to verify...")
    want_video = bool(args.video)
    result = replay_actions(
        actions, initial_state=initial_state,
        level=f"Level{args.level}",
        want_video=want_video,
        verbose=True,
    )
    print(f"  replay outcome: {result['result']}")

    if want_video and result["video"] is not None:
        save_video(result["video"], args.video)
        print(f"  video saved → {args.video}")


if __name__ == "__main__":
    main()
