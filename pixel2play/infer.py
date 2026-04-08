"""
pixel2play inference — run the trained policy on bc_data recordings.

For each recording, the model predicts actions from precomputed image features
(with or without text embeddings) using a sliding window of T=200 frames.
Predicted actions are saved as .npz files replayable by run_npz.py.

Usage:
    python pixel2play/infer.py --checkpoint .tmp/checkpoints/nes_policy/ckpt-00001000.ckpt
    python pixel2play/infer.py --checkpoint <ckpt> --no-text
    python pixel2play/infer.py --checkpoint <ckpt> --recording win_level1_202603301145
"""

import argparse
import os
import sys
import re

import time
import numpy as np
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*Gym.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stable_retro as retro
from contra.replay import rewind_state, replay_actions

from pixel2play.model.backbone import BackboneConfig
from pixel2play.model.nes_policy import NESPolicyModel
from pixel2play.train import NESLightningModule

# NES button order: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Build lookup tables matching the 9-class dpad and 4-class button encoding in nes_actions.py
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

GAME      = "Contra-Nes"
STATE_DIR = "contra/start_states"
DATA_ROOT = "annotate/bc_data/Contra-Nes"


def _level_from_name(name: str) -> int:
    m = re.search(r"level(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else 1


def _load_initial_state(level: int) -> bytes:
    state_label = f"Level{level}"
    env = retro.make(
        game=GAME, state=state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.ALL,
    )
    spread = os.path.join(STATE_DIR, f"{state_label}.state")
    if level > 1 and os.path.exists(spread):
        with open(spread, "rb") as f:
            env.initial_state = f.read()
    env.reset()
    state = env.em.get_state()
    env.close()
    return state


def _load_text(rec_dir: str, n: int, fps: float) -> np.ndarray:
    """Load per-frame text embeddings with duration propagation."""
    from annotate.proto.video_annotation_pb2 import VideoAnnotation
    _TEXT_MODEL = "gemini-3-flash-preview"
    _TEXT_DIM   = 768

    va = VideoAnnotation()
    with open(os.path.join(rec_dir, "annotation.proto"), "rb") as f:
        va.ParseFromString(f.read())

    text = np.zeros((n, 1, _TEXT_DIM), dtype=np.float32)
    fas  = va.frame_annotations[1:]
    for i, fa in enumerate(fas):
        if i >= n:
            break
        if fa.frame_text_annotation:
            fta = fa.frame_text_annotation[0]
            gem = fta.text_embedding_dict.get(_TEXT_MODEL, None)
            if gem is not None:
                emb = gem.text_embeddings.get("gemma", None)
                if emb is not None and emb.values:
                    vec  = np.array(emb.values, dtype=np.float32).reshape(1, 1, _TEXT_DIM)
                    span = round(fta.duration * fps)
                    text[i:i + span] = vec
    return text  # (N, 1, 768)


@torch.inference_mode()
def infer_recording(
    model: NESPolicyModel,
    npz_path: str,
    use_text: bool,
    device: torch.device,
    T: int = 200,
    n_text_tokens: int = 1,
) -> tuple[np.ndarray, int]:
    """Run sliding-window autoregressive inference on one recording.

    Returns (predicted_actions (N, 9) uint8, exact_match_count).
    """
    data = np.load(npz_path, mmap_mode="r")
    img_all = torch.from_numpy(data["img_features"].astype(np.float32))  # (N, 1, D)
    N = len(data["dpad"])

    # gt actions from npz
    gt_dpad   = data["dpad"].astype(np.int64)    # (N,)
    gt_button = data["button"].astype(np.int64)  # (N,)
    gt_all = np.array([_DPAD_NES[d] | _BUTTON_NES[b] for d, b in zip(gt_dpad, gt_button)], dtype=np.uint8)

    if use_text and n_text_tokens > 0 and "text" in data:
        text_all = torch.from_numpy(data["text"][:, :n_text_tokens, :].astype(np.float32))
    else:
        text_all = torch.zeros(N, n_text_tokens, 768, dtype=torch.float32)

    # Rolling buffers (zero-padded at start)
    img_buf    = torch.zeros(T, 1, img_all.shape[-1])
    text_buf   = torch.zeros(T, n_text_tokens, 768)
    dpad_buf   = torch.zeros(T, dtype=torch.long)
    button_buf = torch.zeros(T, dtype=torch.long)

    predicted_actions = np.zeros((N, 9), dtype=np.uint8)
    exact_matches     = 0
    FPS               = 20.0

    t0 = time.perf_counter()
    for t in tqdm(range(N), desc=os.path.basename(npz_path), leave=False):
        img_buf    = torch.roll(img_buf,    -1, dims=0); img_buf[-1]    = img_all[t]
        text_buf   = torch.roll(text_buf,   -1, dims=0); text_buf[-1]   = text_all[t]
        dpad_buf   = torch.roll(dpad_buf,   -1, dims=0)
        button_buf = torch.roll(button_buf, -1, dims=0)

        frames  = img_buf.unsqueeze(0).to(device)
        text_in = text_buf.unsqueeze(0).to(device)
        dpad_in = dpad_buf.unsqueeze(0).to(device)
        btn_in  = button_buf.unsqueeze(0).to(device)

        dpad_logits, button_logits = model(frames, dpad_in, btn_in, text_in)

        dpad_pred   = dpad_logits[0, -1].argmax().item()
        button_pred = button_logits[0, -1].argmax().item()

        dpad_buf[-1]   = dpad_pred
        button_buf[-1] = button_pred

        predicted_actions[t] = _DPAD_NES[dpad_pred] | _BUTTON_NES[button_pred]
        if np.array_equal(predicted_actions[t], gt_all[t]):
            exact_matches += 1

    infer_time   = time.perf_counter() - t0
    game_time    = N / FPS
    ms_per_step  = infer_time / N * 1000

    return predicted_actions, exact_matches, N, infer_time, game_time, ms_per_step


def main():
    parser = argparse.ArgumentParser(description="pixel2play inference")
    parser.add_argument("--checkpoint",  required=True, help="Path to .ckpt file")
    parser.add_argument("--recording",   default=None,  help="Single recording name; default: all")
    parser.add_argument("--no-text",     action="store_true", help="Zero out text embeddings")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from Lightning checkpoint
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "nes_150M.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    n_steps = cfg["shared"]["n_seq_timesteps"]
    lr      = cfg["stage3_finetune"]["optim"]["learning_rate"]

    n_text_tokens = cfg["shared"].get("text_tokenizer_config", {}).get("text_embedding_shape", [1, 768])[0]

    backbone_cfg = BackboneConfig(n_steps=n_steps, n_text_tokens=n_text_tokens)
    module = NESLightningModule(backbone_cfg, lr=lr)

    ckpt       = torch.load(args.checkpoint, map_location=device)
    state_dict = {
        k.replace("model._orig_mod.", "model."): v
        for k, v in ckpt["state_dict"].items()
    }
    module.load_state_dict(state_dict)
    model = torch.compile(module.model.eval().to(device))
    model.backbone._build_block_masks()

    # Recordings to process
    if args.recording:
        if os.path.isabs(args.recording) or os.path.exists(args.recording):
            rec_paths = [args.recording]
        else:
            name = args.recording if args.recording.endswith(".npz") else args.recording + ".npz"
            rec_paths = [os.path.join(DATA_ROOT, name)]
    else:
        rec_paths = sorted([
            os.path.join(DATA_ROOT, f)
            for f in os.listdir(DATA_ROOT)
            if f.endswith(".npz")
        ])

    # Warmup: trigger Triton compilation before timing
    print("Warming up...", flush=True)
    dummy_img  = torch.zeros(1, n_steps, 1, 1024, device=device)
    dummy_act  = torch.zeros(1, n_steps, dtype=torch.long, device=device)
    dummy_text = torch.zeros(1, n_steps, n_text_tokens, 768, device=device)
    with torch.inference_mode():
        model(dummy_img, dummy_act, dummy_act, dummy_text)
    print("Warmup done.\n", flush=True)

    use_text = not args.no_text
    total    = 0
    non_lose = 0

    for rec_path in rec_paths:
        name  = os.path.basename(rec_path)
        level = _level_from_name(name)
        print(f"\n[{name}]  level={level}  text={'on' if use_text else 'off'}")

        actions, exact_matches, n_frames, infer_time, game_time, ms_per_step = infer_recording(model, rec_path, use_text=use_text, device=device, T=n_steps, n_text_tokens=n_text_tokens)
        initial_state = _load_initial_state(level)

        result = replay_actions(
            actions,
            initial_state=initial_state,
            level=f"Level{level}",
            verbose=True,
        )

        outcome = result["result"]
        total  += 1
        if outcome != "lose":
            non_lose += 1
        realtime_ratio = game_time / infer_time
        print(f"  → {outcome}"
              f"  action_match={exact_matches}/{n_frames} ({exact_matches/n_frames:.1%})"
              f"  infer={infer_time:.1f}s  game={game_time:.1f}s  {ms_per_step:.1f}ms/step  "
              f"({'%.1fx realtime' % realtime_ratio if realtime_ratio >= 1 else 'SLOWER than realtime (%.2fx)' % realtime_ratio})")

    print(f"\n{'='*40}")
    print(f"  text={'on' if use_text else 'off'}  {non_lose}/{total} non-lose  rate={non_lose/total:.1%}")


if __name__ == "__main__":
    main()
