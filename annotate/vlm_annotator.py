"""
vlm_annotator.py — Generate vision-grounded movement plans from Contra frames.

The portable counterpart to ram_annotator.py. For each chunk of CHUNK actions
starting at step t, we replay the trace, show the VLM the chunk's frames as the
play unfolds (F[t .. t+CHUNK]), and ask it to narrate the movement plan in one
short sentence. The plan is the manufactured label for a think+act VLA sample:

    [ CHUNK input frames ] -> [ text plan ] -> [ CHUNK actions ]

Annotation is frames-only (no controller inputs in the prompt) and zero-shot —
see annotate/VLM_ANNOTATOR_DESIGN.md. The real controller inputs are decoded to
readable tokens and stored in the output JSON for inspection/validation only.

Requires the VLM service from annotate/serve_vlm.sh (OpenAI-compatible :8000).

Usage
-----
    # preview plans for one trace
    python -m annotate.vlm_annotator --trace synthetic/clear_trace/win_level1_202603301145.npz --preview 12

    # annotate every trace in a directory
    python -m annotate.vlm_annotator --trace_dir synthetic/clear_trace --out_dir annotate/clear_trace_plans
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", message=".*Gym.*")

import stable_retro as retro

from contra.replay import GAME, SKIP, rewind_state
from contra.inputs import DPAD_NAMES, BUTTON_NAMES

CHUNK = 8

# IPv4 literal, not "localhost": vLLM binds 0.0.0.0 (IPv4) but localhost often
# resolves to ::1 (IPv6) first, which refuses the connection. Under WSL2
# mirrored networking even 127.0.0.1 is refused for a 0.0.0.0-bound server, so
# we also probe the machine's primary IP (see _connect_client).
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "annotator"

SYSTEM_PROMPT = (
    "You are an expert Contra (NES) player calling your next move. You are shown a "
    "short burst of consecutive game frames in chronological order; the player is "
    "the small human soldier and left-to-right is forward. Report ONLY what affects "
    "the next decision, in EXACTLY two short sentences, one per line, no labels:\n"
    "  1. Threats: enemies and where they are, incoming bullets/projectiles to "
    "dodge, and any pit, gap or edge you could fall into. Write 'path is clear' if "
    "there is nothing to react to.\n"
    "  2. Plan: where to run or jump and what to shoot to advance safely.\n"
    "Do NOT describe scenery or decoration — grass, mountains, sky, water colour, "
    "background structures, ground texture. Mention terrain only when it is a hazard "
    "that changes the move (a gap to jump, an edge not to fall off). Be concrete "
    "about positions (left/right/above/below); describe only what is visible; do not "
    "invent enemies, bullets or shooting. Keep each sentence under 20 words."
)

USER_INSTRUCTION = (
    "These {n} frames are consecutive moments of one short move (left to right is "
    "forward). Call out the threats to react to — enemies, incoming bullets, and any "
    "pit/gap to avoid — and the resulting move (where to go or jump, what to shoot). "
    "Ignore background scenery."
)

# Appended to the user turn when --action_hint is on. Grounds only the Move
# sentence in the real controller inputs; Terrain/Threats stay pure-vision.
ACTION_HINT_INSTRUCTION = (
    " For reference, the player's actual controller inputs across these frames "
    "were: {summary}. Make the Plan sentence consistent with these inputs — the "
    "player only advances rightward, and do NOT describe shooting if the player "
    "is not firing."
)


def _make_env():
    return retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )


def _collect_frames(env, initial_emu_state: bytes, actions: np.ndarray) -> list[np.ndarray]:
    """Replay once, returning F[0..N] where F[i] is the screen before action i."""
    rewind_state(env, initial_emu_state)
    frames = [env.em.get_screen().copy()]  # F[0]: state before action 0
    for act in actions:
        act_arr = np.asarray(act, dtype=np.uint8)
        for i in range(SKIP):
            obs, _, _, _, _ = env.step(act_arr.copy())
            if i == 0:
                frames.append(obs.copy())  # F[t+1]: first NES frame after action t
    return frames


# ── Action decoding (output JSON only; not shown to the VLM) ─────────────────────

def _action_token(act: np.ndarray) -> str:
    """Render one NES MultiBinary(9) action as e.g. 'R+F', 'UR+FJ', '_+_'."""
    up, down, left, right = int(act[4]), int(act[5]), int(act[6]), int(act[7])
    dpad_idx = {
        (0, 0, 0, 0): 0, (0, 0, 1, 0): 1, (0, 0, 0, 1): 2, (1, 0, 0, 0): 3,
        (0, 1, 0, 0): 4, (1, 0, 1, 0): 5, (1, 0, 0, 1): 6, (0, 1, 1, 0): 7,
        (0, 1, 0, 1): 8,
    }.get((up, down, left, right), 0)
    btn_idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(int(act[8]), int(act[0]))]
    return f"{DPAD_NAMES[dpad_idx]}+{BUTTON_NAMES[btn_idx]}"


def _action_summary(chunk_acts: np.ndarray, prev_act: np.ndarray) -> str:
    """Factual one-line gloss of the chunk's controller inputs (for the hint)."""
    n = len(chunk_acts)
    up, down = int(chunk_acts[:, 4].sum()), int(chunk_acts[:, 5].sum())
    left, right = int(chunk_acts[:, 6].sum()), int(chunk_acts[:, 7].sum())
    fire_steps = int(chunk_acts[:, 0].sum())
    jump_bits = np.concatenate([[prev_act[8]], chunk_acts[:, 8]])
    jumps = int(((jump_bits[1:] == 1) & (jump_bits[:-1] == 0)).sum())

    parts: list[str] = []
    if right:
        parts.append(f"holding Right {right}/{n}")
    elif left:
        parts.append(f"holding Left {left}/{n}")
    else:
        parts.append("not moving horizontally")
    if up >= n // 3:
        parts.append(f"aiming Up {up}/{n}")
    elif down >= n // 3:
        parts.append(f"aiming/crouching Down {down}/{n}")
    parts.append(f"{jumps} jump(s)" if jumps else "no jump")
    parts.append(f"firing {fire_steps}/{n} steps" if fire_steps else "not firing")
    return ", ".join(parts)


# ── VLM call ─────────────────────────────────────────────────────────────────────

def _frame_to_data_url(frame: np.ndarray, scale: int) -> str:
    img = Image.fromarray(frame)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _plan_for_chunk(client, model, frames: list[np.ndarray], scale: float,
                    temperature: float, max_tokens: int,
                    frequency_penalty: float, action_hint: str | None = None) -> str:
    instruction = USER_INSTRUCTION.format(n=len(frames))
    if action_hint:
        instruction += ACTION_HINT_INSTRUCTION.format(summary=action_hint)
    content = [{"type": "text", "text": instruction}]
    for f in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": _frame_to_data_url(f, scale)}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    text = resp.choices[0].message.content.strip()
    # Collapse the three lines into one space-joined plan string.
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


# ── Annotation ─────────────────────────────────────────────────────────────────

def annotate_trace(trace_path: str, client, *, env=None, model: str = DEFAULT_MODEL,
                   chunk: int = CHUNK, scale: int = 3, temperature: float = 0.2,
                   max_tokens: int = 110, frequency_penalty: float = 0.4,
                   action_hint: bool = False, concurrency: int = 8,
                   limit: int = 0) -> list[dict]:
    """Annotate one trace; one plan per full chunk of `chunk` actions.

    `limit` > 0 annotates only the first `limit` chunks (for quick previews).
    """
    ckpt = np.load(trace_path, allow_pickle=True)
    actions = np.asarray(ckpt["actions"], dtype=np.uint8)
    initial_emu_state = bytes(ckpt["initial_state"])

    own_env = env is None
    if own_env:
        env = _make_env()
        env.reset()
    try:
        frames = _collect_frames(env, initial_emu_state, actions)
    finally:
        if own_env:
            env.close()

    starts = list(range(0, len(actions) - chunk + 1, chunk))
    if limit:
        starts = starts[:limit]

    def _one(t: int) -> dict:
        hint = None
        if action_hint:
            prev_act = actions[t - 1] if t > 0 else np.zeros(9, dtype=np.uint8)
            hint = _action_summary(actions[t:t + chunk], prev_act)
        plan = _plan_for_chunk(client, model, frames[t:t + chunk + 1],
                               scale, temperature, max_tokens, frequency_penalty, hint)
        return {"t": t, "plan": plan,
                "actions": [_action_token(a) for a in actions[t:t + chunk]]}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        annotations = list(pool.map(_one, starts))
    return annotations


# ── Server connection ────────────────────────────────────────────────────────

def _primary_ip() -> str | None:
    """The machine's primary outbound IPv4 (no packets sent), or None."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _connect_client(base_url: str):
    """Return an OpenAI client pointed at a reachable server.

    Probes the given base_url first; if it was left at the default and is
    unreachable (the WSL2 mirrored-networking case, where a 0.0.0.0-bound
    server is not reachable over 127.0.0.1), retries on the primary IP.
    """
    from openai import OpenAI

    candidates = [base_url]
    if base_url == DEFAULT_BASE_URL and (ip := _primary_ip()):
        candidates.append(f"http://{ip}:8000/v1")

    last_err: Exception | None = None
    for url in candidates:
        client = OpenAI(base_url=url, api_key="EMPTY")
        try:
            client.models.list()
            if url != base_url:
                print(f"  {base_url} unreachable; using {url}")
            return client
        except Exception as e:  # noqa: BLE001 — report whatever the SDK raised
            last_err = e

    raise SystemExit(
        f"Cannot reach the VLM server (tried {', '.join(candidates)}; "
        f"{type(last_err).__name__}). Is annotate/serve_vlm.sh running? "
        f"Pass --base_url http://<ip>:8000/v1 to point elsewhere."
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VLM movement-plan annotations")
    parser.add_argument("--trace", default=None, help="One trace NPZ to annotate")
    parser.add_argument("--trace_dir", default=None, help="Annotate every NPZ in this directory")
    parser.add_argument("--out_dir", default="tmp/vla/data")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scale", type=int, default=3, help="Integer upscale of frames sent to the VLM")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=110)
    parser.add_argument("--frequency_penalty", type=float, default=0.4)
    parser.add_argument("--action_hint", action="store_true",
                        help="Feed the Move sentence a factual button summary (anti-hallucination)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Annotate only the first N chunks (0 = all)")
    parser.add_argument("--preview", type=int, default=0,
                        help="Print this many evenly spaced plans per trace")
    args = parser.parse_args()

    if not args.trace and not args.trace_dir:
        parser.error("provide --trace or --trace_dir")
    paths = [args.trace] if args.trace else sorted(glob.glob(os.path.join(args.trace_dir, "*.npz")))

    client = _connect_client(args.base_url)

    os.makedirs(args.out_dir, exist_ok=True)
    env = _make_env()
    env.reset()
    try:
        for path in paths:
            out_path = os.path.join(args.out_dir, Path(path).stem + ".json")
            if os.path.exists(out_path):
                print(f"{os.path.basename(path)}: exists, skipping → {out_path}")
                continue

            annotations = annotate_trace(
                path, client, env=env, model=args.model, chunk=args.chunk,
                scale=args.scale, temperature=args.temperature,
                max_tokens=args.max_tokens, frequency_penalty=args.frequency_penalty,
                action_hint=args.action_hint, concurrency=args.concurrency,
                limit=args.limit,
            )

            with open(out_path, "w") as f:
                json.dump({"trace": os.path.basename(path), "chunk": args.chunk,
                           "model": args.model, "annotations": annotations}, f, indent=1)
            print(f"{os.path.basename(path)}: {len(annotations)} plans → {out_path}")

            if args.preview:
                idx = np.linspace(0, len(annotations) - 1, min(args.preview, len(annotations)), dtype=int)
                for i in idx:
                    a = annotations[i]
                    print(f"  t={a['t']:4d}  {a['plan']}")
                    print(f"            actions: {' '.join(a['actions'])}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
