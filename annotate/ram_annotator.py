"""
ram_annotator.py — Generate grounded think-text annotations from Contra RAM.

For each chunk of CHUNK actions starting at step t, produce one short "thought"
built from three sources:

  situation  what is on screen at step t (decoded from the pre-action RAM)
  maneuver   what the expert does during the chunk (decoded controller bits)
  outcome    chunk events (hits, pickups, core breaks), restricted to entities
             already visible at step t so the thought never references facts a
             policy could not know at inference time

Each annotation also carries a `facts` dict (visible enemies, raw events,
action stats) for downstream validation and filtering.

Usage
-----
    # preview thoughts for one trace
    python -m vla.datasets.ram_annotator --trace synthetic/clear_trace/win_level1_202603301145.npz --preview 12

    # annotate every trace in a directory
    python -m vla.datasets.ram_annotator --trace_dir synthetic/clear_trace --out_dir synthetic/clear_trace_thoughts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

import stable_retro as retro

from contra.replay import rewind_state, step_env, scan_hit_events, GAME
from contra.game_state import decode_ram
from contra.events import enemy_type_name, scan_events

CHUNK = 10

# Enemy-table entries that are not shootable enemies.
_PROJECTILE_NAMES = {"Bullet", "Mortar Shot", "Basquez Bullet"}
_PICKUP_NAMES = {"Weapon Item", "Flying Capsule", "Falling Capsule"}

# Covers both naming schemes: game_state.WEAPON_NAMES ("Machine", "Spray", ...)
# and events.WEAPON_NAMES ("MachineGun", "Spread", ...) used in pickup details.
_WEAPON_PRETTY = {
    "Regular": "rifle",
    "Machine": "machine gun",
    "MachineGun": "machine gun",
    "Flame": "flamethrower",
    "Flamethrower": "flamethrower",
    "Spray": "spread gun",
    "Spread": "spread gun",
    "Laser": "laser",
}


def _make_env():
    return retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )


def _collect(env, initial_emu_state: bytes, actions: np.ndarray):
    """Replay once, returning per-step pre-action RAM snapshots and events."""
    rewind_state(env, initial_emu_state)
    pre_rams: list[np.ndarray] = []
    events: list[dict] = []
    for t, act in enumerate(actions):
        pre = env.unwrapped.get_ram().copy()
        pre_rams.append(pre)
        step_env(env, act)
        cur = env.unwrapped.get_ram()
        events.extend(scan_events(pre, cur, t))
        events.extend(scan_hit_events(pre, cur, t))
    return pre_rams, events


# ── Situation ──────────────────────────────────────────────────────────────────

def _relation(dx: int, dy: int) -> str:
    if abs(dx) <= 16:
        return "above" if dy < -16 else "below" if dy > 16 else "right on top of you"
    return "ahead" if dx > 0 else "behind"


def _situation(state: dict) -> tuple[str, dict]:
    """Render the on-screen situation at chunk start; also return fact groups."""
    px, py, level = state["player_x"], state["player_y"], state["level"]

    enemy_groups: Counter[tuple[str, str]] = Counter()
    n_projectiles = 0
    pickups: list[str] = []
    for e in state["enemies"]:
        name = enemy_type_name(e["type"], level)
        if name in _PROJECTILE_NAMES:
            n_projectiles += 1
        elif name in _PICKUP_NAMES:
            pickups.append(name)
        elif name.startswith("unknown"):
            continue
        else:
            enemy_groups[(name, _relation(e["x"] - px, e["y"] - py))] += 1

    parts: list[str] = []
    for (name, rel), count in enemy_groups.most_common(3):
        label = name.lower() if count == 1 else f"{count} {name.lower()}s"
        parts.append(f"{label} {rel}")
    if n_projectiles:
        parts.append("incoming fire")
    if pickups:
        parts.append("a power-up capsule in reach")

    weapon = _WEAPON_PRETTY.get(state["weapon"], state["weapon"].lower())
    if state["rapid_fire"]:
        weapon = f"rapid-fire {weapon}"

    text = (", ".join(parts) if parts else "no threats on screen") + f"; {weapon} in hand"
    facts = {
        "enemies": {f"{name} {rel}": count for (name, rel), count in enemy_groups.items()},
        "projectiles": n_projectiles,
        "pickups": pickups,
        "weapon": state["weapon"],
        "in_air": state["in_air"],
    }
    return text, facts


# ── Maneuver ───────────────────────────────────────────────────────────────────

def _maneuver(chunk_acts: np.ndarray, prev_act: np.ndarray) -> tuple[str, dict]:
    """Describe the expert's controller inputs over one chunk."""
    n = len(chunk_acts)
    up, down = int(chunk_acts[:, 4].sum()), int(chunk_acts[:, 5].sum())
    left, right = int(chunk_acts[:, 6].sum()), int(chunk_acts[:, 7].sum())
    fire_held = int(chunk_acts[:, 0].sum())

    jump_bits = np.concatenate([[prev_act[8]], chunk_acts[:, 8]])
    jumps = int(((jump_bits[1:] == 1) & (jump_bits[:-1] == 0)).sum())
    fire_bits = np.concatenate([[prev_act[0]], chunk_acts[:, 0]])
    fire_presses = int(((fire_bits[1:] == 1) & (fire_bits[:-1] == 0)).sum())

    parts: list[str] = []
    if right >= n // 3 and right > left:
        parts.append("push right")
    elif left >= n // 3 and left > right:
        parts.append("fall back left")
    elif down >= n // 2:
        parts.append("stay prone")
    elif up >= n // 2:
        parts.append("climb up")
    else:
        parts.append("hold position")

    if jumps == 1:
        parts.append("jump")
    elif jumps >= 2:
        parts.append(f"jump {jumps} times")

    if up >= n // 3 and parts[0] != "climb up":
        parts.append("aim up")
    elif down >= n // 3 and parts[0] not in ("stay prone",):
        parts.append("aim down")

    if fire_held >= n // 2 or fire_presses >= 2:
        parts.append("keep firing")
    elif fire_held > 0:
        parts.append("fire")

    facts = {"up": up, "down": down, "left": left, "right": right,
             "jumps": jumps, "fire_steps": fire_held}
    return ", ".join(parts), facts


# ── Outcome ────────────────────────────────────────────────────────────────────

def _hit_names(detail: str) -> list[str]:
    """Parse 'Sniper -1.0HP, Soldier -1.0HP' into enemy names."""
    return [part.rsplit(" -", 1)[0].strip() for part in detail.split(",") if " -" in part]


def _outcome(chunk_events: list[dict], visible_names: set[str],
             pickup_visible: bool) -> tuple[str, dict]:
    """Render hindsight results of the chunk, guarded against future leakage."""
    hits: Counter[str] = Counter()
    flags: list[str] = []
    died = False
    for ev in chunk_events:
        tag = ev["tag"]
        if tag in ("REGULAR_ENEMY_HIT", "BOSS_HIT"):
            for name in _hit_names(ev["detail"]):
                if name in visible_names:
                    hits[name] += 1
        elif tag == "GUN_PICKUP" and pickup_visible:
            new_weapon = ev["detail"].split("→")[-1].strip()
            flags.append(f"grab the {_WEAPON_PRETTY.get(new_weapon, new_weapon.lower())}")
        elif tag == "GUN_POWERUP" and pickup_visible:
            flags.append("grab the rapid-fire upgrade")
        elif tag == "CORE_BROKEN":
            flags.append("destroy the core")
        elif tag in ("LEVELUP", "GAME_CLEAR"):
            flags.append("finish the level")
        elif tag == "PLAYER_DIE":
            died = True

    parts = [f"take out the {name.lower()}" for name in list(hits)[:2]] + flags
    facts = {"hits": dict(hits), "flags": flags, "player_died": died,
             "events": [{"step": e["step"], "tag": e["tag"], "detail": e["detail"]}
                        for e in chunk_events]}
    return " and ".join(parts), facts


# ── Annotation ─────────────────────────────────────────────────────────────────

def annotate_trace(trace_path: str, env=None, chunk: int = CHUNK) -> list[dict]:
    """Annotate one trace; returns one dict per full chunk of `chunk` actions."""
    ckpt = np.load(trace_path, allow_pickle=True)
    actions = np.asarray(ckpt["actions"], dtype=np.uint8)
    initial_emu_state = bytes(ckpt["initial_state"])

    own_env = env is None
    if own_env:
        env = _make_env()
        env.reset()
    try:
        pre_rams, events = _collect(env, initial_emu_state, actions)
    finally:
        if own_env:
            env.close()

    annotations = []
    for t in range(0, len(actions) - chunk + 1, chunk):
        state = decode_ram(pre_rams[t])
        visible_names = {enemy_type_name(e["type"], state["level"]) for e in state["enemies"]}
        chunk_events = [e for e in events if t <= e["step"] < t + chunk]

        situation, sit_facts = _situation(state)
        prev_act = actions[t - 1] if t > 0 else np.zeros(9, dtype=np.uint8)
        maneuver, man_facts = _maneuver(actions[t:t + chunk], prev_act)
        outcome, out_facts = _outcome(chunk_events, visible_names,
                                      pickup_visible=bool(sit_facts["pickups"]))

        thought = f"{situation[0].upper()}{situation[1:]}. {maneuver.capitalize()}"
        if outcome:
            thought += f" — {outcome}"
        thought += "."

        annotations.append({
            "t": t,
            "thought": thought,
            "facts": {**sit_facts, **man_facts, **out_facts},
        })
    return annotations


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAM-grounded think annotations")
    parser.add_argument("--trace", default=None, help="One trace NPZ to annotate")
    parser.add_argument("--trace_dir", default=None, help="Annotate every NPZ in this directory")
    parser.add_argument("--out_dir", default="annotate/clear_trace_thoughts")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--preview", type=int, default=0,
                        help="Print this many evenly spaced thoughts per trace")
    args = parser.parse_args()

    if not args.trace and not args.trace_dir:
        parser.error("provide --trace or --trace_dir")
    paths = [args.trace] if args.trace else sorted(glob.glob(os.path.join(args.trace_dir, "*.npz")))

    os.makedirs(args.out_dir, exist_ok=True)
    env = _make_env()
    env.reset()
    try:
        for path in paths:
            annotations = annotate_trace(path, env=env, chunk=args.chunk)
            out_path = os.path.join(args.out_dir, Path(path).stem + ".json")
            with open(out_path, "w") as f:
                json.dump({"trace": os.path.basename(path), "chunk": args.chunk,
                           "annotations": annotations}, f, indent=1)
            n_died = sum(a["facts"]["player_died"] for a in annotations)
            print(f"{os.path.basename(path)}: {len(annotations)} chunks "
                  f"({n_died} with a death) → {out_path}")

            if args.preview:
                idx = np.linspace(0, len(annotations) - 1, args.preview, dtype=int)
                for i in idx:
                    a = annotations[i]
                    print(f"  t={a['t']:4d}  {a['thought']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
