"""Build the Contra VLA behavior-cloning dataset from mc_trace files."""

from __future__ import annotations

import argparse
import glob
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stable_retro as retro
from PIL import Image

from contra.events import EV_GAME_CLEAR, EV_LEVELUP
from contra.game_state import state_from_ram
from contra.replay import GAME, SKIP, rewind_state

IMG_SIZE = 192
SHARD_SIZE = 32_768
ACTION_DIM = 36
JPG_QUALITY = 90
SHUFFLE_SEED = 0
TRACE_DIR = Path("synthetic/mc_trace")

# NES MultiBinary(9): [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
BUTTON_TO_ID = {
    (0, 0): 0,  # no A/B
    (1, 0): 1,  # A
    (0, 1): 2,  # B
    (1, 1): 3,  # A + B
}

# 9 D-pad states x 4 button states = 36 lossless controller actions.
DPAD_TO_ID = {
    (0, 0, 0, 0): 0,  # neutral
    (0, 0, 1, 0): 1,  # left
    (0, 0, 0, 1): 2,  # right
    (1, 0, 0, 0): 3,  # up
    (0, 1, 0, 0): 4,  # down
    (1, 0, 1, 0): 5,  # up-left
    (1, 0, 0, 1): 6,  # up-right
    (0, 1, 1, 0): 7,  # down-left
    (0, 1, 0, 1): 8,  # down-right
}

ID_TO_DPAD = {value: key for key, value in DPAD_TO_ID.items()}
ID_TO_BUTTON = {value: key for key, value in BUTTON_TO_ID.items()}


@dataclass
class Sample:
    frame_jpg: bytes
    state: np.ndarray
    action: int
    trace_path: str
    timestep: int
    level_id: int
    outcome: str
    raw_action: np.ndarray


def encode_action36(action: np.ndarray) -> int:
    """Encode one NES MultiBinary action into the 36-way BC vocabulary."""
    action = np.asarray(action, dtype=np.uint8)
    dpad = tuple(int(action[i]) for i in (4, 5, 6, 7))
    button = (int(action[8]), int(action[0]))  # A, B
    if dpad not in DPAD_TO_ID:
        raise ValueError(f"Unsupported D-pad combo for 36-way action space: {dpad}")
    if button not in BUTTON_TO_ID:
        raise ValueError(f"Unsupported button combo for 36-way action space: {button}")
    return DPAD_TO_ID[dpad] * len(BUTTON_TO_ID) + BUTTON_TO_ID[button]


def decode_action36(token: int) -> np.ndarray:
    """Decode one 36-way action token back to NES MultiBinary(9)."""
    token = int(token)
    if token < 0 or token >= ACTION_DIM:
        raise ValueError(f"Action token out of range: {token}")
    dpad_id, button_id = divmod(token, len(BUTTON_TO_ID))
    up, down, left, right = ID_TO_DPAD[dpad_id]
    a, b = ID_TO_BUTTON[button_id]
    action = np.zeros(9, dtype=np.uint8)
    action[0] = b
    action[4] = up
    action[5] = down
    action[6] = left
    action[7] = right
    action[8] = a
    return action


def _level_id(level: object) -> int:
    text = str(level)
    if text.lower().startswith("level"):
        try:
            return max(0, int(text[5:]) - 1)
        except ValueError:
            return 0
    return 0


def _encode_jpg(frame: np.ndarray, image_size: int, quality: int) -> bytes:
    image = Image.fromarray(frame).resize((image_size, image_size), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _trace_paths(trace_dir: Path, level: int, limit: int) -> list[str]:
    pattern = "win_level*_*.npz" if level == 0 else f"win_level{level}_*.npz"
    paths = sorted(glob.glob(str(trace_dir / pattern)))
    if limit >= 0:
        paths = paths[:limit]
    return paths


def _make_env():
    return retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )


def _load_trace(path: str) -> tuple[bytes, np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    initial_state = bytes(data["initial_state"])
    actions = np.asarray(data["actions"], dtype=np.uint8)
    level = _level_id(data["level"].item() if data["level"].shape == () else data["level"])
    if actions.ndim != 2 or actions.shape[1] != 9:
        raise ValueError(f"Expected actions with shape [N, 9], got {actions.shape}: {path}")
    return initial_state, actions, level


def replay_trace(path: str, image_size: int, jpg_quality: int) -> tuple[list[Sample], str]:
    """Replay one trace and return aligned one-step BC samples."""
    initial_state, actions, level_id = _load_trace(path)
    env = _make_env()
    env.reset()
    rewind_state(env, initial_state)

    samples: list[Sample] = []
    leveled_up = False
    game_cleared = False

    try:
        for timestep, raw_action in enumerate(actions):
            encoded = encode_action36(raw_action)
            decoded = decode_action36(encoded)
            if not np.array_equal(decoded, raw_action):
                raise ValueError(f"36-way decode changed action at timestep {timestep}: {path}")

            pre_ram = env.unwrapped.get_ram().copy()
            samples.append(
                Sample(
                    frame_jpg=_encode_jpg(env.em.get_screen().copy(), image_size, jpg_quality),
                    state=state_from_ram(pre_ram).astype(np.float32, copy=False),
                    action=encoded,
                    trace_path=path,
                    timestep=timestep,
                    level_id=level_id,
                    outcome="",
                    raw_action=raw_action.copy(),
                )
            )

            for _ in range(SKIP):
                env.step(raw_action.copy())

            curr_ram = env.unwrapped.get_ram().copy()
            if EV_LEVELUP.trigger(pre_ram, curr_ram):
                leveled_up = True
            if EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
                game_cleared = True
    finally:
        env.close()

    outcome = "game_clear" if game_cleared else "level_up" if leveled_up else "lose"
    for sample in samples:
        sample.outcome = outcome
    return samples, outcome


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int, seed: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.seed = seed
        self.samples: list[Sample] = []
        self.shard_idx = 0
        self.saved: list[Path] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_many(self, samples: list[Sample]) -> None:
        self.samples.extend(samples)
        while len(self.samples) >= self.shard_size:
            shard = self.samples[: self.shard_size]
            del self.samples[: self.shard_size]
            self._write(shard)

    def close(self) -> list[Path]:
        if self.samples:
            self._write(self.samples)
            self.samples = []
        return self.saved

    def _write(self, samples: list[Sample]) -> None:
        assert all(sample.outcome in {"level_up", "game_clear"} for sample in samples), (
            "Lose episode samples must never be written to BC shards"
        )
        rng = np.random.default_rng(self.seed + self.shard_idx)
        order = rng.permutation(len(samples))
        shuffled = [samples[i] for i in order]

        offsets = [0]
        for sample in shuffled:
            offsets.append(offsets[-1] + len(sample.frame_jpg))
        blob = np.frombuffer(b"".join(sample.frame_jpg for sample in shuffled), dtype=np.uint8)

        base = self.out_dir / f"shard_{self.shard_idx:04d}"
        np.save(str(base) + "_frames_blob.npy", blob)
        np.save(str(base) + "_frames_offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(str(base) + "_states.npy", np.stack([sample.state for sample in shuffled]))
        np.save(str(base) + "_actions.npy", np.asarray([sample.action for sample in shuffled], dtype=np.uint8))
        np.savez_compressed(
            str(base) + "_meta.npz",
            trace_path=np.asarray([sample.trace_path for sample in shuffled]),
            timestep=np.asarray([sample.timestep for sample in shuffled], dtype=np.int32),
            level_id=np.asarray([sample.level_id for sample in shuffled], dtype=np.int16),
            outcome=np.asarray([sample.outcome for sample in shuffled]),
            raw_action=np.stack([sample.raw_action for sample in shuffled]).astype(np.uint8),
        )

        self.saved.append(base)
        size_mib = blob.nbytes / (1024 * 1024)
        print(f"wrote {base.name}: {len(shuffled)} samples, {size_mib:.1f} MiB JPG")
        self.shard_idx += 1


def _move_shards_to_splits(out_dir: Path, shard_bases: list[Path]) -> None:
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if not shard_bases:
        raise RuntimeError("No shards were written")

    val_base = shard_bases[-1]
    for base in shard_bases:
        dst_dir = val_dir if base == val_base else train_dir
        dst_base = dst_dir / base.name
        for suffix in ("_frames_blob.npy", "_frames_offsets.npy", "_states.npy", "_actions.npy", "_meta.npz"):
            shutil.move(str(base) + suffix, str(dst_base) + suffix)


def preprocess(
    trace_dir: Path,
    level: int,
    out: str,
    limit: int,
    shard_size: int,
    image_size: int,
) -> None:
    paths = _trace_paths(trace_dir, level, limit)
    if not paths:
        raise FileNotFoundError(f"No traces matched level={level} in {trace_dir}")

    out_dir = Path(out)
    staging_dir = out_dir / "_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    writer = ShardWriter(staging_dir, shard_size, SHUFFLE_SEED)
    accepted = 0
    skipped = 0

    for idx, path in enumerate(paths, start=1):
        try:
            samples, outcome = replay_trace(path, image_size, JPG_QUALITY)
        except ValueError as exc:
            skipped += 1
            print(f"[{idx}/{len(paths)}] skip {os.path.basename(path)}: {exc}")
            continue

        assert outcome in {"level_up", "game_clear"}, (
            f"Lose episode must not produce train samples: {path} outcome={outcome}"
        )

        writer.add_many(samples)
        accepted += 1
        print(f"[{idx}/{len(paths)}] accept {os.path.basename(path)}: {len(samples)} samples, {outcome}")

    shard_bases = writer.close()
    _move_shards_to_splits(out_dir, shard_bases)
    shutil.rmtree(staging_dir)
    print(f"done: accepted={accepted}, skipped={skipped}, shards={len(shard_bases)}, out={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="vla/data/level1_400")
    parser.add_argument(
        "--level",
        type=int,
        choices=range(0, 9),
        default=0,
        help="Game level to process: 0 means all levels, 1-8 select win_levelN_*.npz",
    )
    parser.add_argument("--n", type=int, default=-1, help="Maximum number of trace files to consider")
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--image_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    preprocess(
        trace_dir=TRACE_DIR,
        level=args.level,
        out=args.out,
        limit=args.n,
        shard_size=args.shard_size,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
