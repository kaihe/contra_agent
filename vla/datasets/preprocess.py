"""
Preprocess raw trace npz files into VLA shard files.

Each shard set (deduplicated, JPEG-compressed):
  shard_NNNN_frames.npz  —  blob [M] uint8, offsets [M+1] int32  (JPEG bytes)
  shard_NNNN_indices.npy —  [S, 2] int32  (frame indices per sample)
  shard_NNNN_proprio.npy —  [S, 118] float32
  shard_NNNN_actions.npy —  [S, T] int8

Usage:
    python -m vla.datasets.preprocess \
        --traces "synthetic/mc_trace/win_level1_*" \
        --out     vla/data/level1_action2 \
        --n       100 \
        --val_frac 0.2 \
        --T       2 \
        --shard   5000 \
        --workers 8
"""

from __future__ import annotations

import argparse
import glob
import io
import os
from multiprocessing import Pool

import numpy as np
import stable_retro as retro
from PIL import Image

from annotate.prune import prune_actions
from contra.game_state import state_from_ram
from contra.replay import GAME, SKIP, rewind_state
from pixel2play.model.nes_actions import encode, encode_combined

# NES MultiBinary(9): [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
_NES_KEY_MAP = [(0, "f"), (4, "w"), (5, "s"), (6, "a"), (7, "d"), (8, "j")]


def _nes_keys(act: np.ndarray) -> list[str]:
    return [key for idx, key in _NES_KEY_MAP if act[idx]]


def _process_episode(
    npz_path: str, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay, prune, resize, and encode one trace. Returns (frames, proprio, actions).

    frames  : [N+1, image_size, image_size, 3] uint8
    proprio : [N+1, 118]                        float32
    actions : [N]                               int8  (combined index 0..35)

    Single env is reused for replay and pruning to satisfy the one-env-per-process rule.
    """
    ckpt          = np.load(npz_path, allow_pickle=True)
    initial_state = bytes(ckpt["initial_state"])
    raw_actions   = ckpt["actions"]           # [N, 9] uint8

    env = retro.make(
        game=GAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    env.reset()
    rewind_state(env, initial_state)

    frames_list = [env.em.get_screen().copy()]
    rams_list   = [env.unwrapped.get_ram().copy()]

    for act in raw_actions:
        act_arr = np.asarray(act, dtype=np.uint8)
        for _ in range(SKIP):
            env.step(act_arr.copy())
        frames_list.append(env.em.get_screen().copy())
        rams_list.append(env.unwrapped.get_ram().copy())

    # prune before encoding — reuse same env (prune_actions rewinds internally)
    pruned = prune_actions(raw_actions, initial_state, verbose=False, env=env)
    env.close()

    frames  = np.stack([
        np.asarray(Image.fromarray(f).resize((image_size, image_size), Image.BICUBIC))
        for f in frames_list
    ])                                                                  # [N+1, image_size, image_size, 3]
    proprio = np.stack([state_from_ram(r) for r in rams_list])         # [N+1, 118]
    actions = np.array(
        [encode_combined(*encode(_nes_keys(a))) for a in pruned],
        dtype=np.int8,
    )                                                                   # [N]

    return frames, proprio, actions


class _ShardWriter:
    """Accumulates samples and flushes to deduplicated, JPEG-compressed shards."""

    def __init__(self, out_dir: str, shard_size: int, T: int, jpeg_quality: int = 85) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir      = out_dir
        self.shard_size   = shard_size
        self.T            = T
        self.jpeg_quality = jpeg_quality
        self._unique_frames: list[np.ndarray] = []
        self._sample_indices: list[list[int]] = []
        self._proprio: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._frame_offset = 0
        self._idx = 0

    def add_episode(self, frames: np.ndarray, proprio: np.ndarray, actions: np.ndarray) -> None:
        """Slide a window over one episode and buffer the resulting samples.
        Frames are stored once per episode and referenced by index.
        Flushing happens between complete episodes to keep indices valid."""
        N = len(actions)
        episode_start = self._frame_offset
        for f in frames:
            self._unique_frames.append(f)
        self._frame_offset += len(frames)

        for t in range(1, N - self.T + 1):
            self._sample_indices.append([episode_start + t - 1, episode_start + t])
            self._proprio.append(proprio[t].copy())
            self._actions.append(actions[t : t + self.T].copy())

    def close(self) -> int:
        """Flush remaining samples. Returns total shards written."""
        self._flush()
        return self._idx

    def _flush(self) -> None:
        if not self._sample_indices:
            return
        path = os.path.join(self.out_dir, f"shard_{self._idx:04d}")

        # JPEG-encode unique frames
        jpeg_bytes = []
        for f in self._unique_frames:
            img = Image.fromarray(f)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            jpeg_bytes.append(buf.getvalue())

        offsets = [0]
        for b in jpeg_bytes:
            offsets.append(offsets[-1] + len(b))
        blob = np.frombuffer(b"".join(jpeg_bytes), dtype=np.uint8)

        np.save(path + "_frames_blob.npy",    blob)
        np.save(path + "_frames_offsets.npy", np.array(offsets, dtype=np.int32))
        np.save(path + "_indices.npy", np.array(self._sample_indices, dtype=np.int32))
        np.save(path + "_proprio.npy", np.stack(self._proprio))
        np.save(path + "_actions.npy", np.stack(self._actions))

        print(
            f"  shard {self._idx:04d}  {len(self._sample_indices)} samples  "
            f"{len(self._unique_frames)} unique frames  →  {path}"
        )
        self._idx += 1
        self._unique_frames.clear()
        self._sample_indices.clear()
        self._proprio.clear()
        self._actions.clear()
        self._frame_offset = 0


def preprocess(
    traces: str,
    out_dir: str,
    n_episodes: int = 100,
    val_frac: float = 0.2,
    T: int = 2,
    shard_size: int = 5000,
    workers: int = 4,
    image_size: int = 256,
    jpeg_quality: int = 85,
) -> None:
    # collect traces: glob pattern or directory
    if os.path.isdir(traces):
        all_paths = sorted(glob.glob(os.path.join(traces, "*.npz")))
    else:
        all_paths = sorted(glob.glob(traces) or glob.glob(traces + ".npz"))
    if n_episodes < 0:
        paths = all_paths
    else:
        if len(all_paths) < n_episodes:
            raise ValueError(f"Found only {len(all_paths)} traces matching '{traces}', need {n_episodes}")
        paths = all_paths[:n_episodes]

    n_val  = max(1, round(len(paths) * val_frac))
    splits = {"train": paths[:-n_val], "val": paths[-n_val:]}

    print(f"Episodes: {len(paths)}  (train {len(splits['train'])}, val {len(splits['val'])})")
    print(f"Chunk T={T}  image_size={image_size}  shard_size={shard_size}  workers={workers}  jpeg_q={jpeg_quality}  out={out_dir}\n")

    from functools import partial
    worker_fn = partial(_process_episode, image_size=image_size)

    for split, split_paths in splits.items():
        writer = _ShardWriter(os.path.join(out_dir, split), shard_size, T, jpeg_quality)
        with Pool(processes=workers) as pool:
            for i, (frames, proprio, actions) in enumerate(
                pool.imap(worker_fn, split_paths, chunksize=1)
            ):
                print(f"[{split}] {i+1:3d}/{len(split_paths)}  {os.path.basename(split_paths[i])}")
                writer.add_episode(frames, proprio, actions)
                if len(writer._sample_indices) >= shard_size:
                    writer._flush()
        n_shards = writer.close()
        print(f"[{split}] done — {n_shards} shards\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces",   default="synthetic/mc_trace")
    p.add_argument("--out",      default="vla/data/level1_action2")
    p.add_argument("--n",        type=int,   default=-1)
    p.add_argument("--val_frac", type=float, default=0.02)
    p.add_argument("--T",        type=int,   default=2)
    p.add_argument("--shard",    type=int,   default=8192)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--image_size", type=int,   default=192)
    p.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality for frame compression (1-95)")
    args = p.parse_args()

    preprocess(
        traces     = args.traces,
        out_dir    = args.out,
        n_episodes = args.n,
        val_frac   = args.val_frac,
        T          = args.T,
        shard_size   = args.shard,
        workers      = args.workers,
        image_size   = args.image_size,
        jpeg_quality = args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
