# Behavior Cloning Dataset Design

This directory owns the offline behavior cloning dataset for Contra VLA. The
dataset is built from existing successful recordings in `synthetic/mc_trace` by
replaying each recording, extracting the pre-action game image and RAM-derived
state for each action, and writing independent supervised samples into
dataloader-friendly shards.

## Training Contract

Each training instance is one independent supervised example:

```text
(image, goal_text, state) -> action
```

Inputs:

- `image`: one RGB game image resized to `192x192`.
- `goal_text`: natural-language description of the level or goal.
- `state`: structured game-state vector from `contra.game_state.state_from_ram()`.

Target:

- `action`: one integer token in the 36-way action vocabulary, `0..35`.

The action vocabulary is defined as `9 D-pad states x 4 button states = 36`.
The implementation must keep a single canonical encoder and decoder for this
mapping. Dataset preprocessing, model configs, benchmark code, and environment
wrappers must all use the same `action_dim=36`.

## Source Data

Source traces live under `synthetic/mc_trace`. Only traces whose replay outcome
is `level_up` or `game_clear` should be accepted.

For each source trace, generate samples in the recorded action order:

1. Load the trace's initial emulator state and action sequence.
2. Recreate the episode by applying the recorded actions with the same stepping
   semantics as `contra.replay` (`SKIP` emulator frames per logical action).
3. Immediately before applying action `a_i`, capture the current screen and RAM.
4. Resize the screen to `192x192`.
5. Convert RAM to a state vector with `state_from_ram()`.
6. Encode `a_i` as one 36-way action token.
7. Append one training sample: `(image_i, state_i, action_i)`.

The replay process is only a dataset-generation mechanism. The final train
dataset does not store episode transitions, next observations, or trajectory
windows.

## Causal Alignment

The critical invariant is:

```text
image_i, state_i are captured before action_i is applied
```

Using the image or state after `action_i` as the input for `action_i` leaks
future information and creates an off-by-one dataset.

For every generated sample:

```text
sample.image  = pre_action_screen_i
sample.state  = state_from_ram(pre_action_ram_i)
sample.action = encode_36(action_i)
```

After the sample is generated, it is independent of other samples. It is safe to
shuffle samples across timesteps and across episodes because this BC objective
does not use recurrent hidden state, neighboring frames, previous actions, or
future actions.

## Replay Validation

Every accepted source trace should pass these checks before its samples are
written:

- Replaying the source action sequence from its initial state produces
  `level_up` or `game_clear`.
- The number of generated samples equals the number of recorded actions.
- Every generated sample has exactly one frame, one state vector, and one action
  token.
- Every action token is in `[0, 35]`.
- Decoding each 36-way token back to emulator input preserves the intended
  controller action.
- If any action rewriting or pruning is introduced, the rewritten action
  sequence must be replayed and samples must be regenerated from that exact
  rewritten sequence. Never pair frames/states from one action sequence with
  labels from another.

Per-sample debug metadata should be kept out of the hot training path but is
useful for audits: source trace path, episode id, original timestep, level id,
goal id, outcome, and optionally the raw controller action.

## Shard Layout

The dataset should be written as split directories:

```text
vla/data/<dataset_name>/
  train/
    shard_0000_frames_blob.npy
    shard_0000_frames_offsets.npy
    shard_0000_states.npy
    shard_0000_actions.npy
    shard_0000_meta.npz
  val/
    shard_0000_frames_blob.npy
    shard_0000_frames_offsets.npy
    shard_0000_states.npy
    shard_0000_actions.npy
    shard_0000_meta.npz
```

Required arrays per shard:

- `*_frames_blob.npy`: flat `uint8 [B]` byte blob containing concatenated JPG
  images. Each JPG is one pre-resized `192x192` RGB frame.
- `*_frames_offsets.npy`: `int64 [S + 1]` byte offsets into
  `*_frames_blob.npy`. Frame `j` is stored in
  `blob[offsets[j]:offsets[j + 1]]`.
- `*_states.npy`: `float32 [S, state_dim]` output of `state_from_ram()`.
- `*_actions.npy`: `uint8 [S]` 36-way action token.

Optional sidecar:

- `*_meta.npz`: compact debug and reproducibility metadata. It should not be
  loaded during normal training unless needed.

The shard sample count `S` is defined by:

```text
S = len(states) = len(actions) = len(frames_offsets) - 1
```

Sample `j` in a shard is exactly:

```text
decode_jpg(frames_blob[frames_offsets[j]:frames_offsets[j + 1]]), states[j] -> actions[j]
```

No frame indices, next-frame arrays, or trajectory windows are needed in the
training shard. Once samples are materialized, shard order is not semantically
meaningful.

Before saving each shard, randomly shuffle the samples assigned to that shard
with a fixed seed. The same permutation must be applied to the JPG frame list,
state rows, action rows, and metadata rows before writing
`frames_blob/frames_offsets`, `states`, `actions`, and `meta`.

For compatibility with current loader naming, `states` may be called `proprio`,
but the design meaning is the same: the vector is generated by
`state_from_ram()`.

## Throughput-Oriented Organization

The dataloader should do the least possible work per sample:

- Resize frames to `192x192` during preprocessing, not in `__getitem__`.
- Store action tokens directly as integers; do not encode key sets in the
  dataloader.
- Store `state_from_ram()` output directly as `float32`.
- Keep arrays contiguous and load them with `mmap_mode="r"` or fully resident
  RAM depending on measured size and throughput.
- Use `32,768` samples per full shard. The final shard in a split may be
  smaller.
- Avoid one file per sample. The unit of filesystem access should be a shard.
- Pre-tokenize fixed goal text if text tokenization becomes measurable overhead.
  Otherwise keep one short goal id per sample and batch-tokenize in the collate
  function.
- Use `persistent_workers=True`, `pin_memory=True`, and a nontrivial
  `prefetch_factor` in the PyTorch `DataLoader`.

JPG frame blobs are the default storage format. They reduce disk use and page
cache pressure substantially compared with raw `uint8` frames, at the cost of
CPU decode in dataloader workers. Use high-quality JPG, starting with
`quality=90` or `quality=95`, because Contra has small sprites, bullets, and UI
details that should not be blurred by aggressive compression.

Estimated shard sizes:

```text
raw frame bytes per sample = 192 * 192 * 3 = 110,592 bytes ~= 108 KiB
estimated JPG bytes per sample at quality 90-95 ~= 10-30 KiB
state bytes per sample = 118 * 4 = 472 bytes, if state_dim is 118
action bytes per sample = 1 byte
total JPG sample size ~= 10.5-30.5 KiB
```

| Samples per shard | JPG frames, 10 KiB each | States | Actions | Total |
|-------------------|-------------------------|--------|---------|-------|
| `32,768` | `320 MiB` | `14.8 MiB` | `32 KiB` | `~335 MiB` |

The target shard size is `32,768` samples. With the expected `10 KiB` JPG frame
size, each full shard uses about `320 MiB` for frames and `335 MiB` total on
disk, which is acceptable for this dataset.

Expected dataloader speed depends on whether frames are already in page cache:

- Warm page cache: JPG decode and image normalization dominate. Four workers
  should be enough to start; increase workers if GPU utilization is low.
- Cold NVMe reads: one batch of `64` JPG frames reads roughly `0.6-1.9 MiB`
  before decode, versus `6.75 MiB` raw. I/O should rarely be the bottleneck on
  local NVMe.
- HDD or network storage: JPG is strongly preferred over raw frames. Keep
  samples shuffled when writing shards, then read mostly sequentially within
  each shard and shuffle shard/block order if fully random reads are slow.
- CPU-bound decode: if dataloader workers saturate CPU while GPU waits, either
  increase worker count, lower JPG quality only after visual checks, or switch a
  measured subset back to raw frames for comparison.

For `batch_size=64`, start with this loader shape per trainer process:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```

Tuning guidance:

- Use `4` train workers as the default for one GPU.
- Increase to `8` train workers if GPU utilization is low and CPU cores are
  available.
- Drop to `2` workers if the dataset fits in RAM/page cache and worker overhead
  dominates.
- Use `2` validation workers because validation is smaller and does not need
  aggressive prefetching.
- Keep batch size at `64` at the dataloader boundary. If the model does not fit,
  use gradient accumulation in the trainer rather than changing the dataset
  format. For example, micro-batch `16` with `accumulate_grad_batches=4` gives
  an effective batch size of `64`.

## Train/Validation Split

Use the last saved shard as the validation dataset. All earlier shards are the
training dataset.

Recommended procedure:

1. Generate the full list of accepted samples.
2. Partition the samples into `32,768`-sample shards in generation order.
3. Before saving each shard, randomly shuffle only the samples assigned to that
   shard with a fixed seed.
4. Save every shard except the last one under `train/`.
5. Save the last shard under `val/`.

This keeps split construction simple and reproducible. Because each shard is
shuffled internally before it is saved, the validation shard is still a shuffled
set of independent one-step BC samples, even though the split boundary is the
last saved shard.
