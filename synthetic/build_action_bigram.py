
import glob
import os
import numpy as np

from contra.inputs import NUM_DPAD, NUM_BUTTONS, DPAD_NAMES, BUTTON_NAMES

NUM_ACTIONS = NUM_DPAD * NUM_BUTTONS  # 7 * 4 = 28


def nes_to_action_idx(nes: np.ndarray) -> int:
    """Convert a 9-button NES action vector to a flat action index."""
    up, down, left, right = bool(nes[4]), bool(nes[5]), bool(nes[6]), bool(nes[7])
    fire, jump = bool(nes[0]), bool(nes[8])

    if right and up:    dpad = 5
    elif right and down: dpad = 6
    elif right:         dpad = 1
    elif left:          dpad = 2
    elif up:            dpad = 3
    elif down:          dpad = 4
    else:               dpad = 0

    if fire and jump:  btn = 3
    elif fire:         btn = 1
    elif jump:         btn = 2
    else:              btn = 0

    return dpad * NUM_BUTTONS + btn


def build_bigram(recording_dirs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Count bigram transitions from all recordings. Returns (counts, probs)."""
    counts = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    total_files = total_pairs = 0

    for rec_dir in recording_dirs:
        npz_files = glob.glob(os.path.join(rec_dir, "*.npz"))
        for fpath in npz_files:
            try:
                data = np.load(fpath, allow_pickle=True)
            except Exception as e:
                print(f"  WARN: cannot load {fpath}: {e}")
                continue

            actions = data["actions"]
            if actions.ndim != 2 or actions.shape[1] != 9:
                print(f"  WARN: unexpected action shape {actions.shape} in {fpath}, skipping")
                continue

            idxs = [nes_to_action_idx(a) for a in actions]
            for prev, curr in zip(idxs[:-1], idxs[1:]):
                counts[prev, curr] += 1

            total_pairs += len(idxs) - 1
            total_files += 1
            print(f"  {os.path.basename(fpath):40s}  frames={len(actions):5d}")

    print(f"\nTotal files: {total_files}, total bigram pairs: {total_pairs:,}")

    # Normalise rows → probability distribution
    # Rows with zero observations fall back to uniform
    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    probs = np.where(row_sums > 0, counts / row_sums, 1.0 / NUM_ACTIONS)
    return counts, probs.astype(np.float32)


def print_table(probs: np.ndarray) -> None:
    """Print the bigram table in a human-readable form."""
    action_names = [f"{d}{b}" for d in DPAD_NAMES for b in BUTTON_NAMES]

    col_w = 7
    header = f"{'PREV→':12s}" + "".join(f"{n:>{col_w}}" for n in action_names)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(probs):
        row_str = f"{action_names[i]:12s}" + "".join(f"{v:>{col_w}.2f}" for v in row)
        print(row_str)


if __name__ == "__main__":
    BC_ROOT = os.path.join(os.path.dirname(__file__), "..", "contra", "human_recordings")
    OUT_DIR = os.path.dirname(__file__)

    all_level_dirs = sorted(glob.glob(os.path.join(BC_ROOT, "Level*")))
    if not all_level_dirs:
        print(f"No Level* directories found under {BC_ROOT}")
        raise SystemExit(1)

    all_probs = {}
    for level_dir in all_level_dirs:
        level_name = os.path.basename(level_dir)  # e.g. "Level1"
        print(f"\n{'='*60}")
        print(f"Building bigram for {level_name}")
        print(f"{'='*60}")

        counts, probs = build_bigram([level_dir])
        all_probs[level_name] = probs

        print(f"\n── Bigram probability table for {level_name} (rows=prev, cols=next) ──")
        print_table(probs)

    out_path = os.path.join(OUT_DIR, "action_bigram.npz")
    np.savez(out_path, **all_probs)
    print(f"\nSaved all levels → {out_path}  (keys: {list(all_probs.keys())})")
