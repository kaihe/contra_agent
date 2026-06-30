"""Build a per-level action bigram prior for mc_search.

For each level, count action→action transitions in the human recordings and
store an (N, N) row-stochastic matrix indexed in that level's *own* flat action
ordering — i.e. the table from ``synthetic/search_action_space.load_for_level``
(``synthetic/action_configs/level<N>.yaml``, else baseline). This matches what
``mc_search.build_context`` consumes directly, with no 36-combo remapping.

A recorded button vector is mapped to the level's table by its (d-pad, button)
combo; transitions touching an action absent from the table (e.g. a rare combo
trimmed out of the level set) are skipped, so the prior only spans the actions
the search can actually take.

Usage:
    python synthetic/build_action_bigram.py            # all levels
    python synthetic/build_action_bigram.py --levels 1,2
"""

import argparse
import glob
import os

import numpy as np

from synthetic.action_configs.search_action_space import load_for_level

BC_ROOT = os.path.join(os.path.dirname(__file__), "..", "contra", "human_recordings")
OUT_PATH = os.path.join(os.path.dirname(__file__), "action_bigram.npz")


def _combo_index(nes: np.ndarray) -> int:
    """Universal (d-pad, button) combo index for a 9-button NES vector.

    9 d-pad states x 4 button states = 36 combos. Used only to match a recorded
    action against a level's table (which actions share this combo space).
    """
    up, down, left, right = bool(nes[4]), bool(nes[5]), bool(nes[6]), bool(nes[7])
    fire, jump = bool(nes[0]), bool(nes[8])
    if   up and left:    dpad = 5
    elif up and right:   dpad = 6
    elif down and left:  dpad = 7
    elif down and right: dpad = 8
    elif left:           dpad = 1
    elif right:          dpad = 2
    elif up:             dpad = 3
    elif down:           dpad = 4
    else:                dpad = 0
    btn = (3 if fire and jump else 1 if jump else 2 if fire else 0)
    return dpad * 4 + btn


def build_prior(level: int, files: list[str], mode: str = "bigram",
                smooth: float = 0.0, verbose: bool = False):
    """Return (names, (N, 9) action table, (N, N) prior pmf) for `level` from `files`.

    `mode` selects the action-distribution order:

      * ``"bigram"``  — pmf[i, :] = P(next | prev = i), row-normalised transition
        counts. Sampling depends on the previous action.
      * ``"unigram"`` — pmf[i, :] = P(action), the marginal action frequency,
        identical for every row i. Sampling ignores the previous action (and so
        slots into the same ``pmf[prev]`` machinery with no rollout change).

    `files` is the explicit list of trace NPZs to count from — human recordings or
    generated win traces (already in the level's vocabulary, so nothing skipped).
    `smooth` ∈ [0, 1] blends each row toward uniform (``(1-smooth)*p + smooth/n``)
    so rare-but-legal actions keep nonzero weight and stay explorable.
    """
    if mode not in ("bigram", "unigram"):
        raise ValueError(f"mode must be 'bigram' or 'unigram', got {mode!r}")
    actions = load_for_level(level).action_space
    names = list(actions.names)
    table = actions.actions_np()
    n = len(table)
    combo_to_idx = {_combo_index(a): i for i, a in enumerate(table)}

    trans = np.zeros((n, n), dtype=np.int64)   # transition counts (bigram)
    visits = np.zeros(n, dtype=np.int64)       # action occurrences (unigram)
    nfiles = pairs = skipped = 0
    for fpath in files:
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"  WARN: cannot load {fpath}: {e}")
            continue
        rec = data["actions"]
        if rec.ndim != 2 or rec.shape[1] != 9:
            print(f"  WARN: unexpected action shape {rec.shape} in {fpath}, skipping")
            continue

        idxs = [combo_to_idx.get(_combo_index(a)) for a in rec]
        for k in idxs:
            if k is not None:
                visits[k] += 1
        for prev, curr in zip(idxs[:-1], idxs[1:]):
            if prev is None or curr is None:      # touches an out-of-table action
                skipped += 1
                continue
            trans[prev, curr] += 1
        pairs += max(0, len(idxs) - 1)
        nfiles += 1
        if verbose:
            print(f"  {os.path.basename(fpath):40s}  frames={len(rec):5d}")

    if verbose:
        print(f"  files={nfiles}  pairs={pairs:,}  skipped(out-of-table)={skipped:,}  "
              f"actions={n}  mode={mode}  smooth={smooth}")

    if mode == "bigram":
        row_sums = trans.sum(axis=1, keepdims=True).astype(np.float64)
        # Rows with no observed transitions fall back to uniform; divide only where
        # the row has mass to avoid a 0/0 warning.
        probs = np.divide(trans, row_sums, out=np.full(trans.shape, 1.0 / n),
                          where=row_sums > 0)
    else:  # unigram: every row is the marginal action distribution
        total = visits.sum()
        marginal = visits / total if total > 0 else np.full(n, 1.0 / n)
        probs = np.tile(marginal, (n, 1))
    if smooth > 0:
        probs = (1.0 - smooth) * probs + smooth / n
    return names, table.astype(np.uint8), probs.astype(np.float32)


def print_table(probs: np.ndarray, names: list[str]) -> None:
    """Print the bigram table (rows=prev, cols=next) in a readable form."""
    col_w = 6
    header = f"{'PREV→':6s}" + "".join(f"{m:>{col_w}}" for m in names)
    print(header)
    print("-" * len(header))
    for name, row in zip(names, probs):
        print(f"{name:6s}" + "".join(f"{v:>{col_w}.2f}" for v in row))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="1,2,3,4,5,6,7,8",
                   help="Comma-separated levels to (re)build from human recordings (default: all)")
    p.add_argument("--from", dest="from_glob", default=None,
                   help="Build ONE level's bigram from this glob of trace NPZs (e.g. "
                        "'tmp/mc_trace/level1_human_trace/*.npz') instead of human "
                        "recordings; requires --level. Use to bootstrap the prior from "
                        "the search's own win traces.")
    p.add_argument("--level", type=int, default=None, help="Level for --from mode")
    p.add_argument("--prior", choices=["bigram", "unigram"], default="bigram",
                   help="Action-distribution order: 'bigram' = P(next|prev) (default); "
                        "'unigram' = marginal P(action), ignores the previous action")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Blend each row toward uniform by this fraction [0,1] to keep "
                        "rare actions explorable (recommend ~0.1 for win-trace priors)")
    p.add_argument("--out", default=OUT_PATH, help=f"Output NPZ (default: {OUT_PATH})")
    p.add_argument("--quiet", action="store_true", help="Skip the per-level tables")
    args = p.parse_args()

    # Preserve any existing keys so rebuilding a subset doesn't drop other levels.
    all_probs = {}
    if os.path.exists(args.out):
        with np.load(args.out) as existing:
            all_probs = {k: existing[k] for k in existing.files}

    if args.from_glob:
        if args.level is None:
            p.error("--from requires --level")
        files = sorted(glob.glob(args.from_glob))
        if not files:
            p.error(f"no traces match {args.from_glob!r}")
        print(f"\n{'='*60}\nBuilding Level{args.level} {args.prior} prior from "
              f"{len(files)} trace(s)\n{'='*60}")
        names, table, probs = build_prior(args.level, files, mode=args.prior,
                                          smooth=args.smooth, verbose=True)
        all_probs[f"Level{args.level}"] = probs
        all_probs[f"Level{args.level}__actions"] = table
        if not args.quiet:
            print(f"\n── Level{args.level} {args.prior} (rows=prev, cols=next) ──")
            print_table(probs, names)
    else:
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        for level in levels:
            rec_dir = os.path.join(BC_ROOT, f"Level{level}")
            print(f"\n{'='*60}\nBuilding {args.prior} for Level{level}\n{'='*60}")
            if not os.path.isdir(rec_dir):
                print(f"  no recordings at {rec_dir}, skipping")
                continue
            files = sorted(glob.glob(os.path.join(rec_dir, "*.npz")))
            names, table, probs = build_prior(level, files, mode=args.prior,
                                              smooth=args.smooth, verbose=True)
            all_probs[f"Level{level}"] = probs
            all_probs[f"Level{level}__actions"] = table
            if not args.quiet:
                print(f"\n── Level{level} {args.prior} (rows=prev, cols=next) ──")
                print_table(probs, names)

    np.savez(args.out, **all_probs)
    levels_saved = sorted(k for k in all_probs if not k.endswith("__actions"))
    print(f"\nSaved → {args.out}  (levels: {levels_saved})")


if __name__ == "__main__":
    main()
