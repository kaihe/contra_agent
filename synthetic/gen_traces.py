"""Generate N winning mc_traces for a level.

Thin CLI over synthetic.mc_search.generate_traces: runs the bigram-guided
Monte Carlo search repeatedly until N winning traces are collected. Each trace
is kept clean during generation by the fire/jump penalty and saved under
tmp/mc_trace/level<N>/ with a unique, instance-suffixed filename.

(Equivalent to `python synthetic/mc_search.py --level X --runs N`; this script
just exposes the common knobs with a focused interface.)

Usage:
    python synthetic/gen_traces.py --level 2 --n 10
    python synthetic/gen_traces.py --level 5 --n 3 --max-time 300
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Support both `python synthetic/gen_traces.py` (synthetic/ on path) and
# `python -m synthetic.gen_traces` (repo root on path).
try:
    from synthetic.mc_search import generate_traces
except ImportError:
    from mc_search import generate_traces


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True, choices=list(range(1, 9)),
                   help="Level to generate traces for")
    p.add_argument("--n", type=int, default=1,
                   help="Number of winning traces to generate (default: 1)")
    p.add_argument("--max-time", type=int, default=600,
                   help="Per-search time budget in seconds (default: 600)")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Cap on total searches (default: 3*n)")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="Parallel rollout workers per search")
    p.add_argument("--goal", type=str, default="level_up",
                   choices=["level_up", "game_clear"],
                   help="level_up: stop on level-up (default); game_clear: full clear")
    args = p.parse_args()

    print(f"Generating {args.n} winning trace(s) for level {args.level} "
          f"(workers={args.workers}, max_time={args.max_time}s, goal={args.goal})")
    paths = generate_traces(
        args.level, args.n,
        max_time=args.max_time, workers=args.workers, goal=args.goal,
        max_attempts=args.max_attempts,
    )
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
