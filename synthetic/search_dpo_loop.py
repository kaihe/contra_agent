"""
search_dpo_loop.py — Run mc_search_dpo in a loop forever, collecting winning DPO graphs.

Usage:
    python synthetic/search_dpo_loop.py                   # random levels
    python synthetic/search_dpo_loop.py --level 1         # fixed level
    Ctrl-C to stop and print a summary.
"""

import argparse
import os
import random
import signal
import sys
import time

import numpy as np

from synthetic.mc_search import _load_bigram
from synthetic.mc_search_dpo import _run_one_search, DPO_DIR


def _print_summary(run: int, total_graphs: int, total_time: float) -> None:
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Runs completed   : {run}")
    print(f"  Graphs produced  : {total_graphs}")
    if total_graphs > 0:
        print(f"  Avg time / graph : {total_time / total_graphs:.1f}s")
    else:
        print(f"  Avg time / graph : N/A (no wins yet)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run mc_search_dpo in a loop forever, collecting DPO graphs."
    )
    parser.add_argument("--level",          type=int, default=None, choices=list(range(1, 9)),
                        help="Fixed level to search (default: random each run)")
    parser.add_argument("--workers",        type=int, default=os.cpu_count())
    parser.add_argument("--rollouts",       type=int, default=128)
    parser.add_argument("--rollout-len",    type=int, default=48)
    parser.add_argument("--max-rewind",     type=int, default=32)
    parser.add_argument("--max-time",       type=int, default=600)
    parser.add_argument("--max-actions",    type=int, default=4000)
    parser.add_argument("--secondary-prob", type=float, default=0.25)
    parser.add_argument("--goal",           type=str, default="level_up",
                        choices=["level_up", "game_clear"])
    args = parser.parse_args()

    total_graphs = 0
    total_time   = 0.0
    run          = 0

    def _on_exit(sig, frame):
        _print_summary(run, total_graphs, total_time)
        sys.exit(0)
    signal.signal(signal.SIGINT,  _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    np.random.seed(int(time.time() * 1000) % (2**32))

    print("=" * 60)
    print(f"  MC Search DPO Loop  ({'random levels 1-8' if args.level is None else f'level {args.level}'})")
    print(f"  Workers        : {args.workers}")
    print(f"  Goal           : {args.goal}")
    print(f"  Secondary Prob : {args.secondary_prob}")
    print(f"  Time budget    : {args.max_time}s / run")
    print(f"  Graphs → {DPO_DIR}")
    print(f"  Ctrl-C to stop")
    print("=" * 60)

    while True:
        run  += 1
        level = args.level if args.level is not None else random.randint(1, 8)
        _load_bigram(level)

        print(f"\n{'─' * 60}")
        print(f"  Run #{run}   level: {level}")
        print(f"{'─' * 60}")

        t0     = time.time()
        trace_path, graph_path = _run_one_search(
            level=level,
            rollouts=args.rollouts,
            rollout_len=args.rollout_len,
            max_time=args.max_time,
            max_rewind=args.max_rewind,
            max_actions=args.max_actions,
            goal=args.goal,
            workers=args.workers,
            secondary_prob=args.secondary_prob,
            verbose=True,
            instance_id=None,
        )
        elapsed = time.time() - t0

        if graph_path:
            total_graphs += 1
        total_time += elapsed

        avg = total_time / total_graphs if total_graphs else float('nan')
        print(f"\n  Run #{run} done  "
              f"win={'yes' if graph_path else 'no'}  "
              f"time={elapsed:.1f}s  "
              f"total_graphs={total_graphs}  "
              f"avg_time/graph={avg:.1f}s")


if __name__ == "__main__":
    main()
