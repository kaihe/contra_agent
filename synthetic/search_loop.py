"""
search_loop.py — Run parallel MC searches in a loop forever, tracking throughput.

Each round launches N independent search instances in parallel (splitting workers
evenly). After every round the script prints cumulative stats including the average
wall-clock time it took to produce one winning trace.

Usage:
    python search_loop.py                              # random levels, 8 instances
    python search_loop.py --level 1                    # fixed level
    python search_loop.py --instances 8 --workers 128  # 8 × 16-worker searches
    Ctrl-C to stop and print a final summary.
"""

import argparse
import multiprocessing as mp
import os
import random
import signal
import sys
import time

import numpy as np

from mc_search import (
    _run_one_search,
    _load_bigram,
    DEFAULT_STATE_BY_LEVEL,
    TRACE_DIR,
)


# thin shim so pool.map (single-argument) can call _run_one_search
def _run_instance(kwargs: dict) -> str | None:
    return _run_one_search(**kwargs)


def _print_summary(run: int, total_traces: int, total_time: float) -> None:
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Rounds completed : {run}")
    print(f"  Traces produced  : {total_traces}")
    if total_traces > 0:
        print(f"  Avg time / trace : {total_time / total_traces:.1f}s")
    else:
        print(f"  Avg time / trace : N/A (no wins yet)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel MC searches in a loop, collecting winning traces."
    )
    parser.add_argument("--level",       type=int, default=None, choices=list(range(1, 9)),
                        help="Fixed level to search (default: random each round)")
    parser.add_argument("--instances",   type=int, default=8,
                        help="Parallel search instances per round (default: 8)")
    parser.add_argument("--workers",     type=int, default=os.cpu_count(),
                        help="Total rollout workers, split evenly across instances")
    parser.add_argument("--rollouts",    type=int, default=512)
    parser.add_argument("--rollout-len", type=int, default=48)
    parser.add_argument("--max-rewind",  type=int, default=30)
    parser.add_argument("--max-time",    type=int, default=600,
                        help="Time budget per instance in seconds (default: 600)")
    parser.add_argument("--max-actions", type=int, default=4000)
    parser.add_argument("--goal",        type=str, default="level_up",
                        choices=["level_up", "game_clear"])
    args = parser.parse_args()

    workers_per = max(1, args.workers // args.instances)
    ctx = mp.get_context('spawn')

    total_traces = 0
    total_time   = 0.0
    run          = 0

    # allow Ctrl-C to print summary and exit cleanly
    def _on_exit(sig, frame):
        _print_summary(run, total_traces, total_time)
        sys.exit(0)
    signal.signal(signal.SIGINT,  _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    np.random.seed(int(time.time() * 1000) % (2**32))

    print("=" * 60)
    if args.level:
        print(f"  MC Search Loop  (level {args.level}, fixed)")
    else:
        print(f"  MC Search Loop  (random levels 1-8)")
    print(f"  Instances/round : {args.instances}  ×  {workers_per} workers each")
    print(f"  Goal            : {args.goal}")
    print(f"  Time budget     : {args.max_time}s / instance")
    print(f"  Traces → {TRACE_DIR}")
    print(f"  Ctrl-C to stop")
    print("=" * 60)

    while True:
        run  += 1
        level = args.level if args.level is not None else random.randint(1, 8)
        _load_bigram(level)

        print(f"\n{'─' * 60}")
        print(f"  Round #{run}   level: {level}")
        print(f"{'─' * 60}")

        instance_kwargs = [dict(
            level=level,
            rollouts=args.rollouts,
            rollout_len=args.rollout_len,
            max_time=args.max_time,
            max_rewind=args.max_rewind,
            max_actions=args.max_actions,
            goal=args.goal,
            workers=workers_per,
            instance_id=i,
        ) for i in range(args.instances)]

        t0 = time.time()
        with ctx.Pool(args.instances) as ipool:
            results = ipool.map(_run_instance, instance_kwargs)
        elapsed = time.time() - t0

        wins = [r for r in results if r]
        total_traces += len(wins)
        total_time   += elapsed

        avg = total_time / total_traces if total_traces else float('nan')
        print(f"\n  Round #{run} done  "
              f"wins={len(wins)}/{args.instances}  "
              f"time={elapsed:.1f}s  "
              f"total_traces={total_traces}  "
              f"avg_time/trace={avg:.1f}s")


if __name__ == "__main__":
    main()
