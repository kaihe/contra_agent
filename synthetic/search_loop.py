"""
search_loop.py — Run mc_search in a loop forever, tracking avg time per trace.

Usage:
    python search_loop.py                   # random levels
    python search_loop.py --level 1         # fixed level
    Ctrl-C to stop and print a summary.
"""

import argparse
import os
import random
import signal
import sys
import time

import numpy as np

from mc_search import _run_one_search, _load_bigram, TRACE_DIR


def _print_summary(run: int, total_traces: int, total_time: float) -> None:
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Runs completed   : {run}")
    print(f"  Traces produced  : {total_traces}")
    if total_traces > 0:
        print(f"  Avg time / trace : {total_time / total_traces:.1f}s")
    else:
        print(f"  Avg time / trace : N/A (no wins yet)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run mc_search in a loop forever, collecting winning traces."
    )
    parser.add_argument("--level",       type=int, default=None, choices=list(range(1, 9)),
                        help="Fixed level to search (default: random each run)")
    parser.add_argument("--workers",     type=int, default=os.cpu_count())
    parser.add_argument("--rollouts",    type=int, default=512)
    parser.add_argument("--rollout-len", type=int, default=48)
    parser.add_argument("--max-rewind",  type=int, default=30)
    parser.add_argument("--max-time",    type=int, default=600)
    parser.add_argument("--max-actions", type=int, default=4000)
    parser.add_argument("--goal",        type=str, default="level_up",
                        choices=["level_up", "game_clear"])
    args = parser.parse_args()

    total_traces = 0
    total_time   = 0.0
    run          = 0

    def _on_exit(sig, frame):
        _print_summary(run, total_traces, total_time)
        sys.exit(0)
    signal.signal(signal.SIGINT,  _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    np.random.seed(int(time.time() * 1000) % (2**32))

    print("=" * 60)
    print(f"  MC Search Loop  ({'random levels 1-8' if args.level is None else f'level {args.level}'})")
    print(f"  Workers  : {args.workers}")
    print(f"  Goal     : {args.goal}")
    print(f"  Time budget : {args.max_time}s / run")
    print(f"  Traces → {TRACE_DIR}")
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
        result = _run_one_search(
            level=level,
            rollouts=args.rollouts,
            rollout_len=args.rollout_len,
            max_time=args.max_time,
            max_rewind=args.max_rewind,
            max_actions=args.max_actions,
            goal=args.goal,
            workers=args.workers,
            verbose=True,
        )
        elapsed = time.time() - t0

        if result:
            total_traces += 1
        total_time += elapsed

        avg = total_time / total_traces if total_traces else float('nan')
        print(f"\n  Run #{run} done  "
              f"win={'yes' if result else 'no'}  "
              f"time={elapsed:.1f}s  "
              f"total_traces={total_traces}  "
              f"avg_time/trace={avg:.1f}s")


if __name__ == "__main__":
    main()
