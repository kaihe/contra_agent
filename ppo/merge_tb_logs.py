"""
Merge two TensorBoard log directories by appending run2 to run1.

Useful when a training run was resumed but the step counter was reset to 0,
creating two overlapping log files instead of one continuous record.

Usage:
    python merge_tb_logs.py --run1 logs/human_action_1 --run2 logs/human_action_2 --out logs/human_action_merged
    python merge_tb_logs.py --run1 logs/human_action_1 --run2 logs/human_action_2 --out logs/human_action_merged --offset 32000000
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def load_events(log_dir: str) -> list:
    events = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.tfevents.*"))):
        for event in EventFileLoader(path).Load():
            events.append(event)
    events.sort(key=lambda e: (e.wall_time, e.step))
    return events


def max_summary_step(events: list) -> int:
    steps = [e.step for e in events if e.HasField("summary")]
    return max(steps, default=0)


def merge(run1_dir: str, run2_dir: str, out_dir: str, offset: int | None = None):
    print(f"Loading run1: {run1_dir}")
    events1 = load_events(run1_dir)
    print(f"  {len(events1)} events, summary steps "
          f"{min(e.step for e in events1 if e.HasField('summary')):,} → "
          f"{max(e.step for e in events1 if e.HasField('summary')):,}")

    print(f"Loading run2: {run2_dir}")
    events2 = load_events(run2_dir)
    print(f"  {len(events2)} events, summary steps "
          f"{min(e.step for e in events2 if e.HasField('summary')):,} → "
          f"{max(e.step for e in events2 if e.HasField('summary')):,}")

    if offset is None:
        offset = max_summary_step(events1)
    print(f"\nStep offset applied to run2: {offset:,}")

    os.makedirs(out_dir, exist_ok=True)
    writer = EventFileWriter(out_dir)

    # Write run1 as-is
    for event in events1:
        writer.add_event(event)

    # Write run2 with step offset
    for event in events2:
        shifted = event_pb2.Event()
        shifted.CopyFrom(event)
        shifted.step += offset
        writer.add_event(shifted)

    writer.flush()
    writer.close()

    total = max_summary_step(events1) + max_summary_step(events2)
    print(f"\nMerged log written to: {out_dir}")
    print(f"Combined step range: 0 → {total:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", required=True, help="First (earlier) log directory")
    parser.add_argument("--run2", required=True, help="Second (resumed) log directory")
    parser.add_argument("--out",  required=True, help="Output directory for merged log")
    parser.add_argument("--offset", type=int, default=None,
                        help="Step offset for run2 (default: auto = max step of run1)")
    args = parser.parse_args()
    merge(args.run1, args.run2, args.out, args.offset)


if __name__ == "__main__":
    main()
