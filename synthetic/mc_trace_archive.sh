#!/usr/bin/env bash
# Usage:
#   ./mc_trace_archive.sh pack   [output.tar.gz]   — compress synthetic/mc_trace/ into a tar.gz
#   ./mc_trace_archive.sh unpack <archive.tar.gz>   — extract tar.gz into synthetic/mc_trace/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_DIR="$SCRIPT_DIR/mc_trace"

usage() {
    echo "Usage:"
    echo "  $0 pack   [output.tar.gz]"
    echo "  $0 unpack <archive.tar.gz>"
    exit 1
}

[[ $# -lt 1 ]] && usage

MODE="$1"

case "$MODE" in
    pack)
        ARCHIVE="${2:-$SCRIPT_DIR/mc_trace_$(date +%Y%m%d_%H%M%S).tar.gz}"
        echo "Packing $TRACE_DIR -> $ARCHIVE"
        tar -czf "$ARCHIVE" -C "$SCRIPT_DIR" mc_trace
        echo "Done. Archive: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
        ;;
    unpack)
        [[ $# -lt 2 ]] && { echo "Error: unpack requires an archive path."; usage; }
        ARCHIVE="$2"
        [[ ! -f "$ARCHIVE" ]] && { echo "Error: file not found: $ARCHIVE"; exit 1; }
        mkdir -p "$TRACE_DIR"
        echo "Unpacking $ARCHIVE -> $TRACE_DIR"
        tar -xzf "$ARCHIVE" -C "$SCRIPT_DIR"
        echo "Done. Files in $TRACE_DIR: $(find "$TRACE_DIR" -name '*.npz' | wc -l) npz files"
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage
        ;;
esac
