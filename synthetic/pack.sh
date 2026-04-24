#!/usr/bin/env bash
# Usage:
#   ./pack.sh pack   mc_trace  [output.tar.gz]   — compress synthetic/mc_trace/
#   ./pack.sh pack   mc_graph  [output.tar.gz]   — compress synthetic/mc_graph/
#   ./pack.sh unpack mc_trace  <archive.tar.gz>  — extract into synthetic/mc_trace/
#   ./pack.sh unpack mc_graph  <archive.tar.gz>  — extract into synthetic/mc_graph/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage:"
    echo "  $0 pack   mc_trace|mc_graph  [output.tar.gz]"
    echo "  $0 unpack mc_trace|mc_graph  <archive.tar.gz>"
    exit 1
}

[[ $# -lt 2 ]] && usage

MODE="$1"
TARGET="$2"

case "$TARGET" in
    mc_trace|mc_graph) ;;
    *) echo "Error: target must be mc_trace or mc_graph"; usage ;;
esac

TARGET_DIR="$SCRIPT_DIR/$TARGET"

case "$MODE" in
    pack)
        ARCHIVE="${3:-$SCRIPT_DIR/${TARGET}_$(date +%Y%m%d_%H%M%S).tar.gz}"
        echo "Packing $TARGET_DIR -> $ARCHIVE"
        tar -czf "$ARCHIVE" -C "$SCRIPT_DIR" "$TARGET"
        echo "Done. Archive: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
        ;;
    unpack)
        [[ $# -lt 3 ]] && { echo "Error: unpack requires an archive path."; usage; }
        ARCHIVE="$3"
        [[ ! -f "$ARCHIVE" ]] && { echo "Error: file not found: $ARCHIVE"; exit 1; }
        mkdir -p "$TARGET_DIR"
        echo "Unpacking $ARCHIVE -> $TARGET_DIR"
        tar -xzf "$ARCHIVE" -C "$SCRIPT_DIR"
        echo "Done. Files in $TARGET_DIR: $(ls "$TARGET_DIR" | wc -l) files"
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage
        ;;
esac
