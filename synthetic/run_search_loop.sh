#!/bin/bash
# run_search_loop.sh — randomly search any level forever.
# Press Ctrl+C to stop.

trap 'echo ""; echo "Stopped."; exit 0' SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACE_DIR="${SCRIPT_DIR}/mc_trace"
RUN=0
FIXED_LEVEL="${1:-}"  # optional: pass a level number as first argument

if [ -n "$FIXED_LEVEL" ]; then
    echo "=========================================="
    echo "  MC Search Loop (level ${FIXED_LEVEL} only, forever)"
else
    echo "=========================================="
    echo "  MC Search Loop (random levels, forever)"
fi
echo "  Winning traces → ${TRACE_DIR}"
echo "  Press Ctrl+C to stop."
echo "=========================================="

while true; do
    RUN=$((RUN + 1))
    if [ -n "$FIXED_LEVEL" ]; then
        LEVEL=$FIXED_LEVEL
    else
        LEVEL=$((RANDOM % 8 + 1))
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run #${RUN}   level: ${LEVEL}"
    echo "────────────────────────────────────────"

    python "${SCRIPT_DIR}/mc_search.py" \
        --level ${LEVEL} \
        --goal  level_up
done
