#!/bin/bash
# run_search_loop.sh — randomly search any level forever.
# Press Ctrl+C to stop.

trap 'echo ""; echo "Stopped."; exit 0' SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACE_DIR="${SCRIPT_DIR}/mc_trace"
RUN=0

echo "=========================================="
echo "  MC Search Loop (random levels, forever)"
echo "  Winning traces → ${TRACE_DIR}"
echo "  Press Ctrl+C to stop."
echo "=========================================="

while true; do
    RUN=$((RUN + 1))
    LEVEL=$((RANDOM % 8 + 1))

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run #${RUN}   level: ${LEVEL}"
    echo "────────────────────────────────────────"

    conda run --no-capture-output -n vllm-env python "${SCRIPT_DIR}/mc_search.py" \
        --level ${LEVEL} \
        --goal  level_up
done
