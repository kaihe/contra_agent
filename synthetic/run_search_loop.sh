#!/bin/bash
# run_search_loop.sh — randomly search any level forever.
# Press Ctrl+C to stop.

TOTAL_TIME=0
TOTAL_RUNS=0

print_avg_search_time() {
    if [ "${TOTAL_RUNS}" -gt 0 ]; then
        AVG=$(echo "scale=1; ${TOTAL_TIME} / ${TOTAL_RUNS}" | bc)
        echo "  Runs completed : ${TOTAL_RUNS}"
        echo "  Avg search time: ${AVG}s"
    else
        echo "  No runs completed."
    fi
}

trap 'echo ""; echo "Stopped."; print_avg_search_time; exit 0' SIGINT SIGTERM

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

    T_START=$(date +%s%N)
    WORKERS=$(nproc)
    ROLLOUTS=$((WORKERS * 16))
    python "${SCRIPT_DIR}/mc_search.py" \
        --level       ${LEVEL} \
        --goal        level_up \
        --workers     ${WORKERS} \
        --rollouts    ${ROLLOUTS} \
        --rollout-len 64
    T_END=$(date +%s%N)
    ELAPSED=$(echo "scale=1; (${T_END} - ${T_START}) / 1000000000" | bc)
    TOTAL_TIME=$(echo "scale=1; ${TOTAL_TIME} + ${ELAPSED}" | bc)
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo "  Search time: ${ELAPSED}s  (avg over ${TOTAL_RUNS} runs: $(echo "scale=1; ${TOTAL_TIME} / ${TOTAL_RUNS}" | bc)s)"
done
