#!/bin/bash
# run_search_loop.sh — repeatedly run MC search until N wins are collected.
#
# Parameters:
#   LEVEL   : 1-8 for level_up mode, or 'game' for game_clear mode (default: 1)
#   N       : number of wins to collect before stopping (default: 10)
#   MAX_TIME: time budget per run in seconds (default: 3000)
#
# Usage:
#   LEVEL=1 N=10 ./run_search_loop.sh
#   LEVEL=game N=5 MAX_TIME=7200 ./run_search_loop.sh

trap 'echo ""; echo "Stopped."; exit 0' SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEVEL="${LEVEL:-1}"
N="${N:-10}"
MAX_TIME="${MAX_TIME:-3000}"
TRACE_DIR="${SCRIPT_DIR}/mc_trace"
RUN=0
WINS=0

if [ "${LEVEL}" = "game" ]; then
    GOAL="game_clear"
    LEVEL_ARG="1"
    LEVEL_TAG="game"
else
    GOAL="level_up"
    LEVEL_ARG="${LEVEL}"
    LEVEL_TAG="level${LEVEL}"
fi

echo "=========================================="
echo "  MC Search Loop"
echo "  Goal:            ${GOAL}  (LEVEL=${LEVEL})"
echo "  Target wins:     ${N}"
echo "  Budget per run:  ${MAX_TIME}s"
echo "  Winning traces → ${TRACE_DIR}"
echo "=========================================="

while [ "${WINS}" -lt "${N}" ]; do
    RUN=$((RUN + 1))
    WIN_BEFORE=$(ls "${TRACE_DIR}"/win_${LEVEL_TAG}_*.npz 2>/dev/null | wc -l)

    echo ""
    echo "────────────────────────────────────────"
    echo "  Run #${RUN}   wins: ${WINS}/${N}"
    echo "────────────────────────────────────────"

    conda run --no-capture-output -n vllm-env python "${SCRIPT_DIR}/mc_search.py" \
        --level ${LEVEL_ARG} \
        --goal  ${GOAL} \
        --max-time ${MAX_TIME}

    WIN_AFTER=$(ls "${TRACE_DIR}"/win_${LEVEL_TAG}_*.npz 2>/dev/null | wc -l)
    if [ "${WIN_AFTER}" -gt "${WIN_BEFORE}" ]; then
        WINS=$((WINS + 1))
        echo "  ✓ WIN #${WINS}/${N} saved!"
    fi
done

echo ""
echo "=========================================="
echo "  Reached ${N} wins. Done."
echo "=========================================="
