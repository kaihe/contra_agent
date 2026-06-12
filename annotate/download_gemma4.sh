#!/usr/bin/env bash
# Download a Gemma 4 checkpoint into tmp/models/ (resumes if interrupted).
#
# Variants of interest (override with MODEL=...):
#   google/gemma-4-E2B-it-qat-mobile-transformers  wNa8o8 mobile schema, 2.5GB (default)
#   google/gemma-4-E2B-it-qat-w4a16-ct             compressed-tensors, for vLLM serving
#   google/gemma-4-E2B-it                          unquantized bf16
#
# If huggingface.co stalls, prepend: HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E2B-it-qat-mobile-transformers}"
DEST="tmp/models/$(basename "$MODEL")"

mkdir -p "$DEST"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL" --local-dir "$DEST"
else
    huggingface-cli download "$MODEL" --local-dir "$DEST"
fi

echo "done: $DEST"
du -sh "$DEST"
