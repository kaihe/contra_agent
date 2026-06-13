#!/usr/bin/env bash
# Download a VLM checkpoint into tmp/models/ from ModelScope (resumes if
# interrupted). Model IDs below are verified to exist on modelscope.cn under
# the same namespaces as on HuggingFace.
#
# Variants of interest (override with MODEL=...):
#   Qwen/Qwen3-VL-2B-Instruct-FP8                  3.5GB, vLLM native (default)
#   Qwen/Qwen3-VL-4B-Instruct-FP8                  6.0GB, vLLM native
#   Qwen/Qwen3-VL-8B-Instruct-FP8                 10.6GB, vLLM native
#   google/gemma-4-E2B-it-qat-w4a16-ct             8.4GB, vLLM (compressed-tensors)
#   google/gemma-4-E2B-it-qat-mobile-transformers  2.5GB, transformers only (NOT vLLM)

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct-FP8}"
DEST="tmp/models/$(basename "$MODEL")"

mkdir -p "$DEST"

modelscope download --model "$MODEL" --local_dir "$DEST"

echo "done: $DEST"
du -sh "$DEST"
