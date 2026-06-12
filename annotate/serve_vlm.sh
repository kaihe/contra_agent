#!/usr/bin/env bash
# Serve a VLM for thought annotation (OpenAI-compatible API on :8000).
#
# One-time env setup (isolated from vllm-env, pulls its own torch):
#   conda create -n qwen-serve python=3.12 -y
#   conda activate qwen-serve
#   pip install vllm
#
# The GPU is exclusive while this runs — stop the server before training.
#
# Default model: local Gemma 4 E2B w4a16 checkpoint (compressed-tensors, the
# only Gemma 4 QAT serialization vLLM can load). Download it first with:
#   MODEL=google/gemma-4-E2B-it-qat-w4a16-ct ./annotate/download_gemma4.sh
# NOTE: the qat-mobile (wNa8o8) build does NOT work with vLLM (its
# quant_method "gemma" targets phone NPUs) — use it via transformers only.
# Alternatives:
#   MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8
#   MODEL=Qwen/Qwen3-VL-4B-Instruct      (bf16, ~9GB, no quantization)

set -euo pipefail

MODEL="${MODEL:-tmp/models/gemma-4-E2B-it-qat-w4a16-ct}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-serve

exec vllm serve "$MODEL" \
    --served-model-name annotator \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image": 12}'
