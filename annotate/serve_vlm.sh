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
# Default model: local Qwen3-VL-2B-Instruct-FP8 checkpoint.
# Download it first with:
#   ./annotate/download_weight.sh
# Alternatives:
#   MODEL=tmp/models/gemma-4-E2B-it-qat-w4a16-ct
#   MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8
#   MODEL=Qwen/Qwen3-VL-4B-Instruct      (bf16, ~9GB, no quantization)

set -euo pipefail

MODEL="${MODEL:-tmp/models/Qwen3-VL-2B-Instruct-FP8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-serve

# FlashInfer JIT-compiles sampling kernels with the system nvcc; CUDA 12.1
# rejects gcc 13 on this machine. Fall back to the PyTorch-native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve "$MODEL" \
    --served-model-name annotator \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image": 12}'
