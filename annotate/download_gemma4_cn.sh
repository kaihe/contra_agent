#!/usr/bin/env bash
# Download a Gemma 4 checkpoint for China networks.
#
# Tries ModelScope (modelscope.cn) first, then falls back to the Hugging Face
# mirror hf-mirror.com. Resumes partial downloads.
#
# Variants of interest (override with MODEL=...):
#   google/gemma-4-E2B-it-qat-mobile-transformers  wNa8o8 mobile schema, 2.5GB (default)
#   google/gemma-4-E2B-it-qat-w4a16-ct             compressed-tensors, for vLLM serving
#   google/gemma-4-E2B-it                          unquantized bf16
#
# If the ModelScope model id differs from the Hugging Face one, override it with:
#   MODELSCOPE_MODEL=google/gemma-4-E2B-it MODEL=google/gemma-4-E2B-it ./annotate/download_gemma4_cn.sh

set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E2B-it-qat-mobile-transformers}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-$MODEL}"
DEST="tmp/models/$(basename "$MODEL")"

mkdir -p "$DEST"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

ensure_modelscope() {
    if ! python -c "import modelscope" >/dev/null 2>&1; then
        echo "Installing modelscope SDK..."
        pip install -q modelscope
    fi
}

download_modelscope() {
    echo "Downloading from ModelScope: $MODELSCOPE_MODEL -> $DEST"
    ensure_modelscope
    python - <<PY
from modelscope import snapshot_download
snapshot_download(
    "$MODELSCOPE_MODEL",
    local_dir="$DEST",
)
PY
}

download_hf_mirror() {
    echo "Falling back to Hugging Face mirror: $MODEL -> $DEST"
    export HF_ENDPOINT=https://hf-mirror.com
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL" --local-dir "$DEST"
    else
        huggingface-cli download "$MODEL" --local-dir "$DEST"
    fi
}

if ! download_modelscope; then
    download_hf_mirror
fi

echo "done: $DEST"
du -sh "$DEST"
