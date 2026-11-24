#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_INDEX="$(printenv GPU_INDEX || true)"
MODE="$(printenv MODE || true)"
PYTHON_BIN="$(printenv PYTHON_BIN || true)"
MIN_FREE_MB="$(printenv MIN_FREE_MB || true)"
OUT="$(printenv OUT || true)"
[[ -n "$GPU_INDEX" ]] || GPU_INDEX=0
[[ -n "$MODE" ]] || MODE=frozen
[[ -n "$PYTHON_BIN" ]] || PYTHON_BIN=/home/omnisky/miniconda3/envs/chart-vllm/bin/python
[[ -n "$MIN_FREE_MB" ]] || MIN_FREE_MB=16000
[[ -n "$OUT" ]] || OUT="fanganE/runs/$MODE-$(date +%Y%m%d-%H%M%S)"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to verify the selected GPU" >&2
    exit 2
fi

mkdir -p fanganE/runs
FREE_MB=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -z "$FREE_MB" || "$FREE_MB" -lt "$MIN_FREE_MB" ]]; then
    echo "GPU $GPU_INDEX has $FREE_MB MiB free; need at least $MIN_FREE_MB MiB." >&2
    echo "Stop the text/VLM process occupying this GPU or choose another GPU_INDEX." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | head -n1)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD:$(printenv PYTHONPATH || true)"

exec 9>"fanganE/runs/gpu$GPU_INDEX.lock"
if ! flock -n 9; then
    echo "Another fanganE run holds the GPU $GPU_INDEX lock." >&2
    exit 2
fi

echo "Physical GPU $GPU_INDEX; visible as cuda:0; mode=$MODE; output=$OUT"
exec "$PYTHON_BIN" -Bu -m fanganE \
    --config fanganE/config.yaml \
    --mode "$MODE" \
    --device cuda:0 \
    --out "$OUT" \
    "$@"
