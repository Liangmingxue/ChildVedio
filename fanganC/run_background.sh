#!/usr/bin/env bash
# Independent GPU 0 workflow; no model service or annotation process is modified.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
runtime=/home/omnisky/miniconda3/envs/chart-vllm
python="$runtime/bin/python"
export PYTHONPATH="$PWD/data/models/router-python-deps:$PWD/data/msrvtt/source/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
run_dir=${1:-"fanganC/runs/run-$(date +%Y%m%d-%H%M%S)"}
"$python" -B - "$run_dir" <<'PY'
import sys
from fanganC.common import require_local_output
require_local_output(sys.argv[1])
PY
mkdir -p "$run_dir" fanganC/runs
exec 9>fanganC/runs/gpu0.lock
if ! flock -n 9; then
    printf 'Another fanganC workflow already holds GPU 0.\n' >&2
    exit 1
fi
workers=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$workers" ]]; then
    printf 'GPU 0 already has a compute process: %s\n' "$workers" >&2
    exit 1
fi
rm -f -- "$run_dir/exit_code"
printf '%s\n' "$$" > "$run_dir/workflow.pid"
finish() {
    status=$?
    printf '%s\n' "$status" > "$run_dir/exit_code"
}
trap finish EXIT
"$python" -Bu -m fanganC run --run-dir "$run_dir" --device cuda:0 > "$run_dir/workflow.log" 2>&1
