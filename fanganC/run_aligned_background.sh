#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
run_dir=${1:?Run directory is required}
runtime=/home/omnisky/miniconda3/envs/chart-vllm
python="$runtime/bin/python"
export CUDA_HOME=/usr/local/cuda-12.2
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/data/models/router-python-deps:$PWD/data/msrvtt/source/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
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

service_pid=
stop_service() {
    if [[ -n "$service_pid" ]]; then
        kill -TERM "$service_pid" 2>/dev/null || true
        wait "$service_pid" 2>/dev/null || true
        service_pid=
    fi
}
finish() {
    local status=$?
    stop_service
    printf '%s\n' "$status" > "$run_dir/launcher_exit_code"
}
rm -f -- "$run_dir/launcher_exit_code"
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

align() {
    "$python" -Bu -m fanganC.alignment "$1" --run-dir "$run_dir"
}

start_service() {
    local port=$1 model=$2 log=$3 workers
    shift 3
    if [[ -n "$(ss -H -ltn "sport = :$port")" ]]; then
        printf 'Port %s is occupied\n' "$port" >&2
        return 1
    fi
    workers=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$workers" ]]; then
        printf 'GPU 0 already has a compute process: %s\n' "$workers" >&2
        return 1
    fi
    "$runtime/bin/vllm" serve "$@" --host 127.0.0.1 --port "$port" --dtype bfloat16 \
        --enable-prefix-caching > "$run_dir/$log" 2>&1 9>&- &
    service_pid=$!
    for _ in {1..120}; do
        if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null |
                jq -e --arg model "$model" 'any(.data[]; .id == $model)' >/dev/null 2>&1; then
            return
        fi
        kill -0 "$service_pid" 2>/dev/null || return 1
        sleep 2
    done
    printf 'Model service %s did not become ready on port %s\n' "$model" "$port" >&2
    return 1
}

align prepare
count=$(jq -r '.queries' "$run_dir/alignment/manifest.json")
if [[ ! -f "$run_dir/alignment/hypotheses.jsonl" || $(wc -l < "$run_dir/alignment/hypotheses.jsonl") -ne "$count" ]]; then
    start_service 8001 Qwen/Qwen2.5-7B-Instruct llm_server.log data/models/qwen2.5-7b-instruct \
        --served-model-name Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.80
    align text
    stop_service
fi
if [[ ! -f "$run_dir/alignment/scored_manifest.json" ]] && \
        [[ ! -f "$run_dir/alignment/scores.jsonl" || $(wc -l < "$run_dir/alignment/scores.jsonl") -ne "$count" ]]; then
    vision=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
    start_service 8000 "$vision" vlm_server.log "$vision" --trust-remote-code --max-model-len 6144 \
        --gpu-memory-utilization 0.78 --max-num-batched-tokens 4096 --max-num-seqs 32 \
        --limit-mm-per-prompt '{"image": 8, "video": 0}' --max-logprobs 20
    pilot="$run_dir/verification_smoke"
    if [[ ! -f "$pilot/pilot/test_yesno_k10_metrics.json" ]]; then
        mkdir -p "$pilot/retrieval"
        cp outputs/stage2_clip4clip_qwen3vl8b/retrieval/test.jsonl "$pilot/retrieval/test.jsonl"
        "$python" -Bu scripts/03_pilot_yesno.py --config outputs/stage2_clip4clip_qwen3vl8b/config.yaml \
            --split test --n 50 --set "paths.work_dir=$pilot"
    fi
fi
align score
stop_service
flock -u 9
exec 9>&-
bash fanganC/run_background.sh "$run_dir"
