#!/usr/bin/env bash
# Extend the Stage 2 labels and save the result beside the original dataset.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

stage_dir=outputs/stage2_clip4clip_qwen3vl8b
cfg="$stage_dir/config.yaml"
original="$stage_dir/router_data/router_data.jsonl"
output="$stage_dir/router_data_9000"
runtime=/home/omnisky/miniconda3/envs/chart-vllm
python="$runtime/bin/python"
export CUDA_HOME=/usr/local/cuda-12.2
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/data/models/router-python-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

mkdir -p "$output"
exec 9>"$output/workflow.lock"
if ! flock -n 9; then
    printf 'An annotation workflow is already using %s\n' "$output" >&2
    exit 1
fi
exec >> "$output/workflow.log" 2>&1
: > "$output/exit_code"
printf '%s\n' "$$" > "$output/workflow.pid"
service_pid=
builder_pid=

stop_service() {
    if [[ -n "$service_pid" ]]; then
        kill -TERM "$service_pid" 2>/dev/null || true
        wait "$service_pid" 2>/dev/null || true
        service_pid=
    fi
}

finish() {
    local status=$?
    trap - EXIT
    if [[ -n "$builder_pid" ]]; then
        kill -TERM "$builder_pid" 2>/dev/null || true
        wait "$builder_pid" 2>/dev/null || true
    fi
    stop_service
    printf '%s\n' "$status" > "$output/exit_code"
    printf '%s Workflow exited with status %s\n' "$(date -Is)" "$status"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

[[ "$(wc -l < "$stage_dir/retrieval/train.jsonl")" -eq 9000 ]]
"$python" -B - "$original" <<'PY'
from collections import Counter
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    counts = Counter(json.loads(line)["split"] for line in source if line.strip())
if counts != {"train": 2123, "dev": 285, "calib": 592}:
    raise SystemExit(f"Unexpected splits in the original annotations {dict(counts)}")
PY
text_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
vision_uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
text_model=$("$python" -Bc 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["llm"]["model"])' "$cfg")
vision_model=$("$python" -Bc 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["vlm"]["model"])' "$cfg")

start_service() {
    local phase=$1 gpu uuid port model log workers listener
    local -a options
    if [[ "$phase" == text ]]; then
        gpu=0 uuid="$text_uuid" port=8001 model="$text_model" log=llm_server.log
        options=(data/models/qwen2.5-7b-instruct --served-model-name "$model" --max-model-len 4096
                 --gpu-memory-utilization 0.80)
    else
        gpu=1 uuid="$vision_uuid" port=8000 model="$vision_model" log=vlm_server.log
        options=("$model" --trust-remote-code --max-model-len 6144 --gpu-memory-utilization 0.78
                 --max-num-batched-tokens 4096 --max-num-seqs 32
                 --limit-mm-per-prompt '{"image": 8, "video": 0}')
    fi
    listener=$(ss -H -ltn "sport = :$port") || return 1
    if [[ -n "$listener" ]]; then
        printf 'Port %s is occupied\n' "$port" >&2
        return 1
    fi
    workers=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) || return 1
    if [[ -n "$workers" ]]; then
        printf 'GPU %s is already running a compute process\n' "$gpu" >&2
        return 1
    fi
    printf '%s Starting %s service on physical GPU %s\n' "$(date -Is)" "$phase" "$gpu"
    options+=(--host 127.0.0.1 --port "$port" --dtype bfloat16 --enable-prefix-caching)
    CUDA_VISIBLE_DEVICES="$uuid" "$runtime/bin/vllm" serve "${options[@]}" >> "$output/$log" 2>&1 &
    service_pid=$!
    for _ in {1..120}; do
        if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 &&
           curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" |
               jq -e --arg model "$model" 'any(.data[]; .id == $model)' >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$service_pid" 2>/dev/null || return 75
        sleep 2
    done
    printf '%s service did not become ready on port %s\n' "$phase" "$port" >&2
    return 75
}

run_builder() {
    local phase=$1 status=0
    local -a args=(-Bu scripts/05_build_router_data.py --config "$cfg" --extend-from "$original"
                   --output-dir "$output" --phase "$phase" --resume)
    CUDA_VISIBLE_DEVICES="$text_uuid" "$python" "${args[@]}" &
    builder_pid=$!
    wait "$builder_pid" || status=$?
    builder_pid=
    return "$status"
}

run_phase() {
    local phase=$1 checkpoint=$2 attempt status
    if [[ -f "$checkpoint" && "$(wc -l < "$checkpoint")" -eq 6000 ]]; then
        run_builder "$phase"
        return
    fi
    for attempt in {1..3}; do
        status=0
        if start_service "$phase"; then
            if run_builder "$phase"; then
                stop_service
                return 0
            else
                status=$?
            fi
            if kill -0 "$service_pid" 2>/dev/null; then
                return "$status"
            fi
        else
            status=$?
            if [[ "$status" -ne 75 ]]; then
                return "$status"
            fi
        fi
        stop_service
        printf '%s %s service failed on attempt %s/3. Saved progress is retained.\n' "$(date -Is)" "$phase" "$attempt"
    done
    return "$status"
}

run_phase text "$output/text_labels.jsonl"
run_phase score "$output/progress.jsonl"
[[ "$(wc -l < "$output/router_data.jsonl")" -eq 9000 ]]
