#!/usr/bin/env bash
# Follow the MSR-VTT router training job with the held-out evaluation.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

eval_env=/home/omnisky/miniconda3/envs/chart-vllm
eval_dir=outputs/stage2_clip4clip_qwen3vl8b
eval_cfg="$eval_dir/config.yaml"
eval_python="$eval_env/bin/python"
export LD_LIBRARY_PATH="$eval_env/lib:$eval_env/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/data/models/router-python-deps${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY https_proxy http_proxy

exec 9>"$eval_dir/evaluation.lock"
flock -n 9

run_stage() {
    local entry="$1"
    shift
    "$eval_python" -Bu "scripts/$entry.py" --config "$eval_cfg" "$@"
}

eval_service_pids=()
stop_services() {
    local pid
    for pid in "${eval_service_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${eval_service_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap stop_services EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

service_matches() {
    local port="$1" model="$2"
    curl --noproxy '*' -fsS --max-time 4 "http://127.0.0.1:$port/v1/models" 2>/dev/null |
        jq -e --arg model "$model" 'any(.data[]; .id == $model)' >/dev/null
}

ensure_service() {
    local port="$1" model="$2" gpu="$3" logfile="$4"
    shift 4
    if service_matches "$port" "$model"; then return; fi
    if ss -H -ltn "sport = :$port" | rg -q .; then
        printf 'Port %s does not serve the expected model %s\n' "$port" "$model" >&2
        return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$eval_env/bin/vllm" serve "$@" \
        --host 127.0.0.1 --port "$port" --dtype bfloat16 --enable-prefix-caching \
        >> "$eval_dir/$logfile" 2>&1 &
    local service_pid=$!
    eval_service_pids+=("$service_pid")
    for attempt in {1..120}; do
        if service_matches "$port" "$model"; then return; fi
        kill -0 "$service_pid" 2>/dev/null || return 1
        sleep 2
    done
    printf 'Timed out loading %s on port %s\n' "$model" "$port" >&2
    return 1
}

vision_model=$("$eval_python" -B -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["vlm"]["model"])' "$eval_cfg")
ensure_vision() {
    ensure_service 8000 "$vision_model" 1 eval_vlm_server.log "$vision_model" \
        --trust-remote-code --max-model-len 6144 --gpu-memory-utilization 0.78 \
        --max-num-batched-tokens 4096 --max-num-seqs 32 --limit-mm-per-prompt '{"image": 8, "video": 0}'
}

run_light() {
    if [ -s "$eval_dir/pipeline/test_light_metrics.json" ]; then return; fi
    ensure_vision
    printf 'Evaluating always-light on MSR-VTT 1K-A\n'
    run_stage 10_run_pipeline --mode light
}

if [ "$#" -gt 0 ]; then
    if ! ss -H -ltn 'sport = :8000' | rg -q .; then
        run_light
    fi
    printf 'Waiting for router training workflow PID %s\n' "$1"
    tail --pid="$1" --sleep-interval=15 -f /dev/null
fi

adapter="$eval_dir/router/best/adapter_model.safetensors"
if [ ! -s "$adapter" ]; then
    printf 'Trained router adapter missing: %s\n' "$adapter" >&2
    exit 1
fi
if [ ! -s "$eval_dir/router/calib.json" ] || [ "$adapter" -nt "$eval_dir/router/calib.json" ]; then
    printf 'Calibrating the trained router\n'
    run_stage 07_calibrate_router
fi
if [ ! -s "$eval_dir/routes/test.jsonl" ] || [ "$eval_dir/router/calib.json" -nt "$eval_dir/routes/test.jsonl" ]; then
    printf 'Predicting routes for the held-out queries\n'
    run_stage 08_route --split test
fi

run_light
if [ ! -s "$eval_dir/pipeline/test_deep_prompt_metrics.json" ] ||
   [ ! -s "$eval_dir/pipeline/test_router_prompt_metrics.json" ] ||
   [ "$eval_dir/routes/test.jsonl" -nt "$eval_dir/pipeline/test_router_prompt_metrics.json" ]; then
    ensure_vision
    ensure_service 8001 Qwen/Qwen2.5-7B-Instruct 0 eval_llm_server.log data/models/qwen2.5-7b-instruct \
        --served-model-name Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.80
fi
if [ ! -s "$eval_dir/pipeline/test_router_prompt_metrics.json" ] ||
   [ "$eval_dir/routes/test.jsonl" -nt "$eval_dir/pipeline/test_router_prompt_metrics.json" ]; then
    printf 'Evaluating calibrated routing on MSR-VTT 1K-A\n'
    run_stage 10_run_pipeline --mode router --tag prompt
fi
if [ ! -s "$eval_dir/pipeline/test_deep_prompt_metrics.json" ]; then
    printf 'Evaluating always-deep on MSR-VTT 1K-A\n'
    run_stage 10_run_pipeline --mode deep --tag prompt
fi

printf 'MSR-VTT 1K-A evaluation complete\n'
for mode in retriever light router_prompt deep_prompt; do
    jq -c '{mode:.mode,R1:.new["R@1"],R5:.new["R@5"],R10:.new["R@10"],HFR:.flips.HFR,HelpR:.flips.HelpR,vlm_calls_per_query}' \
        "$eval_dir/pipeline/test_${mode}_metrics.json"
done
