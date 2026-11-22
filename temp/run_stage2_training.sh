#!/usr/bin/env bash
# Run the Stage 2 data and training steps after the MSR-VTT training videos arrive.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

stage2_env=/home/omnisky/miniconda3/envs/chart-vllm
stage2_dir=outputs/stage2_clip4clip_qwen3vl8b
stage2_cfg="$stage2_dir/config.yaml"
stage2_py="$stage2_env/bin/python"
export LD_LIBRARY_PATH="$stage2_env/lib:$stage2_env/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/data/models/router-python-deps${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY https_proxy http_proxy

stage2_service_pid=
stage2_verifier_pid=
stop_text_service() {
    if [ -n "$stage2_service_pid" ]; then
        kill -TERM "$stage2_service_pid" 2>/dev/null || true
        wait "$stage2_service_pid" 2>/dev/null || true
        stage2_service_pid=
    fi
}

stop_verifier() {
    if [ -n "$stage2_verifier_pid" ]; then
        kill -TERM "$stage2_verifier_pid" 2>/dev/null || true
        wait "$stage2_verifier_pid" 2>/dev/null || true
        stage2_verifier_pid=
    fi
}
trap 'stop_text_service; stop_verifier' EXIT

stage2_verifier_model=$("$stage2_py" -B -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["vlm"]["model"])' "$stage2_cfg")
verifier_healthy() {
    curl --noproxy '*' -fsS --max-time 4 http://127.0.0.1:8000/health >/dev/null 2>&1
}

ensure_verifier() {
    if ! verifier_healthy; then
        stop_verifier
        if ss -H -ltn 'sport = :8000' | rg -q .; then
            printf 'Port 8000 is occupied by a service without a healthy verifier endpoint\n' >&2
            return 1
        fi
        CUDA_VISIBLE_DEVICES=1 "$stage2_env/bin/vllm" serve "$stage2_verifier_model" \
            --host 127.0.0.1 --port 8000 --dtype bfloat16 --trust-remote-code \
            --max-model-len 6144 --gpu-memory-utilization 0.78 --max-num-batched-tokens 4096 \
            --max-num-seqs 32 --limit-mm-per-prompt '{"image": 8, "video": 0}' \
            --enable-prefix-caching >> "$stage2_dir/vlm_server.log" 2>&1 &
        stage2_verifier_pid=$!
        for attempt in {1..120}; do
            if verifier_healthy; then break; fi
            kill -0 "$stage2_verifier_pid" 2>/dev/null || return 1
            sleep 2
        done
    fi
    curl --noproxy '*' -fsS --max-time 4 http://127.0.0.1:8000/v1/models |
        jq -e --arg model "$stage2_verifier_model" 'any(.data[]; .id == $model)' >/dev/null
}

printf 'Waiting for the 9,000 training videos\n'
for attempt in {1..1440}; do
    stage2_count=$(find data/msrvtt/videos -maxdepth 1 -name '*.mp4' -printf '.' | wc -c)
    if [ "$stage2_count" -eq 10000 ]; then break; fi
    sleep 10
done
[ "$stage2_count" -eq 10000 ]
printf 'Encoding the 9K training gallery and extracting verifier frames\n'
if [ ! -s "$stage2_dir/retrieval/train.jsonl" ]; then
    "$stage2_py" -Bu scripts/01_extract_frames.py --config "$stage2_cfg" --split train --workers 8 > "$stage2_dir/frames_train.log" 2>&1 &
    stage2_frames_pid=$!
    "$stage2_py" -Bu temp/stage0_clip4clip.py --split train --output "$stage2_dir/clip4clip_train_sim.npy" > "$stage2_dir/clip4clip_train.log" 2>&1
    wait "$stage2_frames_pid"
    "$stage2_py" -Bu scripts/02_retrieve.py --config "$stage2_cfg" --split train --sim_npy "$stage2_dir/clip4clip_train_sim.npy" | tee "$stage2_dir/retrieve_train.log"
fi

stage2_labels="$stage2_dir/router_data/text_labels.jsonl"
if [ ! -s "$stage2_labels" ]; then
    if ss -H -ltn 'sport = :8001' | rg -q .; then
        printf 'Port 8001 is occupied\n' >&2
        exit 1
    fi
    "$stage2_env/bin/vllm" serve data/models/qwen2.5-7b-instruct \
        --served-model-name Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8001 \
        --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.80 \
        --enable-prefix-caching > "$stage2_dir/llm_server.log" 2>&1 &
    stage2_service_pid=$!
    for attempt in {1..120}; do
        if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then break; fi
        kill -0 "$stage2_service_pid"
        sleep 2
    done
    curl --noproxy '*' -fsS --max-time 4 http://127.0.0.1:8001/v1/models | jq -e '.data[0].id == "Qwen/Qwen2.5-7B-Instruct"'
fi

printf 'Scoring the three routes on 3,000 training queries\n'
for score_attempt in {1..3}; do
    ensure_verifier
    "$stage2_py" -Bu scripts/05_build_router_data.py --config "$stage2_cfg" --resume >> "$stage2_dir/build_router_data.log" 2>&1 &
    stage2_build_pid=$!
    while [ ! -s "$stage2_labels" ] && kill -0 "$stage2_build_pid" 2>/dev/null; do
        sleep 5
    done
    stop_text_service
    if wait "$stage2_build_pid"; then break; fi
    if verifier_healthy || [ "$score_attempt" -eq 3 ]; then
        printf 'Stage 2 route scoring failed at attempt %s\n' "$score_attempt" >&2
        exit 1
    fi
    printf 'Verifier stopped during scoring. Restarting from the saved query prefix.\n'
done
[ "$(wc -l < "$stage2_dir/router_data/router_data.jsonl")" -eq 3000 ]
stop_verifier

printf 'Training the Jev-style Qwen3-4B router\n'
"$stage2_py" -Bu scripts/06_train_router.py --config "$stage2_cfg" > "$stage2_dir/train_router.log" 2>&1
[ -s "$stage2_dir/router/best/adapter_model.safetensors" ]
printf 'Stage 2 labels and router training complete\n'
