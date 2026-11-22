#!/usr/bin/env bash
# Complete the full router labels, then train and test Plan B on the selected GPU.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

stage=outputs/stage2_clip4clip_qwen3vl8b
experiments="$stage/routing_experiment"
runtime=/home/omnisky/miniconda3/envs/chart-vllm
python="$runtime/bin/python"
data="$stage/router_data_9000/router_data.jsonl"
gpu=${PLAN_B_GPU:-0}
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    printf 'PLAN_B_GPU must be a physical GPU index\n' >&2
    exit 1
fi

mkdir -p "$experiments"
exec 8>"$experiments/full-plan-b.lock"
if ! flock -n 8; then
    printf 'A full Plan B experiment is already running\n' >&2
    exit 1
fi
run=${1:-$(mktemp -d "$experiments/gpu$gpu-plan-b-full-XXXXXXXX")}
mkdir -p "$run"
if [[ -e "$run/report.json" ]]; then
    printf 'Choose a new experiment directory\n' >&2
    exit 1
fi
printf '%s\n' "$run"
exec >> "$run/run.log" 2>&1
: > "$run/exit_code"
trap 'printf "%s\n" "$?" > "$run/exit_code"' EXIT

if [[ ! -s "$data" ]]; then
    printf '%s Completing the expanded three-route annotations\n' "$(date -Is)"
    bash scripts/run_stage2_full_labels.sh
fi

"$python" -B - "$data" <<'PY'
from collections import Counter
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
counts = Counter(row["split"] for row in rows)
if (len(rows) != 9000 or len({row["qid"] for row in rows}) != 9000
        or counts != {"train": 8123, "dev": 285, "calib": 592}
        or any(set(row["ranks"]) != {"accept", "light", "deep"} for row in rows)):
    raise SystemExit(f"Expanded annotations are incomplete {dict(counts)}")
print(f"Verified expanded router data {dict(counts)}", flush=True)
PY

if [[ -n "${PLAN_B_WAIT_PID:-}" ]]; then
    [[ "$PLAN_B_WAIT_PID" =~ ^[1-9][0-9]*$ ]]
    printf '%s Waiting for GPU %s workflow %s\n' "$(date -Is)" "$gpu" "$PLAN_B_WAIT_PID"
    tail --pid="$PLAN_B_WAIT_PID" --sleep-interval=15 -f /dev/null
fi
workers=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$workers" ]]; then
    printf '%s Waiting for physical GPU %s to become idle\n' "$(date -Is)" "$gpu"
fi
while [[ -n "$workers" ]]; do
    sleep 30
    workers=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
done
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/data/models/router-python-deps${PYTHONPATH:+:$PYTHONPATH}"
printf '%s Running Plan B on physical GPU %s (%s)\n' "$(date -Is)" "$gpu" "$gpu_uuid"
args=(--config "$stage/config.yaml" --device cuda:0 --gain-microbatch 4 --deep-fraction 0.876
      --gain-head factorized --skip-checkpoint-replay --router-data "$data"
      --output-dir "$run")
if [[ -n "${PLAN_B_TEST_HOLDOUT:-}" ]]; then
    args+=(--holdout-fraction "$PLAN_B_TEST_HOLDOUT")
else
    args+=(--test-budget-report "$stage/routing_control/test_matched_budget_prompt.json")
fi
if [[ -n "${PLAN_B_LR:-}" ]]; then
    args+=(--learning-rate "$PLAN_B_LR")
fi
"$python" -Bu scripts/13_stage2_router_experiment.py "${args[@]}"
