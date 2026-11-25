#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run=$1
runtime=/home/omnisky/miniconda3/envs/chart-vllm
export CUDA_HOME=/usr/local/cuda-12.2 FLASHINFER_NVCC=/usr/local/cuda-12.2/bin/nvcc
export PATH="$CUDA_HOME/bin:$runtime/bin:$PATH"
export PYTHONPATH="$PWD/data/models/router-python-deps:$PWD/data/msrvtt/source/python-deps:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
exec 9>fanganC/runs/gpu0.lock
flock -n 9
for gpu in 0 1; do
    test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
done
model=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader) "$runtime/bin/vllm" serve "$model" --host 127.0.0.1 --port 8026 --dtype bfloat16 --enable-prefix-caching --trust-remote-code --max-model-len 6144 --gpu-memory-utilization 0.78 --max-num-batched-tokens 4096 --max-num-seqs 32 --limit-mm-per-prompt '{"image":8,"video":0}' --max-logprobs 20 >"$run/verifier.log" 2>&1 9>&- &
server=$!
trap 'status=$?; kill -TERM "$server" 2>/dev/null || true; echo "$status" > "$run/exit_code"' EXIT
until curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1; do kill -0 "$server"; sleep 3; done
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
"$runtime/bin/python" -Bu -m method2.full "$run"
