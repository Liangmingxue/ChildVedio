#!/usr/bin/env bash
# Frozen VLM verifier on GPU1 (Qwen2.5-VL-7B bf16 ~16 GB weights; fits a 24 GB 4090).
# If you hit OOM: MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ, or lower --max-model-len / frames.max_side.
# Older vLLM versions use:  --limit-mm-per-prompt image=16
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
CUDA_VISIBLE_DEVICES=${GPU:-1} vllm serve "$MODEL" \
  --port "${PORT:-8000}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_LEN:-6144}" \
  --gpu-memory-utilization "${MEM:-0.92}" \
  --limit-mm-per-prompt '{"image": 16}' \
  --enable-prefix-caching \
  --max-logprobs 20
