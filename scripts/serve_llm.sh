#!/usr/bin/env bash
# Text LLM on GPU0 for fragile labels + counterfactual generation.
#   prompt-only:  bash scripts/serve_llm.sh
#   GRPO LoRA:    MODEL=Qwen/Qwen2.5-3B-Instruct LORA=outputs/cf_grpo/final bash scripts/serve_llm.sh
#                 then use --set cfgen.model=cfgen in 10_run_pipeline.py
# MEM leaves room on GPU0 for the NLI filter (~1 GB).
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
EXTRA=()
if [[ -n "${LORA:-}" ]]; then
  EXTRA=(--enable-lora --max-lora-rank 64 --lora-modules "cfgen=${LORA}")
fi
CUDA_VISIBLE_DEVICES=${GPU:-0} vllm serve "$MODEL" \
  --port "${PORT:-8001}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization "${MEM:-0.80}" \
  --enable-prefix-caching \
  "${EXTRA[@]}"
