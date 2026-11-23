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
run_dir=$1
exec 9>fanganC/runs/gpu0.lock
flock -n 9
"$python" -Bu -m fanganD.evidence_jev fanganC/runs/aligned-full-20261122 "$run_dir"
