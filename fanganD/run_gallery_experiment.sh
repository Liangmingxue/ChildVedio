#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
runtime=/home/omnisky/miniconda3/envs/chart-vllm
export PYTHONPATH="$PWD:$PWD/data/models/router-python-deps:$PWD/data/msrvtt/source/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
exec 9>fanganC/runs/gpu0.lock
flock -n 9
exec "$runtime/bin/python" -Bu -m fanganD.gallery_rankers "$1"
