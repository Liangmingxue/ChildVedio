#!/usr/bin/env bash
# Independent GPU 0 workflow; no model service or annotation process is modified.
set -euo pipefail
cd -- /media/omnisky/11D1C4300244251B1/xlm/ChildVedio
runtime=/home/omnisky/miniconda3/envs/chart-vllm
python="$runtime/bin/python"
export PYTHONPATH="$PWD/data/models/router-python-deps:$PWD/data/msrvtt/source/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime/lib:$runtime/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
exec 9>fanganC/runs/gpu0.lock
flock -n 9
trap 'echo $? > "$1/exit_code"' EXIT
"$python" -Bu -m fanganC.exchange.run fanganC/runs/aligned-full-20261122 "$1"
