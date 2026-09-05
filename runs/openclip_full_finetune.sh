#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${OPENCLIP_PYTHON:-${HOME}/.conda/envs/uav-openclip/bin/python}"
exec python scripts/openclip_finetune.py \
  --config "${1:-configs/yaml/openclip_full_finetune.yaml}"
