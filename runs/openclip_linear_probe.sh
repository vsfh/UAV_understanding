#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${OPENCLIP_PYTHON:-${HOME}/.conda/envs/uav-openclip/bin/python}"
exec "${PYTHON}" scripts/openclip_linear_probe.py \
  --config "${1:-configs/yaml/openclip_linear_probe.yaml}"
