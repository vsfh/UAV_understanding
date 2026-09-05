#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${OPENCLIP_PYTHON:-${HOME}/.conda/envs/uav-openclip/bin/python}"
exec python scripts/test_openclip.py \
  --config "${1:-configs/yaml/openclip_test.yaml}"
