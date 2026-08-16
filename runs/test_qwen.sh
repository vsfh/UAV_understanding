#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${QWEN_PYTHON:-/home/feihong/miniconda3/bin/python}"
exec "${PYTHON}" scripts/test_qwen.py \
  --config "${1:-configs/yaml/qwen_test.yaml}"
