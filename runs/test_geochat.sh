#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${GEOCHAT_PYTHON:-/home/feihong/miniconda3/bin/python}"
exec python scripts/test_geochat.py \
  --config "${1:-configs/yaml/geochat_test.yaml}"
