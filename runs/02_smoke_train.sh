#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./um7}"
MODEL_PATH="${2:-./hf_cache/qwen3-vl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/train_qwen.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --train-csv "${DATA_ROOT}/forward_temporal/train.csv" \
  --output-dir ./outputs/smoke \
  --view pair \
  --max-samples 4 \
  --epochs 1 \
  --gradient-accumulation 1 \
  --save-steps 4
