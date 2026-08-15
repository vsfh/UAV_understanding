#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./um7}"
MODEL_PATH="${2:-./hf_cache/qwen3-vl}"
ADAPTER_PATH="${3:-./outputs/clear_full_seed42/final}"
VIEW="${4:-pair}"
OUTPUT="${5:-${ADAPTER_PATH%/final}/val_predictions.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/evaluate_qwen.py \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --data-root "${DATA_ROOT}" \
  --csv "${DATA_ROOT}/session_disjoint/val.csv" \
  --labels-file ./configs/core18_complete.txt \
  --view "${VIEW}" \
  --output "${OUTPUT}"
