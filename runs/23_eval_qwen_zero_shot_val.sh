#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./um7}"
MODEL_PATH="${2:-./hf_cache/qwen3-vl}"
PROMPT="${3:-definition}"
OUTPUT="${4:-./outputs/qwen_zero_shot_val.jsonl}"
VIEW="${5:-context}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/evaluate_qwen.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --csv "${DATA_ROOT}/session_disjoint/val.csv" \
  --labels-file ./configs/core18_complete.txt \
  --view "${VIEW}" \
  --prompt "${PROMPT}" \
  --output "${OUTPUT}"
