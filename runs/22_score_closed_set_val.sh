#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/media/data1/feihong/uav_understanding_data}"
MODEL_PATH="${2:-./models/qwen3-vl}"
ADAPTER_PATH="${3:-./outputs/clear_full_seed42/final}"
VIEW="${4:-pair}"
OUTPUT="${5:-${ADAPTER_PATH%/final}/val_closed_set.json}"
CANDIDATE_BATCH_SIZE="${6:-4}"
SET_AGGREGATOR="${7:-logsumexp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/evaluate_closed_set.py \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --data-root "${DATA_ROOT}" \
  --csv "${DATA_ROOT}/session_disjoint/val.csv" \
  --labels-file ./configs/core18_complete.txt \
  --view "${VIEW}" \
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --set-aggregator "${SET_AGGREGATOR}" \
  --fit-thresholds \
  --output "${OUTPUT}"
