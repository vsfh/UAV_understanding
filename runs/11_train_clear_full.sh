#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/media/data1/feihong/uav_understanding_data}"
MODEL_PATH="${2:-./models/qwen3-vl}"
OUTPUT_DIR="${3:-./outputs/clear_full_seed42}"
TARGETS_JSONL="${4:?pass the adjudicated targets JSONL as argument 4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/train_qwen.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --train-csv "${DATA_ROOT}/session_disjoint/train.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --labels-file ./configs/core18_complete.txt \
  --max-per-class 250 \
  --targets-jsonl "${TARGETS_JSONL}" \
  --require-audited-targets \
  --view pair \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lambda-neighbor 0.1 \
  --lambda-cf 0.1 \
  --context-dropout 0.1 \
  --evidence-dropout 0.1 \
  --seed 42
