#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./um7}"
MODEL_PATH="${2:-./hf_cache/qwen3-vl}"
OUTPUT_DIR="${3:-./outputs/evidence_label_seed42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/train_qwen.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --train-csv "${DATA_ROOT}/session_disjoint/train.csv" \
  --labels-file ./configs/core18_complete.txt \
  --max-per-class 250 \
  --output-dir "${OUTPUT_DIR}" \
  --view evidence \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --seed 42
