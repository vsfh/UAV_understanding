#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/media/data1/feihong/uav_understanding_data}"
MODEL_PATH="${2:-/media/data2/feihong/hf_cache/openclip}"
PROMPT="${3:-definition}"
OUTPUT="${4:-./outputs/openclip_session_val.json}"

python scripts/evaluate_clip.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --csv "${DATA_ROOT}/session_disjoint/val.csv" \
  --labels-file ./configs/core18_complete.txt \
  --view context \
  --prompt "${PROMPT}" \
  --output "${OUTPUT}"
