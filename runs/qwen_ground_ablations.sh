#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"

ABLATIONS=(
  qwen_ground_ms_ablate_no_heatmap
  qwen_ground_ms_ablate_single_scale
  qwen_ground_cls_ablate_fixed_classifier
  qwen_ground_cls_ablate_roi
  qwen_ground_cls_ablate_curriculum
  qwen_ground_cls_ablate_global_context
)

for experiment in "${ABLATIONS[@]}"; do
  echo "===== train: ${experiment} ====="
  bash "${RUNS_DIR}/${experiment}.sh"
  echo "===== test:  ${experiment} ====="
  bash "${RUNS_DIR}/test_${experiment}.sh"
done
