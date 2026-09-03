#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
for experiment in table4_grounding_dino_t1 table4_yolo_world_t1 table4_dfine_dinov2_t5; do
  bash "${RUNS_DIR}/${experiment}.sh"
  bash "${RUNS_DIR}/test_${experiment}.sh"
done
