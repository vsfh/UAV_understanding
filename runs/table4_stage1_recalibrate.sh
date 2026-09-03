#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
for experiment in table4_dfine_t1 table4_florence2_t4; do
  bash "${RUNS_DIR}/${experiment}.sh"
  bash "${RUNS_DIR}/test_${experiment}.sh"
done
