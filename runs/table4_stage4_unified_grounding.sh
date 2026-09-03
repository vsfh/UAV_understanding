#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${RUNS_DIR}/table4_qwen_ground_cls_t4.sh"
bash "${RUNS_DIR}/test_table4_qwen_ground_cls_t4.sh"
