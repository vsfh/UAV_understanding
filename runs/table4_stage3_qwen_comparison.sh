#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
for experiment in table4_qwen3vl_t4 table4_qwen3vl_tool_agent; do
  bash "${RUNS_DIR}/${experiment}.sh"
  bash "${RUNS_DIR}/test_${experiment}.sh"
done
