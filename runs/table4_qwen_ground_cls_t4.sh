#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${RUNS_DIR}/qwen_ground_ms.sh" configs/yaml/table4_qwen_ground_ms.yaml
bash "${RUNS_DIR}/qwen_ground_cls.sh" configs/yaml/table4_qwen_ground_cls_t4.yaml
