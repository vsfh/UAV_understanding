#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/test_qwen_ground_ms.sh" configs/yaml/qwen_ground_ms_ablate_no_heatmap.yaml
