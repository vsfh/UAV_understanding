#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/qwen_ground_ms.sh" configs/yaml/qwen_ground_ms_ablate_single_scale.yaml
