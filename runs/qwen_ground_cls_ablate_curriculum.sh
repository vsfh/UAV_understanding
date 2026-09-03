#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/qwen_ground_cls.sh" configs/yaml/qwen_ground_cls_ablate_curriculum.yaml
