#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/test_qwen_ground_cls.sh" configs/yaml/qwen_ground_cls_ablate_fixed_classifier.yaml
