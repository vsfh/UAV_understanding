#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/test_qwen_ground_cls.sh" configs/yaml/table4_qwen_ground_cls_t4.yaml
