#!/usr/bin/env bash
set -euo pipefail

MODELS_ROOT="${1:-./models}"
python scripts/download_models.py qwen3-vl --models-root "${MODELS_ROOT}"

