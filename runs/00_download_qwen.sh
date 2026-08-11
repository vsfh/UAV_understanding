#!/usr/bin/env bash
set -euo pipefail

MODELS_ROOT="${1:-/media/data2/feihong/hf_cache}"
python scripts/download_models.py qwen3-vl --models-root "${MODELS_ROOT}"
