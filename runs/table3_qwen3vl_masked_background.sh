#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HOME="${REPO_ROOT}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec python scripts/table3_qwen3vl_masked_background.py
