#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HOME="${REPO_ROOT}/hf_cache" TORCH_HOME="${REPO_ROOT}/hf_cache/torch"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec python scripts/table4_dfine_dinov2_t5.py
