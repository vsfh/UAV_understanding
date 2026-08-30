#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HOME="${REPO_ROOT}/hf_cache" TORCH_HOME="${REPO_ROOT}/hf_cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python scripts/test_table4_grounding_dino_t1.py
