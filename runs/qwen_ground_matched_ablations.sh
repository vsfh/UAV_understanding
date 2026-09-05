#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
# No implicit GPU. The Python driver requires GPU_ID for --execute.
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="${GPU_ID}"; fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${REPO_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "${QWEN_PYTHON:-python}" scripts/run_matched_ablations.py "$@"
