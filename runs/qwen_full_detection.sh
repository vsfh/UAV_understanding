#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE LOCAL_KERNELS
export HF_HOME="${REPO_ROOT}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${QWEN_PYTHON:-/home/feihong/miniconda3/bin/python}"
exec "${PYTHON}" scripts/qwen_full_detection.py \
  --config "${1:-configs/yaml/qwen_full_detection.yaml}"
