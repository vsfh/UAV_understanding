#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${REPO_ROOT}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${QWEN_PYTHON:-/home/feihong/miniconda3/bin/python}"
exec "${PYTHON}" scripts/test_qwen_ground_cls.py --config configs/yaml/qwen_ground_cls_ablate_hard_negatives.yaml
