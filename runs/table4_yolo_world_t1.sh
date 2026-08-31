#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HOME="${REPO_ROOT}/hf_cache" TORCH_HOME="${REPO_ROOT}/hf_cache/torch"
YOLO_WORLD_PYTHON="${YOLO_WORLD_PYTHON:-${REPO_ROOT}/hf_cache/yolo-world/venv/bin/python}"
if [[ ! -x "${YOLO_WORLD_PYTHON}" ]]; then
  echo "YOLO-World environment is missing. Run: bash runs/setup_yolo_world_env.sh" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec "${YOLO_WORLD_PYTHON}" scripts/table4_yolo_world_t1.py
