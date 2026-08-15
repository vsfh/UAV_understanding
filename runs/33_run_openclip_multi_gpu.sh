#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${OPENCLIP_PYTHON:-${HOME}/.conda/envs/uav-openclip/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/um7}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/openclip_multi_gpu_e20_20g}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/openclip_multi_gpu_e20_20g}"
GPU_IDS="${GPU_IDS:-}"

for required_path in \
  "${PYTHON_BIN}" \
  "${DATA_ROOT}" \
  "${MODELS_ROOT}/openclip/config.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

GPU_ARGS=()
if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
  GPU_ARGS=(--gpus "${GPU_ARRAY[@]}")
fi

unset CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "OpenCLIP heterogeneous multi-GPU suite"
echo "GPU IDs: ${GPU_IDS:-all detected GPUs}"
echo "Data:    ${DATA_ROOT}"
echo "Models:  ${MODELS_ROOT}"

exec "${PYTHON_BIN}" scripts/run_openclip_multi_gpu.py \
  "${GPU_ARGS[@]}" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --protocols forward_temporal session_disjoint unseen_site \
  --seeds 42 43 44 \
  "$@"
