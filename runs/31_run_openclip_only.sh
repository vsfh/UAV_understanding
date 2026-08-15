#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${OPENCLIP_PYTHON:-${HOME}/.conda/envs/uav-openclip/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/um7}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/openclip_only}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/openclip_only}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OPENCLIP_BATCH_SIZE="${OPENCLIP_BATCH_SIZE:-16}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-4096}"

for required_path in \
  "${PYTHON_BIN}" \
  "${DATA_ROOT}" \
  "${MODELS_ROOT}/openclip/config.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    if [[ "${required_path}" == "${PYTHON_BIN}" ]]; then
      echo "Create the environment first: bash runs/setup_openclip_env.sh" >&2
    fi
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "OpenCLIP Python: ${PYTHON_BIN}"
echo "Data root:       ${DATA_ROOT}"
echo "Model root:      ${MODELS_ROOT}"
echo "GPU selection:   ${CUDA_VISIBLE_DEVICES}"
echo "Batch size:      ${OPENCLIP_BATCH_SIZE}"

exec "${PYTHON_BIN}" scripts/run_experiment_suite.py \
  --profile development \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --paper-tables-dir "${RESULTS_ROOT}/paper_tables" \
  --protocols forward_temporal session_disjoint unseen_site \
  --zero-shot-only \
  --zero-shot-models openclip \
  --openclip-batch-size "${OPENCLIP_BATCH_SIZE}" \
  --cuda-devices "${CUDA_VISIBLE_DEVICES}" \
  --min-free-gpu-mib "${MIN_FREE_GPU_MIB}" \
  --resume \
  "$@"
