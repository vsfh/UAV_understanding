#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${OPENCLIP_PYTHON:-${HOME}/.conda/envs/uav-openclip/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/um7}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/openclip_full_suite_e20}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/openclip_full_suite_e20}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-16}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-256}"
LINEAR_FEATURE_BATCH_SIZE="${LINEAR_FEATURE_BATCH_SIZE:-64}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-8}"
FULL_GRADIENT_ACCUMULATION="${FULL_GRADIENT_ACCUMULATION:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-20000}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "OpenCLIP full comparison suite"
echo "Python:                  ${PYTHON_BIN}"
echo "Data:                    ${DATA_ROOT}"
echo "Model:                   ${MODELS_ROOT}/openclip"
echo "GPU:                     ${CUDA_VISIBLE_DEVICES}"
echo "Zero-shot batch:         ${ZERO_SHOT_BATCH_SIZE}"
echo "Linear-probe batch:      ${LINEAR_BATCH_SIZE}"
echo "Linear feature batch:    ${LINEAR_FEATURE_BATCH_SIZE}"
echo "Full-tune micro-batch:   ${FULL_BATCH_SIZE}"
echo "Full-tune accumulation:  ${FULL_GRADIENT_ACCUMULATION}"

exec "${PYTHON_BIN}" scripts/run_experiment_suite.py \
  --profile development \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --paper-tables-dir "${RESULTS_ROOT}/paper_tables" \
  --protocols forward_temporal session_disjoint unseen_site \
  --seeds 42 43 44 \
  --zero-shot-only \
  --zero-shot-models openclip \
  --openclip-finetuning \
  --openclip-batch-size "${ZERO_SHOT_BATCH_SIZE}" \
  --openclip-linear-epochs 20 \
  --openclip-linear-batch-size "${LINEAR_BATCH_SIZE}" \
  --openclip-linear-feature-batch-size "${LINEAR_FEATURE_BATCH_SIZE}" \
  --openclip-linear-learning-rate 1e-3 \
  --openclip-full-epochs 20 \
  --openclip-full-batch-size "${FULL_BATCH_SIZE}" \
  --openclip-full-gradient-accumulation "${FULL_GRADIENT_ACCUMULATION}" \
  --openclip-full-learning-rate 5e-4 \
  --openclip-backbone-learning-rate 1e-5 \
  --openclip-num-workers "${NUM_WORKERS}" \
  --cuda-devices "${CUDA_VISIBLE_DEVICES}" \
  --min-free-gpu-mib "${MIN_FREE_GPU_MIB}" \
  --resume \
  "$@"
