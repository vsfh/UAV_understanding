#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/media/data1/feihong/uav_understanding_data}"
MODELS_ROOT="${MODELS_ROOT:-/media/data2/feihong/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper_tables}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/paper_tables}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
MULTI_LOSS_BATCH_SIZE="${MULTI_LOSS_BATCH_SIZE:-1}"
MULTI_LOSS_GRADIENT_ACCUMULATION="${MULTI_LOSS_GRADIENT_ACCUMULATION:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-40000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec python scripts/run_experiment_suite.py \
  --profile development \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --paper-tables-dir "${RESULTS_ROOT}/paper_tables" \
  --cropped-captions-root description \
  --protocols forward_temporal session_disjoint unseen_site \
  --seeds 42 43 44 \
  --cuda-devices "${CUDA_DEVICES}" \
  --min-free-gpu-mib "${MIN_FREE_GPU_MIB}" \
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
  --multi-loss-batch-size "${MULTI_LOSS_BATCH_SIZE}" \
  --multi-loss-gradient-accumulation "${MULTI_LOSS_GRADIENT_ACCUMULATION}" \
  --max-length "${MAX_LENGTH}" \
  --max-pixels "${MAX_PIXELS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --resume \
  "$@"
