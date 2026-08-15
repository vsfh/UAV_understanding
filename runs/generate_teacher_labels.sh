#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-./um7}"
MODEL_PATH="${2:-./hf_cache/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989}"
OUTPUT_DIR="${3:-${REPO_ROOT}/data/targets/qwen36_35b_session_seeds42_43_44_v5}"
EXTRA_ARGS=()
if (( $# > 3 )); then
  EXTRA_ARGS=("${@:4}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${REPO_ROOT}"
python scripts/generate_teacher_targets.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --train-csv "${DATA_ROOT}/session_disjoint/train.csv" \
  --ontology configs/ontology.yaml \
  --labels-file configs/core18_complete.txt \
  --output-dir "${OUTPUT_DIR}" \
  --max-per-class 250 \
  --seeds 42 43 44 \
  "${EXTRA_ARGS[@]}"

DRY_RUN=false
for argument in "${EXTRA_ARGS[@]}"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done
if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

python scripts/export_teacher_review.py \
  --generic-targets "${OUTPUT_DIR}/generic_targets.pending_review.jsonl" \
  --grounded-targets "${OUTPUT_DIR}/grounded_targets.pending_review.jsonl" \
  --output "${OUTPUT_DIR}/human_review.tsv"

echo "Teacher generation complete."
echo "Grounded labels: ${OUTPUT_DIR}/grounded_targets.pending_review.jsonl"
echo "Generic baseline labels: ${OUTPUT_DIR}/generic_targets.pending_review.jsonl"
echo "Automatic audit: ${OUTPUT_DIR}/automatic_audit.jsonl"
echo "Human-review ledger: ${OUTPUT_DIR}/human_review.tsv"
echo "All generated labels remain pending human review."
