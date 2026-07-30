#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${REPO_ROOT}/data/targets/qwen36_35b_session_seeds42_43_44_v5}"
REVIEW_TSV="${2:-${OUTPUT_DIR}/human_review.tsv}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

python scripts/finalize_teacher_reviews.py \
  --generic-targets "${OUTPUT_DIR}/generic_targets.pending_review.jsonl" \
  --grounded-targets "${OUTPUT_DIR}/grounded_targets.pending_review.jsonl" \
  --reviews "${REVIEW_TSV}" \
  --generic-output "${OUTPUT_DIR}/generic_targets.human_audited.jsonl" \
  --grounded-output "${OUTPUT_DIR}/grounded_targets.human_audited.jsonl"

echo "Human-audited teacher targets are ready in ${OUTPUT_DIR}."
