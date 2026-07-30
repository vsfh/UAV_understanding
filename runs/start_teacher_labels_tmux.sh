#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-clear_teacher_v5}"
GPU_ID="${2:-1}"
LOG_PATH="${REPO_ROOT}/results/teacher_generation/qwen36_35b_v5.log"

if [[ ! "${SESSION_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid tmux session name: ${SESSION_NAME}" >&2
  exit 2
fi
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one integer visible device index" >&2
  exit 2
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "Inspect it with: bash runs/teacher_labels_status.sh ${SESSION_NAME}" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_PATH}")"
tmux new-session \
  -d \
  -s "${SESSION_NAME}" \
  -c "${REPO_ROOT}" \
  "set -o pipefail; CUDA_VISIBLE_DEVICES=${GPU_ID} bash runs/generate_teacher_labels.sh 2>&1 | tee ${LOG_PATH}"

echo "started ${SESSION_NAME} on physical GPU ${GPU_ID}"
echo "log: ${LOG_PATH}"
echo "status: bash runs/teacher_labels_status.sh ${SESSION_NAME}"
