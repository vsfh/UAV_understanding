#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-clear_teacher_v5}"
OUTPUT_DIR="${REPO_ROOT}/data/targets/qwen36_35b_session_seeds42_43_44_v5"
LOG_PATH="${REPO_ROOT}/results/teacher_generation/qwen36_35b_v5.log"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "status: running (${SESSION_NAME})"
  tmux list-panes \
    -t "${SESSION_NAME}" \
    -F 'pane_pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}'
else
  echo "status: tmux session not running (${SESSION_NAME})"
fi

COMPLETED=0
if [[ -d "${OUTPUT_DIR}/cache" ]]; then
  while IFS= read -r -d '' record_path; do
    if grep -q '"automatic_audit"' "${record_path}" 2>/dev/null; then
      ((COMPLETED += 1))
    fi
  done < <(find "${OUTPUT_DIR}/cache" -type f -name '*.json' -print0)
fi
echo "fully cached records: ${COMPLETED}/5270"

if [[ -f "${OUTPUT_DIR}/generation_summary.json" ]]; then
  echo "final summary: ${OUTPUT_DIR}/generation_summary.json"
else
  echo "final summary: pending"
fi

if [[ -f "${LOG_PATH}" ]]; then
  echo "recent log:"
  tail -n 12 "${LOG_PATH}"
fi
