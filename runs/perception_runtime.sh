#!/usr/bin/env bash
# Source this file after changing to the repository root. No model/job starts here.
perception_resolve_python() {
  if [[ -n ${PYTHON:-} ]]; then
    python_bin=$PYTHON
  elif [[ -n ${PYTHON_BIN:-} ]]; then
    python_bin=$PYTHON_BIN
  elif command -v python >/dev/null 2>&1; then
    python_bin=$(command -v python)
  elif [[ -x ${HOME:-}/miniconda3/bin/python ]]; then
    python_bin=${HOME}/miniconda3/bin/python
  else
    echo "No Python runtime found. Set PYTHON=/path/to/python (remote: /home/feihong/miniconda3/bin/python)." >&2
    return 127
  fi
  if [[ ! -x $python_bin ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python is not executable: $python_bin" >&2
    return 127
  fi
  export PYTHON="$python_bin"
}
