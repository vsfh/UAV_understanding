#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-python}"
VENV_DIR="${REPO_ROOT}/hf_cache/yolo-world/venv"

"${BASE_PYTHON}" - <<'PY'
import transformers

if transformers.__version__ != "5.8.0":
    raise SystemExit(
        f"Expected base Transformers 5.8.0, found {transformers.__version__}"
    )
PY

"${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install "ultralytics>=8.3,<9"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import ultralytics
from ultralytics import YOLOWorld

assert transformers.__version__ == "5.8.0"
print(f"Ultralytics {ultralytics.__version__}")
print(f"Transformers {transformers.__version__} (unchanged)")
print(f"Torch {torch.__version__}")
print(f"YOLOWorld {YOLOWorld.__name__} ready")
PY
