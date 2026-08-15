#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${OPENCLIP_ENV_PREFIX:-${HOME}/.conda/envs/uav-openclip}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if [[ -n "${CONDA_BIN:-}" ]]; then
  CONDA_EXECUTABLE="${CONDA_BIN}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXECUTABLE="$(command -v conda)"
elif [[ -x "${REPO_ROOT}/conda/bin/conda" ]]; then
  CONDA_EXECUTABLE="${REPO_ROOT}/conda/bin/conda"
elif [[ -x "/feihong/miniconda3/bin/conda" ]]; then
  CONDA_EXECUTABLE="/feihong/miniconda3/bin/conda"
else
  echo "Conda was not found." >&2
  echo "Set CONDA_BIN=/absolute/path/to/conda and rerun this script." >&2
  exit 1
fi

echo "Conda:              ${CONDA_EXECUTABLE}"
echo "Environment prefix: ${ENV_PREFIX}"
echo "PyTorch wheel index: ${PYTORCH_INDEX_URL}"

"${CONDA_EXECUTABLE}" create --yes --prefix "${ENV_PREFIX}" python=3.11 pip
"${CONDA_EXECUTABLE}" run --prefix "${ENV_PREFIX}" \
  python -m pip install --upgrade pip setuptools wheel
"${CONDA_EXECUTABLE}" run --prefix "${ENV_PREFIX}" \
  python -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
"${CONDA_EXECUTABLE}" run --prefix "${ENV_PREFIX}" \
  python -m pip install \
    "transformers>=5.8,<5.9" \
    "numpy>=1.26" \
    "pillow>=10" \
    "pyyaml>=6" \
    "tqdm>=4.66"
"${CONDA_EXECUTABLE}" run --prefix "${ENV_PREFIX}" \
  python -m pip install --no-deps --editable "${REPO_ROOT}"

"${CONDA_EXECUTABLE}" run --prefix "${ENV_PREFIX}" python -c \
  'import torch, transformers, clear_uav; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("transformers", transformers.__version__)'

echo "Environment ready. This script did not run any experiment."
echo "Zero-shot only: CUDA_VISIBLE_DEVICES=0 bash runs/31_run_openclip_only.sh"
echo "Full comparison: CUDA_VISIBLE_DEVICES=0 bash runs/32_run_openclip_full_suite_24g.sh"
