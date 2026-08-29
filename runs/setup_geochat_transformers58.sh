#!/usr/bin/env bash
set -euo pipefail

# Download a fresh official GeoChat checkpoint and install its custom model code
# without allowing pip to downgrade Transformers 5.8.0.
#
# This script stages and validates every download before replacing the existing
# checkpoint. It does not make GeoChat a built-in Transformers architecture:
# Python code must still import `geochat.model` before using AutoConfig/AutoModel,
# or use GeoChat's load_pretrained_model helper (as scripts/test_geochat.py does).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CACHE_ROOT="${REPO_ROOT}/hf_cache"
THIRD_PARTY_ROOT="${REPO_ROOT}/third_party"
MODEL_TARGET="${CACHE_ROOT}/geochat"
VISION_TARGET="${CACHE_ROOT}/geochat-vision-tower"
SOURCE_TARGET="${THIRD_PARTY_ROOT}/GeoChat"
PYTHON="${GEOCHAT_PYTHON:-/home/feihong/miniconda3/bin/python}"

if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}" ]]; then
  echo "Invalid repository root: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ "${MODEL_TARGET}" != "${REPO_ROOT}/hf_cache/geochat" ]]; then
  echo "Refusing unexpected model target: ${MODEL_TARGET}" >&2
  exit 1
fi
if [[ "${VISION_TARGET}" != "${REPO_ROOT}/hf_cache/geochat-vision-tower" ]]; then
  echo "Refusing unexpected vision target: ${VISION_TARGET}" >&2
  exit 1
fi
if [[ "${SOURCE_TARGET}" != "${REPO_ROOT}/third_party/GeoChat" ]]; then
  echo "Refusing unexpected source target: ${SOURCE_TARGET}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import transformers

if transformers.__version__ != "5.8.0":
    raise SystemExit(
        f"GEOCHAT_PYTHON must contain Transformers 5.8.0; found {transformers.__version__}"
    )
print(f"Using Transformers {transformers.__version__}")
PY

if command -v hf >/dev/null 2>&1; then
  HF_COMMAND=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_COMMAND=(huggingface-cli)
else
  echo "Neither 'hf' nor 'huggingface-cli' is available." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to download the official GeoChat source." >&2
  exit 1
fi

mkdir -p -- "${CACHE_ROOT}" "${THIRD_PARTY_ROOT}"
STAGE_DIR="${CACHE_ROOT}/.geochat-download-transformers58"
if [[ -e "${STAGE_DIR}" || -L "${STAGE_DIR}" ]]; then
  if [[ -L "${STAGE_DIR}" || ! -d "${STAGE_DIR}" ]]; then
    echo "Refusing unsafe staging path: ${STAGE_DIR}" >&2
    exit 1
  fi
else
  mkdir -- "${STAGE_DIR}"
fi
if [[ -z "${STAGE_DIR}" || "${STAGE_DIR}" != "${CACHE_ROOT}/.geochat-download-transformers58" ]]; then
  echo "Invalid staging directory: ${STAGE_DIR}" >&2
  exit 1
fi

INSTALL_SUCCEEDED=0
cleanup() {
  if [[ "${INSTALL_SUCCEEDED}" == "1" \
        && -n "${STAGE_DIR:-}" \
        && "${STAGE_DIR}" == "${CACHE_ROOT}/.geochat-download-transformers58" \
        && -d "${STAGE_DIR}" \
        && ! -L "${STAGE_DIR}" ]]; then
    rm -rf -- "${STAGE_DIR}"
  elif [[ "${INSTALL_SUCCEEDED}" != "1" ]]; then
    echo "Installation stopped; resumable downloads remain in: ${STAGE_DIR}" >&2
  fi
}
trap cleanup EXIT

MODEL_STAGE="${STAGE_DIR}/geochat"
VISION_STAGE="${STAGE_DIR}/geochat-vision-tower"
SOURCE_STAGE="${STAGE_DIR}/GeoChat"

"${HF_COMMAND[@]}" download MBZUAI/geochat-7B \
  --local-dir "${MODEL_STAGE}"
"${HF_COMMAND[@]}" download openai/clip-vit-large-patch14-336 \
  --local-dir "${VISION_STAGE}"
if [[ -d "${SOURCE_STAGE}/.git" && ! -L "${SOURCE_STAGE}" ]]; then
  git -C "${SOURCE_STAGE}" pull --ff-only
elif [[ -e "${SOURCE_STAGE}" || -L "${SOURCE_STAGE}" ]]; then
  echo "Refusing unexpected source staging path: ${SOURCE_STAGE}" >&2
  exit 1
else
  git clone --depth 1 https://github.com/mbzuai-oryx/GeoChat.git "${SOURCE_STAGE}"
fi

if [[ ! -f "${MODEL_STAGE}/config.json" ]]; then
  echo "Downloaded GeoChat config.json is missing." >&2
  exit 1
fi
if [[ ! -f "${MODEL_STAGE}/pytorch_model.bin.index.json" ]]; then
  echo "Downloaded GeoChat weight index is missing." >&2
  exit 1
fi
if [[ ! -f "${VISION_STAGE}/config.json" ]]; then
  echo "Downloaded vision-tower config.json is missing." >&2
  exit 1
fi
if [[ ! -f "${SOURCE_STAGE}/geochat/model/__init__.py" ]]; then
  echo "Downloaded GeoChat model source is missing." >&2
  exit 1
fi

# The official package imports both its Llama and MPT implementations at module
# import time. GeoChat-7B is Llama-based, while the unused legacy MPT backend
# imports Transformers-private _expand_mask helpers removed before 5.8. Keep
# only the implementation required by this checkpoint.
"${PYTHON}" - "${SOURCE_STAGE}/geochat/model/__init__.py" <<'PY'
import sys
from pathlib import Path

init_path = Path(sys.argv[1])
init_path.write_text(
    "from .language_model.geochat_llama import GeoChatLlamaForCausalLM, GeoChatConfig\n",
    encoding="utf-8",
)
PY

# This import gate runs before deletion. If the official custom code is not
# import-compatible with Transformers 5.8.0, the old checkpoint remains intact.
PYTHONPATH="${SOURCE_STAGE}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" - "${MODEL_STAGE}" <<'PY'
import sys

import transformers
import geochat.model  # Registers model_type="geochat" with the Auto classes.
from transformers import AutoConfig

model_path = sys.argv[1]
if transformers.__version__ != "5.8.0":
    raise SystemExit(f"Expected Transformers 5.8.0, found {transformers.__version__}")
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
if config.model_type != "geochat":
    raise SystemExit(f"Unexpected model type: {config.model_type}")
print("GeoChat custom configuration registered successfully under Transformers 5.8.0")
PY

# Make the checkpoint use the freshly downloaded local vision tower.
"${PYTHON}" - "${MODEL_STAGE}/config.json" "${VISION_TARGET}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
vision_path = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
config["mm_vision_tower"] = str(vision_path)
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

# Refuse links, unexpected resolutions, and non-directory replacement targets.
for target in "${MODEL_TARGET}" "${VISION_TARGET}"; do
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ -L "${target}" || ! -d "${target}" ]]; then
      echo "Refusing non-directory or symbolic-link target: ${target}" >&2
      exit 1
    fi
    resolved="$(realpath -e -- "${target}")"
    if [[ "${resolved}" != "${target}" ]]; then
      echo "Refusing unexpectedly resolved target: ${target} -> ${resolved}" >&2
      exit 1
    fi
    echo "Will replace: ${target}"
  fi
done
if [[ -e "${SOURCE_TARGET}" || -L "${SOURCE_TARGET}" ]]; then
  echo "Source target already exists; refusing to overwrite: ${SOURCE_TARGET}" >&2
  exit 1
fi

# The exact old model directories are removed only after all preceding downloads
# and compatibility checks have succeeded.
rm -rf -- "${MODEL_TARGET}" "${VISION_TARGET}"
mv -- "${MODEL_STAGE}" "${MODEL_TARGET}"
mv -- "${VISION_STAGE}" "${VISION_TARGET}"
mv -- "${SOURCE_STAGE}" "${SOURCE_TARGET}"

# --no-deps preserves Transformers 5.8.0 instead of applying GeoChat's old pin.
"${PYTHON}" -m pip install --no-deps -e "${SOURCE_TARGET}"

PYTHONPATH="${SOURCE_TARGET}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" - "${MODEL_TARGET}" <<'PY'
import sys

import transformers
import geochat.model
from transformers import AutoConfig

config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
assert transformers.__version__ == "5.8.0"
assert config.model_type == "geochat"
print("Installed fresh GeoChat checkpoint and custom code for Transformers 5.8.0")
PY

INSTALL_SUCCEEDED=1
echo "Model: ${MODEL_TARGET}"
echo "Vision tower: ${VISION_TARGET}"
echo "Source: ${SOURCE_TARGET}"
