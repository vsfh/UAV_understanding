#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODELS_ROOT="${1:-./hf_cache}"

if [[ "${BYPASS_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  echo "download network: direct connection (proxy variables unset)"
fi

if [[ "${USE_HF_MIRROR:-0}" == "1" ]]; then
  DOWNLOAD_ARGS=(--backend aria2 --endpoint https://hf-mirror.com)
  echo "download endpoint: https://hf-mirror.com (aria2 resumable backend)"
elif [[ "${HF_XET_HIGH_PERFORMANCE:-0}" == "1" ]]; then
  export HF_XET_HIGH_PERFORMANCE=1
  DOWNLOAD_ARGS=(--backend huggingface)
  echo "download endpoint: Hugging Face official with Xet high-performance mode"
else
  DOWNLOAD_ARGS=(--backend huggingface)
  echo "download endpoint: ${HF_ENDPOINT:-https://huggingface.co}"
fi

cd "${REPO_ROOT}"
exec python scripts/download_models.py all --models-root "${MODELS_ROOT}" "${DOWNLOAD_ARGS[@]}"
