#!/usr/bin/env bash
set -euo pipefail
# From the repository root; choose GPU with CUDA_VISIBLE_DEVICES.
python scripts/run_spatial_evidence.py --config configs/yaml/spatial_evidence.yaml "$@"
