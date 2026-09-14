#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Default: fresh SFT, validation calibration, test.
config=${1:-configs/yaml/perception_spatial_lora.yaml}
python scripts/train_perception_spatial.py --config "$config"
python scripts/test_perception_spatial.py --config "$config" --split all
