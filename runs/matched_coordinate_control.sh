#!/usr/bin/env bash
set -euo pipefail
arm=${1:-all}
config=${2:-configs/yaml/matched_coordinate_control.yaml}
case "$arm" in
  all) arms=(token continuous) ;;
  token|continuous) arms=("$arm") ;;
  *) echo 'Usage: bash runs/matched_coordinate_control.sh [all|token|continuous] [config.yaml]'; exit 2 ;;
esac
for variant in "${arms[@]}"; do
  python scripts/train_matched_coordinate_control.py --config "$config" --arm "$variant"
  python scripts/test_matched_coordinate_control.py --config "$config" --arm "$variant"
done
