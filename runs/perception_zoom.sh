#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
config=configs/yaml/perception_zoom.yaml
if [[ $# -gt 0 && $1 != --* && $1 != -h ]]; then config=$1; shift; fi
case "$mode" in
  help|-h|--help)
    echo "Usage: bash runs/perception_zoom.sh [prepare|prepare-test|train|val|test|all] [config.yaml] [--epochs N --max-train-samples N --max-val-samples N --max-test-samples N --output PATH]"
    echo "prepare builds train/val proposals; prepare-test is an explicit separate inference stage. all runs both around training and validation."
    exit 0 ;;
esac
source runs/perception_runtime.sh
perception_resolve_python
case "$mode" in
  prepare) "$python_bin" scripts/train_perception_zoom.py --config "$config" --mode prepare --prepare-split train_val "$@" ;;
  prepare-test) "$python_bin" scripts/train_perception_zoom.py --config "$config" --mode prepare --prepare-split test "$@" ;;
  train) "$python_bin" scripts/train_perception_zoom.py --config "$config" "$@" ;;
  val|test) "$python_bin" scripts/test_perception_zoom.py --config "$config" --split "$mode" "$@" ;;
  all)
    "$python_bin" scripts/train_perception_zoom.py --config "$config" --mode prepare --prepare-split train_val "$@"
    "$python_bin" scripts/train_perception_zoom.py --config "$config" "$@"
    "$python_bin" scripts/test_perception_zoom.py --config "$config" --split val "$@"
    "$python_bin" scripts/train_perception_zoom.py --config "$config" --mode prepare --prepare-split test "$@"
    "$python_bin" scripts/test_perception_zoom.py --config "$config" --split test "$@"
    ;;
  *) echo "Usage: bash runs/perception_zoom.sh [prepare|prepare-test|train|val|test|all] [config.yaml] [--epochs N --max-train-samples N --max-val-samples N --max-test-samples N --output PATH]" >&2; exit 2 ;;
esac
