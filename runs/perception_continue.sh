#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
usage() {
  cat <<'USAGE'
Usage: bash runs/perception_continue.sh [check|train|val|test|all] [options]
  check: CPU-only recipe/data/checkpoint inspection; creates no experiment output.
  train: warm-start the original Perception checkpoint for exactly three epochs.
  val:   calibrate the presence threshold using the complete validation set.
  test:  run complete validation calibration followed by test.
  all:   train, calibrate on validation, then test.
Options:
  --gpu ID       One GPU (default: CUDA_VISIBLE_DEVICES, otherwise 1)
  --config PATH  Default configs/yaml/perception_continue.yaml
  --output PATH  New independent output; training refuses an existing directory.
PYTHON selects the interpreter through the existing perception_runtime.sh.
The configuration must match the saved Spatial run. No quick/subset mode exists.
USAGE
}
mode=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in help|-h|--help) usage; exit 0;; check|train|val|test|all) ;; *) usage; exit 2;; esac
config=configs/yaml/perception_continue.yaml
gpu=${CUDA_VISIBLE_DEVICES:-1}
common=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --config|--gpu|--output)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --config) config=$2;;
        --gpu) gpu=$2;;
        --output) common+=(--output "$2");;
      esac
      shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ $gpu =~ ^[0-9]+$ ]] || { echo 'Choose one GPU ID for this matched control' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$gpu
source runs/perception_runtime.sh
perception_resolve_python
common=(--config "$config" "${common[@]}")
case "$mode" in
  check) "$python_bin" scripts/train_perception_continue.py "${common[@]}" --check-only;;
  train) "$python_bin" scripts/train_perception_continue.py "${common[@]}";;
  val) "$python_bin" scripts/test_perception_continue.py "${common[@]}" --split val;;
  test) "$python_bin" scripts/test_perception_continue.py "${common[@]}" --split all;;
  all)
    "$python_bin" scripts/train_perception_continue.py "${common[@]}"
    "$python_bin" scripts/test_perception_continue.py "${common[@]}" --split all
    ;;
esac
