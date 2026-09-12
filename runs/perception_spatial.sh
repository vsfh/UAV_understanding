#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
usage() {
  cat <<'USAGE'
Usage: bash runs/perception_spatial.sh [train|val|test|all] [config.yaml] [options]
Options:
  --output PATH                 Separate experiment output directory
  --initial-checkpoint PATH     Training warm-start adapter plus box head
  --epochs N                    Number of SFT epochs
  --max-train-samples N          Limit training data for smoke runs
  --max-val-samples N            Limit validation data for smoke runs
  --max-test-samples N           Limit test data for smoke runs
  -h, --help                    Print help
Use CUDA_VISIBLE_DEVICES to choose the GPU, and PYTHON to choose the interpreter.
Example:
  CUDA_VISIBLE_DEVICES=0 bash runs/perception_spatial.sh all
  bash runs/perception_spatial.sh all --output ./outputs/spatial_smoke --epochs 1 --max-train-samples 4 --max-val-samples 2 --max-test-samples 2
USAGE
}
mode=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in help|-h|--help) usage; exit 0;; train|val|test|all) ;; *) usage; exit 2;; esac
config=configs/yaml/perception_spatial.yaml
if [[ $# -gt 0 && $1 != --* && $1 != -h ]]; then config=$1; shift; fi
train_args=()
eval_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --output|--max-val-samples)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      train_args+=("$1" "$2"); eval_args+=("$1" "$2"); shift 2;;
    --initial-checkpoint|--epochs|--max-train-samples)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      train_args+=("$1" "$2"); shift 2;;
    --max-test-samples)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      eval_args+=("$1" "$2"); shift 2;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
source runs/perception_runtime.sh
perception_resolve_python
case "$mode" in
  train) "$python_bin" scripts/train_perception_spatial.py --config "$config" "${train_args[@]}";;
  val|test) "$python_bin" scripts/test_perception_spatial.py --config "$config" --split "$mode" "${eval_args[@]}";;
  all)
    "$python_bin" scripts/train_perception_spatial.py --config "$config" "${train_args[@]}"
    "$python_bin" scripts/test_perception_spatial.py --config "$config" --split all "${eval_args[@]}"
    ;;
esac
