#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  cat <<'USAGE'
Usage: bash runs/perception_agents.sh [plan|prepare|train|val|test|all|smoke] [config.yaml] [options]

Default is plan: print the experiment without allocating a model or starting jobs.
all runs proposal preparation, role SFT, validation calibration, then test.
smoke runs limited preparation, one SFT step, and limited validation/test.

Options:
  --output DIR              Output directory; smoke has a separate default.
  --gpu INDEX               Set CUDA_VISIBLE_DEVICES (or export GPU).
  --device DEVICE           e.g. cuda or cpu.
  --mode MODE               full (default), no_verify, verify_only, or all.
  --max-train-samples N      Limit preparation/training images.
  --max-val-samples N        Limit preparation/validation images.
  --max-test-samples N       Limit test images.
  --epochs N                SFT epochs.
  --steps N                 Stop SFT after N optimizer steps.
  --batch-size N            Evaluation batch size only.

Examples:
  bash runs/perception_agents.sh plan
  GPU=0 bash runs/perception_agents.sh smoke
  GPU=0 bash runs/perception_agents.sh all
  GPU=0 bash runs/perception_agents.sh all --mode all
  bash runs/perception_agents.sh test --mode full

PYTHON chooses the existing Python interpreter. This script installs no packages.
USAGE
}

stage=${1:-plan}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in help|-h|--help) usage; exit 0;; plan|prepare|train|val|test|all|smoke) ;; *) usage; exit 2;; esac
config=configs/yaml/perception_agents.yaml
if [[ $# -gt 0 && $1 != --* ]]; then config=$1; shift; fi

mode=full
output=""
device=""
train_limit=""
val_limit=""
test_limit=""
epochs=""
steps=""
batch_size=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --output|--gpu|--device|--mode|--max-train-samples|--max-val-samples|--max-test-samples|--epochs|--steps|--batch-size)
      if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
      option=$1; value=$2; shift 2
      case "$option" in
        --output) output=$value;; --gpu) export CUDA_VISIBLE_DEVICES=$value;;
        --device) device=$value;; --mode) mode=$value;;
        --max-train-samples) train_limit=$value;; --max-val-samples) val_limit=$value;;
        --max-test-samples) test_limit=$value;; --epochs) epochs=$value;;
        --steps) steps=$value;; --batch-size) batch_size=$value;;
      esac;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
case "$mode" in full|no_verify|verify_only|all) ;; *) echo "Unknown evaluation mode: $mode" >&2; exit 2;; esac
if [[ -n ${GPU:-} && -z ${CUDA_VISIBLE_DEVICES:-} ]]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
if [[ -n ${PYTHON:-} ]]; then
  python_bin=$PYTHON
elif [[ -x /home/feihong/miniconda3/bin/python ]]; then
  python_bin=/home/feihong/miniconda3/bin/python
else
  python_bin=python3
fi

if [[ $stage == smoke ]]; then
  output=${output:-./outputs/perception_agents_smoke}
  train_limit=${train_limit:-4}
  val_limit=${val_limit:-4}
  test_limit=${test_limit:-4}
  epochs=${epochs:-1}
  steps=${steps:-1}
fi
common=(--config "$config")
if [[ -n $output ]]; then common+=(--output "$output"); fi
if [[ -n $device ]]; then common+=(--device "$device"); fi
train_args=("${common[@]}")
eval_args=("${common[@]}" --mode "$mode")
if [[ -n $train_limit ]]; then train_args+=(--max-train-samples "$train_limit"); fi
if [[ -n $val_limit ]]; then train_args+=(--max-val-samples "$val_limit"); eval_args+=(--max-val-samples "$val_limit"); fi
if [[ -n $test_limit ]]; then eval_args+=(--max-test-samples "$test_limit"); fi
if [[ -n $epochs ]]; then train_args+=(--epochs "$epochs"); fi
if [[ -n $steps ]]; then train_args+=(--steps "$steps"); fi
if [[ -n $batch_size ]]; then eval_args+=(--batch-size "$batch_size"); fi

case "$stage" in
  plan|prepare|train) exec "$python_bin" scripts/train_perception_agents.py --stage "$stage" "${train_args[@]}";;
  val|test) exec "$python_bin" scripts/test_perception_agents.py --split "$stage" "${eval_args[@]}";;
  all|smoke)
    "$python_bin" scripts/train_perception_agents.py --stage prepare "${train_args[@]}"
    "$python_bin" scripts/train_perception_agents.py --stage train "${train_args[@]}"
    exec "$python_bin" scripts/test_perception_agents.py --split all "${eval_args[@]}";;
esac
