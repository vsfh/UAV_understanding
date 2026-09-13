#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
usage() {
  cat <<'USAGE'
Usage: bash runs/perception_agents_dual.sh [plan|check|smoke|benchmark|tune|fast|train|val|test|all] [options]
  all: train from existing proposals, then full val calibration and test.
  fast: benchmark four global-batch-8 settings, choose measured fastest, then all.
  tune: benchmark only; print selected.yaml. Never generates candidates.
Options:
  --config PATH              Default configs/yaml/perception_agents_dual_pro6000.yaml
  --gpus 0,1                 Two CUDA device IDs (default 0,1)
  --output DIR               New training output, independent of proposal cache
  --proposal-cache DIR       Completed train.json/val.json/manifest.json/portable_samples.json
  --batch-size N             Per-GPU training image batch (default 2)
  --gradient-accumulation N  Default 2; 2 GPUs * local batch * accumulation = global batch
  --gradient-checkpointing | --no-gradient-checkpointing
  --eval-batch-size N        Per-GPU inference batch (default 4)
  --epochs N | --steps N     Training or benchmark limits
  --mode full|all|no_verify|verify_only  Default full
  --memory-gib N             tune/fast memory ceiling, default 76 GiB per card
PYTHON selects the active environment's Python. No packages are installed.
USAGE
}
stage=${1:-plan}
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in help|-h|--help) usage; exit 0;; plan|check|smoke|benchmark|tune|fast|train|val|test|all) ;; *) usage; exit 2;; esac
config=configs/yaml/perception_agents_dual_pro6000.yaml
gpus=0,1
output=""
cache=""
mode=full
batch=""
accumulation=""
checkpointing=""
eval_batch=""
epochs=""
steps=""
memory=76
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0;;
    --gradient-checkpointing|--no-gradient-checkpointing) checkpointing=$1; shift;;
    --config|--gpus|--gpu|--output|--proposal-cache|--mode|--batch-size|--gradient-accumulation|--eval-batch-size|--epochs|--steps|--memory-gib)
      if [[ $# -lt 2 ]]; then echo "Missing value: $1" >&2; exit 2; fi
      option=$1; value=$2; shift 2
      case "$option" in
        --config) config=$value;; --gpus|--gpu) gpus=$value;; --output) output=$value;;
        --proposal-cache) cache=$value;; --mode) mode=$value;; --batch-size) batch=$value;;
        --gradient-accumulation) accumulation=$value;; --eval-batch-size) eval_batch=$value;;
        --epochs) epochs=$value;; --steps) steps=$value;; --memory-gib) memory=$value;;
      esac;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
case "$mode" in full|all|no_verify|verify_only) ;; *) echo "Invalid mode: $mode" >&2; exit 2;; esac
if [[ ! $gpus =~ ^[0-9]+,[0-9]+$ || ${gpus%,*} == ${gpus#*,} ]]; then
  echo '--gpus needs two different IDs, e.g. 0,1' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=$gpus
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python_bin=${PYTHON:-python}
launch=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2)
common=(--config "$config")
if [[ -n $output ]]; then common+=(--output "$output"); fi
train_args=("${common[@]}")
eval_args=("${common[@]}" --mode "$mode")
if [[ -n $cache ]]; then train_args+=(--proposal-cache "$cache"); fi
if [[ -n $batch ]]; then train_args+=(--batch-size "$batch"); fi
if [[ -n $accumulation ]]; then train_args+=(--gradient-accumulation "$accumulation"); fi
if [[ -n $checkpointing ]]; then train_args+=("$checkpointing"); fi
if [[ -n $epochs ]]; then train_args+=(--epochs "$epochs"); fi
if [[ -n $steps ]]; then train_args+=(--steps "$steps"); fi
if [[ -n $eval_batch ]]; then eval_args+=(--batch-size "$eval_batch"); fi
if [[ $stage == plan ]]; then
  exec "$python_bin" scripts/train_perception_agents_dual.py --stage plan "${train_args[@]}"
fi
if [[ $stage == check ]]; then
  exec "$python_bin" - <<'PY'
import json,torch,transformers
devices=[{'index':i,'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i),
          'total_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]
print(json.dumps({'torch':torch.__version__,'cuda_build':torch.version.cuda,'transformers':transformers.__version__,
                  'cuda_arch_list':torch.cuda.get_arch_list(),'devices':devices},indent=2))
if len(devices)!=2: raise SystemExit('Exactly two visible GPUs are required')
if any(d['capability'][0]>=10 for d in devices) and tuple(map(int,torch.version.cuda.split('.')[:2]))<(12,8):
    raise SystemExit('Blackwell needs a PyTorch CUDA 12.8+ build; the old cu124 environment cannot be copied unchanged')
PY
fi
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/perception_agents_dual_logs
log="outputs/perception_agents_dual_logs/${stage}_${stamp}.log"
exec > >(tee -a "$log") 2>&1
echo "Log: $log"
case "$stage" in
  tune|fast)
    if [[ -n $batch || -n $accumulation || -n $checkpointing || -n $epochs || -n $eval_batch ]]; then
      echo 'tune/fast uses YAML plus its measured candidates; put epochs/eval batch in YAML, or use train/all overrides.' >&2; exit 2
    fi
    tune_args=("${common[@]}" --mode "$mode" --memory-gib "$memory")
    if [[ -n $cache ]]; then tune_args+=(--proposal-cache "$cache"); fi
    if [[ -n $steps ]]; then tune_args+=(--steps "$steps"); fi
    if [[ $stage == fast ]]; then tune_args+=(--run); fi
    "$python_bin" scripts/perception_agents_dual_tune.py "${tune_args[@]}";;
  benchmark)
    "${launch[@]}" scripts/train_perception_agents_dual.py --stage benchmark "${train_args[@]}";;
  train)
    "${launch[@]}" scripts/train_perception_agents_dual.py --stage train "${train_args[@]}";;
  val|test)
    "${launch[@]}" scripts/test_perception_agents_dual.py --split "$stage" "${eval_args[@]}";;
  all)
    if [[ -n $steps ]]; then echo 'Use smoke for a limited train+test run; all requires full training.' >&2; exit 2; fi
    "${launch[@]}" scripts/train_perception_agents_dual.py --stage train --require-full-run "${train_args[@]}"
    "${launch[@]}" scripts/test_perception_agents_dual.py --split all "${eval_args[@]}";;
  smoke)
    smoke_output=${output:-./outputs/perception_agents_dual_smoke_$stamp}
    "${launch[@]}" scripts/train_perception_agents_dual.py --stage train "${train_args[@]}" \
      --output "$smoke_output" --batch-size 1 --gradient-accumulation 1 --gradient-checkpointing \
      --steps 2 --max-val-samples 4
    "${launch[@]}" scripts/test_perception_agents_dual.py --split all "${eval_args[@]}" \
      --output "$smoke_output" --batch-size 1 --max-val-samples 4 --max-test-samples 4;;
esac
