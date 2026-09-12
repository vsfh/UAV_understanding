#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source runs/perception_runtime.sh
perception_resolve_python
mode="${1:-all}"
config="${2:-configs/yaml/perception_reward.yaml}"
if (($#)); then shift; fi
if (($#)) && [[ "$1" != --* ]]; then config="$1"; shift; else config='configs/yaml/perception_reward.yaml'; fi
case "$mode" in
  prepare|rm|sft|grpo|train)
    "$python_bin" scripts/train_perception_reward.py --config "$config" --stage "$mode" "$@" ;;
  prepare-test)
    "$python_bin" scripts/train_perception_reward.py --config "$config" --stage prepare --prepare-split test "$@" ;;
  val|test)
    "$python_bin" scripts/test_perception_reward.py --config "$config" --split "$mode" "$@" ;;
  quality)
    "$python_bin" scripts/test_perception_reward.py --config "$config" --mode quality --split val "$@" ;;
  all)
    if (($#)); then echo 'Use separate stages for smoke flags; all takes only a YAML.' >&2; exit 2; fi
    "$python_bin" scripts/train_perception_reward.py --config "$config" --stage train
    "$python_bin" scripts/test_perception_reward.py --config "$config" --mode quality --split val
    "$python_bin" scripts/test_perception_reward.py --config "$config" --stage sft --split all
    "$python_bin" scripts/test_perception_reward.py --config "$config" --stage grpo --split all ;;
  help|-h|--help)
    echo 'Usage: bash runs/perception_reward.sh [prepare|prepare-test|rm|sft|grpo|train|quality|val|test|all] [config.yaml] [stage flags]'
    echo 'train: prepare train/val features -> fit RM -> action SFT -> RM-assisted action GRPO.'
    echo 'For matched controls: grpo --checkpoint outputs/perception_actions/PROTOCOL/seed43/sft/best'
    echo 'prepare: --max-train-samples N --max-val-samples N --output NEW_ROOT'
    echo 'rm: --rm-epochs N --output ROOT; sft/grpo: --epochs N --steps N --output ROOT'
    echo 'quality: held-out validation ROI-quality metrics; val/test: action policy metrics.' ;;
  *) echo "Unknown mode: $mode (use help)" >&2; exit 2 ;;
esac
