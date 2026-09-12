#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source runs/perception_runtime.sh
perception_resolve_python
mode="${1:-all}"
if (($#)); then shift; fi
config="configs/yaml/perception_actions.yaml"
if (($#)) && [[ "$1" != --* ]]; then config="$1"; shift; fi
case "$mode" in
  sft) "$python_bin" scripts/train_perception_actions.py --config "$config" --stage sft "$@" ;;
  grpo) "$python_bin" scripts/train_perception_actions.py --config "$config" --stage grpo "$@" ;;
  train|all)
    if (($#)); then echo "For smoke flags use sft/grpo separately; train/all accept only a YAML." >&2; exit 2; fi
    "$python_bin" scripts/train_perception_actions.py --config "$config" --stage sft
    if [[ "$mode" == all ]]; then "$python_bin" scripts/test_perception_actions.py --config "$config" --stage sft --split all; fi
    "$python_bin" scripts/train_perception_actions.py --config "$config" --stage grpo
    if [[ "$mode" == all ]]; then "$python_bin" scripts/test_perception_actions.py --config "$config" --stage grpo --split all; fi
    ;;
  val|test) "$python_bin" scripts/test_perception_actions.py --config "$config" --split "$mode" "$@" ;;
  help|-h|--help)
    echo "Usage: bash runs/perception_actions.sh [sft|grpo|train|val|test|all|help] [config.yaml] [stage-specific Python flags]"
    echo "sft/grpo: --max-train-samples 4 --max-val-samples 4 --steps 1 --epochs 1 --output NEW_ROOT"
    echo "val/test: --stage sft|grpo --output ROOT --max-val-samples 4 --max-test-samples 4"
    echo "GRPO YAML updates_per_rollout=2 reuses each frozen rollout buffer; --steps counts optimizer updates, including reuses."
    echo "train runs SFT then GRPO, using their internal validation only. all evaluates both stages on val and test. No optimizer resume; existing stage outputs are rejected."
    ;;
  *) echo "Unknown mode: $mode (use help)" >&2; exit 2 ;;
esac
