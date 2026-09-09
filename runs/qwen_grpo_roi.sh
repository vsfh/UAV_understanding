#!/usr/bin/env bash
set -e
# Run from the project root. CUDA_VISIBLE_DEVICES is supplied by the caller.
mode=all
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then mode="$1"; shift; fi
case "$mode" in
  all)
    python scripts/train_qwen_grpo_roi.py --config configs/yaml/qwen_grpo_roi.yaml "$@"
    # Follow the actual training output, including custom output/resume options.
    config=configs/yaml/qwen_grpo_roi.yaml
    output=""
    resume=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --config) config="$2"; shift ;;
        --output) output="$2"; shift ;;
        --resume) resume="$2"; shift ;;
        --config=*) config="${1#*=}" ;;
        --output=*) output="${1#*=}" ;;
        --resume=*) resume="${1#*=}" ;;
      esac
      shift
    done
    if [ -n "$resume" ]; then config="$resume/config.yaml"; fi
    if [ -n "$output" ]; then config="$output/config.yaml"; fi
    python scripts/test_qwen_grpo_roi.py --config "$config" --split all
    ;;
  train) python scripts/train_qwen_grpo_roi.py --config configs/yaml/qwen_grpo_roi.yaml "$@" ;;
  val) python scripts/test_qwen_grpo_roi.py --config configs/yaml/qwen_grpo_roi.yaml --split val "$@" ;;
  test) python scripts/test_qwen_grpo_roi.py --config configs/yaml/qwen_grpo_roi.yaml --split test "$@" ;;
  *) echo "Usage: bash runs/qwen_grpo_roi.sh [all|train|val|test] [arguments]"; exit 2 ;;
esac
