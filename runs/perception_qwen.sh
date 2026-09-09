#!/usr/bin/env bash
set -e
# Run from the repository root; select your GPU with CUDA_VISIBLE_DEVICES.
config=${2:-configs/yaml/perception_qwen.yaml}
case ${1:-all} in
  train) python scripts/train_perception_qwen.py --config "$config" ;;
  val)   python scripts/test_perception_qwen.py --config "$config" --split val ;;
  test)  python scripts/test_perception_qwen.py --config "$config" --split test ;;
  all)
    python scripts/train_perception_qwen.py --config "$config"
    python scripts/test_perception_qwen.py --config "$config" --split all
    ;;
  *) echo "Usage: bash runs/perception_qwen.sh [train|val|test|all] [config.yaml]"; exit 2 ;;
esac
