#!/usr/bin/env bash
set -e
# Select a free/shared GPU yourself: CUDA_VISIBLE_DEVICES=0 bash runs/qwen_pilot.sh control
arm=${1:-control}
python scripts/train_qwen_pilot.py --arm "$arm"
if [ "$arm" != remaining ]; then
  python scripts/test_qwen_pilot.py --arm "$arm"
fi
