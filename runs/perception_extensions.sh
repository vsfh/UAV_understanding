#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
usage() {
  cat <<'USAGE'
Usage: bash runs/perception_extensions.sh [spatial|zoom|actions|reward|all] [mode] [config.yaml] [options]

Common modes: train | val | test | all
Zoom stages: prepare | prepare-test
Actions stages: sft | grpo
Reward stages: prepare | rm | quality | sft | grpo | prepare-test
Examples:
  bash runs/perception_extensions.sh spatial all
  bash runs/perception_extensions.sh zoom prepare --max-train-samples 2 --max-val-samples 2 --output ./outputs/zoom_smoke
  bash runs/perception_extensions.sh actions sft --steps 1 --output ./outputs/actions_smoke
  bash runs/perception_extensions.sh all all

The all-variant mode runs the four separate default YAMLs sequentially.
It accepts only train/val/test/all, with no shared config or extra options.
Use an individual variant for smoke limits, custom outputs, or shared SFT checkpoints.
PYTHON selects the interpreter; CUDA_VISIBLE_DEVICES selects the GPU.
See EXPERIMENTS_PERCEPTION_EXTENSIONS.md for provenance and fair comparisons.
USAGE
}
variant=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in help|-h|--help) usage; exit 0;; spatial|zoom|actions|reward|all) ;; *) usage; exit 2;; esac
mode=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $mode == help || $mode == -h || $mode == --help ]]; then usage; exit 0; fi
source runs/perception_runtime.sh
perception_resolve_python
if [[ $variant != all ]]; then
  exec bash "runs/perception_${variant}.sh" "$mode" "$@"
fi
case "$mode" in train|val|test|all) ;; *) echo "all-variant mode supports train, val, test, or all" >&2; exit 2;; esac
if [[ $# -ne 0 ]]; then
  echo "all-variant mode takes no shared config/options; run each variant with its own output and limits." >&2
  exit 2
fi
for current in spatial zoom actions reward; do
  echo "Running perception extension: $current / $mode"
  if [[ $current == zoom && $mode == train ]]; then
    bash runs/perception_zoom.sh prepare
  elif [[ $current == zoom && $mode == test ]]; then
    bash runs/perception_zoom.sh prepare-test
  fi
  bash "runs/perception_${current}.sh" "$mode"
done
