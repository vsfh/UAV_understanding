#!/usr/bin/env bash
set -euo pipefail

# Run from the project directory. CUDA_VISIBLE_DEVICES is set by the caller.
# No training, no multi-seed jobs, no modification of existing outputs.
CONFIG=${1:-configs/yaml/paper_small_checks.yaml}

echo '[1/8] Existing predictions: class comparison, errors, group false alarms'
python scripts/paper_small_checks.py analysis --config "$CONFIG"
python scripts/paper_small_checks.py shifts --config "$CONFIG"

echo '[2/8] Paired reference-ROI perturbation (CPU only)'
python scripts/paper_small_checks.py perturb --config "$CONFIG"

echo '[3/8] Prepare independent human ROI review (does not wait for annotation)'
python scripts/paper_roi_review.py prepare --config "$CONFIG"

echo '[4/8] Short Direct / Perception latency and memory profiles'
python scripts/paper_efficiency.py --model direct --config "$CONFIG"
python scripts/paper_efficiency.py --model perception --config "$CONFIG"

echo '[5/8] D-FINE top-1 / top-10 / all-candidate AP50'
python scripts/paper_candidate_eval.py --model dfine --config "$CONFIG"

echo '[6/8] Existing DINOv2 classifier on the top-10 D-FINE proposals'
python scripts/paper_candidate_eval.py --model dfine_dinov2 --config "$CONFIG"

echo '[7/8] Grounding DINO multi-candidates and canonical class mapping'
python scripts/paper_candidate_eval.py --model grounding_dino --config "$CONFIG"

echo '[8/8] Collect the finished results'
python scripts/paper_small_checks.py summary --config "$CONFIG"
