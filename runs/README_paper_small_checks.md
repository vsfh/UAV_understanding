# Paper small checks — no new training

In the project directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/paper_small_checks.sh
```

Order: saved-prediction analysis → reference perturbation → blind review package
→ short efficiency profiles → D-FINE → D-FINE+DINOv2 top-10 → Grounding DINO.
The order is expected-cost based, not a measured runtime guarantee. All loops
show tqdm progress. Existing model weights and original results remain unchanged.
New outputs go to `outputs/paper_small_checks/`; candidate inference resumes
from complete per-image JSONL records. A truncated final JSON line raises an
error rather than silently declaring a complete result. Use a different output
folder in the YAML when changing weights, recipe, or cohort. Checkpoint content
is not hashed by the candidate cache, so do not replace weights under a cached run.

## What each result means

- `diagnostics.json`, `per_class.csv`: all 18 classes, Direct vs final Perception,
  error decomposition, actual validation score range, test false alarms per group.
- `existing_shifts.json`: reuses saved cross-domain Direct and **base** Perception
  predictions only when IDs and targets match. It does not certify matched training
  schedules; do not automatically insert those comparisons into the paper.
- `roi_perturbation.json`: paired synthetic reference shifts/scales, 5/10/20%,
  50 trials per setting. This is not inter-annotator agreement. Never choose the
  best perturbation for the main result.
- `efficiency_*.json`: same 96 validation records, warm-up, batch=1 and 2,
  synchronized end-to-end timings, generation/replay timings, peak allocated and
  reserved memory, processed pixels. Run on an otherwise idle GPU for publication.
- `*_candidates.jsonl`, `*_metrics.json`: one-to-one non-interpolated reference AP,
  not COCO AP. No NMS or test-selected filtering. Negatives and duplicate boxes
  contribute false positives. Stable score ties use saved source/proposal order.
- D-FINE `all` means every returned postprocessor proposal (not undocumented raw
  decoder-layer boxes). Grounding DINO `all` means every query above the original
  0.05 score floor across the original prompt chunks, with no cross-chunk NMS.
- Grounding DINO maps a query to the canonical class whose label+definition token
  span has the highest sigmoid score, within its chunk. Box ranking retains the
  original query confidence. This is an explicit added mapping, not fine-tuning.
- D-FINE has no event classifier. Only **D-FINE+DINOv2** reports class-dependent
  AP at K=1/10. We deliberately do not classify all hundreds of D-FINE candidates,
  since that would no longer be a small experiment.
- `old_top1_max_box_coordinate_difference` helps detect changed inference. A
  non-negligible discrepancy requires inspection before comparing candidate budgets.

## Independent ROI review

The script samples 20 test positives per class, five per within-class ROI-area
quartile: 360 frames. Sampling uses no model errors. The package is at
`outputs/paper_small_checks/roi_review/`.

Give reviewer A only `A_draw.html`, `A_utility.html`, and `images/`; give B the
corresponding B files plus `images/`. Open the HTML files in a browser locally.
Do not share `private_key.json`. Keep reviewers independent.

1. Complete the `draw` file first, without seeing original or model ROIs.
2. Then complete `utility`: judge whether each anonymous candidate preserves
   enough evidence for the indicated event. Use yes/no/uncertain, not closeness
   to an imagined reference boundary. No ROI is a valid observed model failure.
3. Export the four JSON files. Local browser storage is only a convenience, not
   a substitute for saving the exports. Put them in the project's review folder.
4. Score the completed exports:

```bash
python scripts/paper_roi_review.py score \
  --draw outputs/paper_small_checks/roi_review/A_draw.json outputs/paper_small_checks/roi_review/B_draw.json \
  --utility outputs/paper_small_checks/roi_review/A_utility.json outputs/paper_small_checks/roi_review/B_utility.json
```

`agreement_results.json` includes completion counts and unweighted stratified
agreement, a population-weighted mean, and blinded utility rating counts.
Missing/uncertain boxes are explicitly excluded with completion counts, not
treated as agreement. `adjudication_needed.json` lists low-IoU pairs for an
independent third review. Do not report automatic adjudication: none is done.
This analysis must not be used to tune models or rewrite the test references.

## Matched control is separate

This small-check script does not train. The already installed single-seed
matched experiment is still:

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/matched_coordinate_control.sh
```

Its result is required before claiming an isolated benefit from the output
design. Multi-seed training is intentionally not part of this delivery.

## Smoke checks

```bash
python scripts/test_paper_small_checks.py
python scripts/paper_candidate_eval.py --model dfine --limit 2
python scripts/paper_candidate_eval.py --model dfine_dinov2 --limit 2
python scripts/paper_candidate_eval.py --model grounding_dino --limit 2
```

Limited candidate runs use distinct `_smoke2` files and cannot satisfy a full-run
cache. Smoke commands load existing models but do not train them.
