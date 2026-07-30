# CLEAR-UAV experiments

The end-to-end paper runner is documented in [`EXPERIMENTS.md`](EXPERIMENTS.md). Inspect the
current three-seed development plan with:

```bash
bash runs/run_all_experiments.sh --profile development --dry-run
```

This repository turns the current paper draft into a small, offline-first experiment codebase.
The implemented path is Qwen3-VL-8B-Instruct LoRA training and evaluation on the three curated
protocols under `/media/data1/feihong/uav_understanding_data`.

Current scores are development diagnostics, not registered main-table results: the paper's
acquisition-provenance `READY` gate and human-adjudication requirements are still unmet.

## What is implemented

- strict validation for `forward_temporal`, `session_disjoint`, and `unseen_site`;
- context-only, evidence-only, and ordered context+evidence inputs;
- resumable Qwen3.6-35B teacher generation with label-agnostic perception, constrained
  commercial-event rewriting, label-blind entailment/counterfactual checks, and a strict human
  review ledger;
- Qwen3-VL language/projector LoRA with the paper defaults (`r=16`, `alpha=32`, dropout `0.05`);
- canonical-label token weighting, graph-neighbor margin loss, counterfactual unlikelihood, and
  deterministic view dropout;
- free-generation JSON diagnostics;
- normalized candidate log-likelihood, independent thresholds, and log-sum-exp set aggregation;
- OpenCLIP ViT-L/14 direct/definition zero-shot baseline;
- pair macro/micro F1, label mAP, top-1 error AURC, worst-class recall, exact-set accuracy, and
  set metrics.

The code does not silently skip missing data, invent audited captions, or download model files during
an experiment.

Generate the pending-review generic and grounded teacher targets from the local 35B snapshot with:

```bash
CUDA_VISIBLE_DEVICES=1 bash runs/generate_teacher_labels.sh
```

The exact protocol, dry-run command, outputs, and human-review finalization are documented in
[`EXPERIMENTS.md`](EXPERIMENTS.md#generate-the-teacher-descriptions).

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Qwen3-VL-8B LoRA training requires a CUDA GPU with enough memory. Adjust batch size and gradient
accumulation explicitly if needed.

## 2. Download models first

Downloading is deliberately separate from every experiment:

```bash
python scripts/download_models.py qwen3-vl --models-root ./models
python scripts/download_models.py openclip --models-root ./models
python scripts/download_models.py geochat --models-root ./models
```

The training and evaluation loaders require a local directory containing `config.json`, set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and call Transformers with
`local_files_only=True`. Passing a Hub model ID therefore fails instead of contacting the network.

## 3. Validate data

```bash
bash runs/01_validate_data.sh /media/data1/feihong/uav_understanding_data
```

Current data status (checked 2026-07-23): most of `photos_2025-05-30_2025-06-30` has been restored,
but 115 pairs (230 context/crop paths) referenced by the full session-disjoint protocol remain
missing. The full 67-class validator therefore still stops at the first missing file. The frozen
18-class subset in `configs/core18_complete.txt` excludes affected classes and passes strict file,
content-group, and session leakage validation with 16,597/2,384/4,778 train/validation/test pairs:

```bash
python scripts/validate_split.py \
  --data-root /media/data1/feihong/uav_understanding_data \
  --protocol session_disjoint \
  --labels-file configs/core18_complete.txt
```

A four-row code-path smoke run is also available:

```bash
bash runs/02_smoke_train.sh /media/data1/feihong/uav_understanding_data ./models/qwen3-vl
```

## 4. Train

The two-view label baseline can be launched with:

```bash
bash runs/10_train_two_view.sh DATA_ROOT MODEL_PATH OUTPUT_DIR
```

The full method must receive adjudicated supervision. This is intentional: the supplied CSV files
contain labels and paths, but not the paper's audited evidence statements or factor annotations.
Each JSONL line must have this shape:

```json
{
  "record_uid": "rec_...",
  "target": {
    "events": ["floating_garbage"],
    "factors": {"domain": "water", "entity": "garbage", "state": "floating", "relation": "on_water", "context": null},
    "evidence": "Visible debris is floating on the water surface.",
    "uncertain": false
  },
  "counterfactual_target": {
    "events": ["waterside_garbage"],
    "factors": {"relation": "at_water_edge"},
    "evidence": "The debris lies on land at the water edge.",
    "uncertain": false
  },
  "supervision_tier": "human_audited"
}
```

Then run:

```bash
bash runs/11_train_clear_full.sh DATA_ROOT MODEL_PATH OUTPUT_DIR audited_targets.jsonl
```

The script fails if any training record lacks an audited target, if the verified event is missing,
or if the structured schema is invalid.

## 5. Evaluate

Free-generation JSON parsing is a useful diagnostic:

```bash
bash runs/20_eval_qwen_val.sh DATA_ROOT MODEL_PATH ADAPTER_PATH
```

The paper's closed-set protocol is the main scorer. Fit thresholds only on validation data:

```bash
bash runs/22_score_closed_set_val.sh DATA_ROOT MODEL_PATH ADAPTER_PATH
```

The closed-set output keeps every label score. Prespecified set aggregators can therefore be
recomputed without repeating model inference:

```bash
python scripts/rescore_sets.py \
  --scores ./outputs/two_view_seed42/val_closed_set.json \
  --aggregator max \
  --output ./results/two_view_val_closed_set_max.json
```

For test evaluation, pass the generated validation threshold file and private labels explicitly:

```bash
python scripts/evaluate_closed_set.py \
  --model-path ./models/qwen3-vl \
  --adapter-path ./outputs/clear_full_seed42/final \
  --data-root /media/data1/feihong/uav_understanding_data \
  --csv /media/data1/feihong/uav_understanding_data/session_disjoint/test_inputs.csv \
  --private-labels /media/data1/feihong/uav_understanding_data/session_disjoint/test_labels_private.csv \
  --thresholds ./outputs/clear_full_seed42/val_closed_set.thresholds.json \
  --output ./outputs/clear_full_seed42/test_closed_set.json
```

OpenCLIP validation is:

```bash
python scripts/download_models.py openclip --models-root ./models
bash runs/21_eval_clip_val.sh DATA_ROOT ./models/openclip
```

Qwen3-VL zero-shot direct/definition prompts use the base model without an adapter:

```bash
bash runs/23_eval_qwen_zero_shot_val.sh DATA_ROOT ./models/qwen3-vl definition \
  ./outputs/qwen_definition_val.jsonl context
```

## Experiment boundaries

`configs/ontology.yaml` contains compact working definitions and graph edges for all 67 visible
classes. They are suitable for pipeline development, but the comments and paper both require them to
be replaced or signed off against the final evidence cards before a main-table run. The downloader
supports GeoChat, but no local GeoChat snapshot is currently present, and its upstream custom
inference stack is not silently emulated by this Transformers-native code.

The current implementation covers label/set recognition, ranking metrics, and grouped bootstrap
confidence intervals. Classwise calibration plots, proposal-crop evaluation, evidence-deletion
diagnostics, and human explanation-quality ledgers require predictions or annotations not present in
the supplied data and should be added only when those inputs exist.
