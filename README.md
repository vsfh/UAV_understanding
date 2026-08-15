# CLEAR-UAV experiments

The end-to-end paper runner is documented in [`EXPERIMENTS.md`](EXPERIMENTS.md). Inspect the
current three-seed development plan with:

```bash
bash runs/run_all_experiments.sh --profile development --dry-run
```

This repository turns the current paper draft into a small, offline-first experiment codebase.
The implemented path is Qwen3-VL-8B-Instruct LoRA training and evaluation on the three curated
protocols under `./um7`.

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
- OpenCLIP ViT-L/14 direct/definition zero-shot, linear-probe, and full visual fine-tuning
  baselines;
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
bash runs/00_download_paper_models.sh
```

The one-click command downloads and validates all paper checkpoints under
`./hf_cache`:

- `qwen3-vl`: `Qwen/Qwen3-VL-8B-Instruct`;
- `openclip`: `openai/clip-vit-large-patch14`;
- `geochat`: `MBZUAI/geochat-7B`;
- `uavit-1m`: `ZhanYang-nwpu/GeoChat-UAV`, an official UAVIT-1M-adapted checkpoint;
- `geochat-vision-tower`: `openai/clip-vit-large-patch14-336`, required by both GeoChat models.

`UAVIT-1M` itself is an instruction-tuning dataset, not a model checkpoint. The downloader uses
the official GeoChat-UAV model trained on it for the paper's UAVIT-1M-adapted baseline. Download a
subset or inspect paths without downloading with:

```bash
python scripts/download_models.py qwen3-vl openclip \
  --models-root ./hf_cache
python scripts/download_models.py all --dry-run
```

GeoChat and GeoChat-UAV still require the upstream GeoChat inference code; downloading their
weights and CLIP-336 vision tower does not make them compatible with the Qwen evaluator.

If downloading from China is slow, the launcher exposes two opt-in network modes. First try the
official endpoint without the configured proxy and with Xet high-performance mode:

```bash
BYPASS_PROXY=1 HF_XET_HIGH_PERFORMANCE=1 bash runs/00_download_paper_models.sh
```

If the direct route is unavailable, use the community mirror instead:

```bash
BYPASS_PROXY=1 USE_HF_MIRROR=1 bash runs/00_download_paper_models.sh
```

The mirror mode reads the mirror's model API and uses `aria2c` directly because recent
`huggingface_hub` versions reject mirror responses that omit Hugging Face-specific commit headers.
Both commands keep the same target directories and can be rerun after interruption; switching
from Hub/Xet to aria2 starts separate `.aria2` partial files for unfinished weight shards.

The training and evaluation loaders require a local directory containing `config.json`, set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and call Transformers with
`local_files_only=True`. Passing a Hub model ID therefore fails instead of contacting the network.

## 3. Validate data

```bash
bash runs/01_validate_data.sh ./um7
```

Current data status (checked 2026-07-23): most of `photos_2025-05-30_2025-06-30` has been restored,
but 115 pairs (230 context/crop paths) referenced by the full session-disjoint protocol remain
missing. The full 67-class validator therefore still stops at the first missing file. The frozen
18-class subset in `configs/core18_complete.txt` excludes affected classes and passes strict file,
content-group, and session leakage validation with 16,597/2,384/4,778 train/validation/test pairs:

```bash
python scripts/validate_split.py \
  --data-root ./um7 \
  --protocol session_disjoint \
  --labels-file configs/core18_complete.txt
```

A four-row code-path smoke run is also available:

```bash
bash runs/02_smoke_train.sh
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
  --model-path ./hf_cache/qwen3-vl \
  --adapter-path ./outputs/clear_full_seed42/final \
  --data-root ./um7 \
  --csv ./um7/session_disjoint/test_inputs.csv \
  --private-labels ./um7/session_disjoint/test_labels_private.csv \
  --thresholds ./outputs/clear_full_seed42/val_closed_set.thresholds.json \
  --output ./outputs/clear_full_seed42/test_closed_set.json
```

OpenCLIP validation is:

```bash
python scripts/download_models.py openclip
bash runs/21_eval_clip_val.sh
```

To run every OpenCLIP direct/definition validation baseline over all three protocols, first create
the small dedicated environment and then launch the resumable suite:

```bash
bash runs/setup_openclip_env.sh
CUDA_VISIBLE_DEVICES=0 bash runs/31_run_openclip_only.sh
```

The environment is stored under `~/.conda/envs/uav-openclip`; the setup script installs
dependencies only and never starts an experiment. Set `OPENCLIP_ENV_PREFIX` during setup and
`OPENCLIP_PYTHON` during evaluation if you prefer another location.

For the fair same-backbone comparison on a 24 GB RTX 4090, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/32_run_openclip_full_suite_24g.sh
```

This schedules, on each of the three protocols:

- OpenCLIP zero-shot with direct and definition prompts;
- OpenCLIP linear probe with the entire pretrained model frozen and only a normalized classifier
  trained;
- OpenCLIP full visual fine-tuning with the vision encoder, visual projection, and classifier
  trained while the text encoder remains frozen.

Linear probe and full fine-tuning both run for 20 epochs with seeds 42/43/44. The 24 GB single-card
defaults are a full-tune micro-batch of 8 with gradient accumulation 2, BF16 autocast, gradient
checkpointing, and a 20 GB free-memory preflight gate. Existing completed outputs are skipped.
Lower `FULL_BATCH_SIZE` and raise `FULL_GRADIENT_ACCUMULATION` by the same factor if another process
or driver overhead reduces available memory.

For multiple GPUs with different memory capacities, use the heterogeneous scheduler:

```bash
# Use every GPU reported by nvidia-smi
bash runs/33_run_openclip_multi_gpu.sh

# Or select physical GPU indices explicitly
GPU_IDS=0,1,2 bash runs/33_run_openclip_multi_gpu.sh
```

GPU memory is not pooled into one virtual card. Instead, independent protocol/seed shards are
assigned dynamically to available GPUs. Before every task, current free VRAM is queried again;
larger free-memory cards receive larger image/feature micro-batches while full fine-tuning keeps an
effective batch of 16 through gradient accumulation. CUDA OOM automatically retries with half the
micro-batch. Each GPU runs one process at a time. Failed or interrupted
shards can be resumed with the same command, and the final unified summary is written to
`results/openclip_multi_gpu_e20/suite_summary.json` with combined CSV, LaTeX, and paper tables.
The terminal shows one overall job bar plus one live status row per GPU, including current shard,
elapsed time, and the latest nested feature-cache/training/evaluation `tqdm` update. Complete child
output remains available under `results/openclip_multi_gpu_e20/logs/`.

Qwen3-VL zero-shot direct/definition prompts use the base model without an adapter:

```bash
bash runs/23_eval_qwen_zero_shot_val.sh \
  ./um7 \
  ./hf_cache/qwen3-vl definition \
  ./outputs/qwen_definition_val.jsonl context
```

## Experiment boundaries

`configs/ontology.yaml` contains compact working definitions and graph edges for all 67 visible
classes. They are suitable for pipeline development, but the comments and paper both require them to
be replaced or signed off against the final evidence cards before a main-table run. GeoChat and
GeoChat-UAV snapshots are present locally, but their upstream custom inference stack is not yet
integrated or silently emulated by this Transformers-native code.

The current implementation covers label/set recognition, ranking metrics, and grouped bootstrap
confidence intervals. Classwise calibration plots, proposal-crop evaluation, evidence-deletion
diagnostics, and human explanation-quality ledgers require predictions or annotations not present in
the supplied data and should be added only when those inputs exist.

## One-click paper-table run

Treat the existing per-crop descriptions under
`./um7/description` as the grounded-caption supervision and
run every currently executable three-seed row with:

```bash
bash runs/30_run_paper_tables.sh
```

The launcher uses all three protocols, seeds 42/43/44, Qwen3-VL and OpenCLIP from
`./hf_cache`, and resumable outputs under `outputs/paper_tables`. It also adds a
normalized-likelihood Qwen definition baseline so mAP, hard-negative accuracy, and AURC are
available for that main-table row.

After the model runs finish, five generated LaTeX tables and a cell-level coverage ledger are
written to `results/paper_tables/paper_tables/`. The generated files do not overwrite the
registered templates under `paper/table/`. Crop captions are recorded as
`development_only_non_human_audited`; development CLEAR rows are explicitly marked as using proxy
counterfactual targets. GeoChat inference, four unimplemented PEFT modes, reviewed robustness
slices, proposal crops, evidence-assignment scores, and blinded caption ratings remain `\tbd`
rather than receiving invented values.

Extra suite arguments can be appended. For example, select another GPU without editing code:

```bash
CUDA_VISIBLE_DEVICES=1 bash runs/30_run_paper_tables.sh
```

This command is intentionally documented but is not launched automatically.

The 49,140 MiB one-click defaults are deliberately conservative: ordinary LoRA keeps batch 2 with
gradient accumulation 8, while random/graph/CLEAR multi-loss training automatically uses batch 1
with accumulation 16. Closed-set candidate batch is 2, with 262,144 pixels per image, sequence
length 2,048, and 128 generated tokens. The runner requires at least 40,000 MiB free before starting
and enables PyTorch expandable CUDA segments. Closed-set scoring projects vocabulary logits only at
supervised answer-token positions and never asks Transformers to compute its unused built-in loss.

The terminal shows a total-suite progress bar, Hugging Face training progress, per-image free
generation/OpenCLIP progress, target-construction progress, bootstrap progress, and closed-set
candidate-batch progress with allocated/peak GPU memory. Rerunning the same command after a failure
is safe because `--resume` skips outputs that are complete and restarts incomplete steps.

The safety knobs remain explicit environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 \
CANDIDATE_BATCH_SIZE=2 \
TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION=8 \
MULTI_LOSS_BATCH_SIZE=1 \
MULTI_LOSS_GRADIENT_ACCUMULATION=16 \
MIN_FREE_GPU_MIB=40000 \
bash runs/30_run_paper_tables.sh
```

Increasing `CANDIDATE_BATCH_SIZE` above 4 for pair-view evaluation or increasing multi-loss pair
training above batch 1 is rejected on GPUs with at most 52 GiB.
