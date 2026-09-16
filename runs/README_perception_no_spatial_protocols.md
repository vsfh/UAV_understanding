# Perception w/o Spatial: unseen-site and forward-temporal

Run from `/media/data2/feihong/UAV_understanding` with the existing Python environment.
These commands implement the current session-disjoint baseline's full recipe:
**original Qwen -> 3-epoch Perception base -> 3-epoch Perception continuation -> full validation calibration -> full test**.
Both stages have no Spatial module. Each new protocol starts from the original
`hf_cache/qwen3-vl`; no session-disjoint trained weights are reused.

```bash
# One command per protocol; choose an available GPU.
bash runs/perception_no_spatial_unseen_site.sh --gpu 0
bash runs/perception_no_spatial_forward_temporal.sh --gpu 1
```

The two protocols can run in separate terminals on different GPUs. The scripts
do not stop other jobs or choose a free GPU automatically. Set `PYTHON` to an
interpreter if needed; the existing `runs/perception_runtime.sh` is reused.

## Separate stages

The following options work with either protocol's shell entry point:

```bash
# Read-only CPU checks; no model loading, training or experiment output creation.
bash runs/perception_no_spatial_unseen_site.sh --mode check

# Print the exact subprocess commands without running them.
bash runs/perception_no_spatial_unseen_site.sh --dry-run

# Both training stages, or a single stage.
bash runs/perception_no_spatial_unseen_site.sh --mode train --gpu 0
bash runs/perception_no_spatial_unseen_site.sh --mode base --gpu 0
bash runs/perception_no_spatial_unseen_site.sh --mode continue --gpu 0

# Validation calibration, then full test (safe standalone test entry).
bash runs/perception_no_spatial_unseen_site.sh --mode test --gpu 0

# Direct Python entry points.
python scripts/train_perception_no_spatial.py --protocol forward_temporal --stage base
python scripts/train_perception_no_spatial.py --protocol forward_temporal --stage continue
python scripts/test_perception_no_spatial.py --protocol forward_temporal --split all
```

The Python evaluator also supports `--split val` and `--split test`. Test-only
requires an existing validation calibration for the same protocol and checkpoint.
The one-click `--mode test` recalibrates on validation before testing.

## Exact recipe

The four protocol YAMLs copy the actual saved session-disjoint base and
continuation configurations. Two `perception_no_spatial_session_*.yaml` files
preserve those reference configurations for comparison. Only `protocol` changes;
the historical `matched_spatial_config` audit pointer is removed because this
experiment has no Spatial branch. Templated output and initial-checkpoint paths
resolve to the new protocol automatically.

Unchanged: Qwen3-VL, LoRA rank 16 / alpha 32 / dropout 0.05; seed 43; batch 2;
gradient accumulation 4; adapter/head learning rates 0.0002; weight decay 0.01;
cosine schedule and 10% warmup; clipping 1.0; class-balance exponent 0.5;
positive-record epoch sampling budget; four loader workers; input pixel range
65,536-995,328; all three loss weights 1; evaluation batch 2; validation N-FPR
constraint 0.1. Each stage runs three epochs and selects minimum validation loss.
Continuation starts from the base's selected checkpoint with a fresh optimizer
and scheduler, exactly as the session-disjoint continuation did. It is not a
single uninterrupted six-epoch run. Epoch length and update counts naturally
depend on the protocol's training-set size.

Existing implementations are reused without editing:

- `train_perception_qwen.train`: model, collator, losses, optimizer and training loop.
- `train_perception_continue.build_model`: load trainable base adapter and ROI head.
- `test_perception_continue.evaluate`: generation, validation threshold and metrics.

## Data and outputs

The existing `um7/unseen_site/` and `um7/forward_temporal/` manifests are read
directly. No split generation or dataset selection is rerun. All 18 ontology
classes remain enabled, including classes absent from forward-temporal training.
Training inspects public split membership only; private test labels are read by
the existing evaluator when testing.

The original no-event pool and splitting algorithm are unchanged. Its grouping
is seeded by protocol; it does not have site/time metadata, so negative-image
false-positive rates do not establish strict unseen-site or temporal rejection.

For each `<protocol>` in `unseen_site` and `forward_temporal`:

- `outputs/perception_qwen/<protocol>/seed43/best/`: protocol-specific base.
- `outputs/perception_continue/<protocol>/seed43/best/`: final w/o-Spatial model.
- `outputs/perception_continue/<protocol>/seed43/val_results.json`
- `outputs/perception_continue/<protocol>/seed43/calibration.json`
- `outputs/perception_continue/<protocol>/seed43/test_results.json`
- Each training stage saves config, history, source/manifest provenance and training-data state.

Repeating the command reuses completed training stages and reruns evaluation.
Partial training is preserved and reported; the existing trainer does not
restore optimizer state. No training or GPU inference was launched during
deployment; CPU recipe/data checks and mocked entry-point tests are sufficient
to verify wiring without spending the full experiment budget.
