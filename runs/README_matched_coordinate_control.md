# Matched coordinate-output control

Run from the repository root. The launcher trains and then evaluates automatically.

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/matched_coordinate_control.sh
```

Or run the two independent arms concurrently on different GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/matched_coordinate_control.sh token
CUDA_VISIBLE_DEVICES=1 bash runs/matched_coordinate_control.sh continuous
```

Do not run `all` together with the separate-arm commands. Existing arm output directories are not overwritten.

## What is matched

- Original Qwen3-VL base checkpoint, random seed, full multimodal backbone, vision freezing, and LoRA targets/rank/alpha/dropout.
- One tokenizer, including `<vis>` and all 1,000 location tokens, in both arms. All added embedding/output rows have identical initial values and training eligibility.
- The ROI MLP is instantiated identically in both arms. It is unused (no gradient/update) in the token arm. This avoids arm-dependent RNG consumption during initialization.
- Exactly the same prompt, training records, class-balanced draws, image resolutions, batches, gradient accumulation, optimizer, learning rates, warmup, clipping, and three epochs.
- Prompt processing is shared and answers use common padded sequence lengths, preserving shared-prefix positions and dropout tensor shapes across the two formats.
- Both use the fixed epoch-3 checkpoint. No continued stage, early stopping, or selection by incomparable validation losses.
- Evaluation uses the same val/test records, event definitions, scoring code, greedy constrained category decoding, and validation-only alert calibration.

Initialization fingerprints, tokenizer/recipe/data fingerprints, per-epoch sampled-record order, and optimizer-step/LR traces are checked before a joint comparison is accepted. The shared base is reloaded with the same seed in each run, without loading any previously adapted Perception or Direct-Qwen checkpoint. Do not change base checkpoint files between the two runs.

## The intended difference

| Arm | Positive answer | Coordinate supervision | Box inference |
| --- | --- | --- | --- |
| token | category + `<vis>` + four `<loc_i>` tokens | Coordinate/EOS cross-entropy | Four autoregressively generated coordinates |
| continuous | category + `<vis>` | L1 + GIoU | MLP on the final assistant `<vis>` state |

Both negative answers are `no_event` and receive no coordinate supervision. Location tokens represent the existing 1,000-bin 0--1000 coordinate grid. Quantized edges are minimally separated if rounding collapses a box.

Semantic cross-entropy is averaged over the same category + `<vis>` targets (or negative answer + EOS). Coordinate tokens are excluded from this average so that the longer token answer cannot dilute category supervision. The token arm separately supervises its four coordinate tokens and EOS. The continuous arm's positive EOS is grammar-forced. Both native coordinate losses have unit weight. No hyperparameter search is built into this comparison.

This is a **matched output-parameterization control**, including each representation's necessary coordinate loss. It is not a same-trained-checkpoint head swap, nor evidence that only the presence of an MLP causes every difference. Token decoding needs one generation call. Continuous decoding uses the existing generation-and-replay procedure, so sequence length and inference work legitimately differ.

The controlled continuous run is a new from-base run, not the continued checkpoint currently in the paper. Compare the two new arms with one another.

## Outputs

```text
outputs/matched_coordinate_control/session_disjoint/seed43/
  token/
    config.yaml
    matched_manifest.json
    history.json
    epoch_1/ epoch_2/ epoch_3/
    calibration.json
    val_results.json
    test_results.json
    evaluation_complete.json
  continuous/
    (same layout)
  comparison.json
```

`comparison.json` is generated after both evaluations complete and matching checks pass. It contains both metric dictionaries. The result files retain all per-record predictions for paired analysis. If needed, rebuild the summary without inference:

```bash
python scripts/test_matched_coordinate_control.py --compare-only
```

To reevaluate one existing epoch-3 checkpoint (validation followed by test):

```bash
python scripts/test_matched_coordinate_control.py --arm token
python scripts/test_matched_coordinate_control.py --arm continuous
```

Evaluation overwrites that arm's evaluation files, not its trained checkpoint. Training does not overwrite existing arm directories. To start a different run, copy the YAML and use a new output location or seed, then pass that YAML as the launcher's second argument.
