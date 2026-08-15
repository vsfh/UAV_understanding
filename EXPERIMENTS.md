# Paper experiment suite

`runs/run_all_experiments.sh` is the single entry point for the executable paper
experiments. It validates every selected split, trains the requested seed matrix, evaluates
free generation and closed-set likelihoods, fits validation thresholds, compares paired runs,
rescales cached set scores with max pooling, and produces JSON, CSV, and LaTeX summaries.

The runner is fail-closed and resumable. It never downloads a model during an experiment and
development mode never opens private test labels.

The legacy `outputs/*_seed42` adapters used suffix-only LoRA matching, which also selected visual
block MLPs. They do not satisfy the paper's frozen-vision-tower configuration. The suite uses
full module-path matching and writes new checkpoints under `outputs/paper_suite`; do not mix the
legacy and corrected runs in one result table.

## Generate the teacher descriptions

The paper's two-stage caption construction is implemented with the local
`Qwen/Qwen3.6-35B-A3B-FP8` snapshot under `./hf_cache`. The full one-command
generation run is:

```bash
CUDA_VISIBLE_DEVICES=1 bash runs/generate_teacher_labels.sh
```

It generates the union of the capped 18-class `session_disjoint` training records selected by
seeds 42, 43, and 44, so one caption ledger covers the complete registered three-seed development
matrix without duplicate teacher calls. Each record uses three deterministic teacher calls:

1. ordered context+crop, with no event label, to produce the generic visible description;
2. the same images plus the verified event definition, factors, and deterministic confusion
   neighbor to produce the grounded statement and counterfactual hypothesis;
3. a separate label-blind prompt to verify image--text support for both statements.

The validated v5 smoke run took 179 seconds for one record on one RTX 4090 after weight/kernel
cache warm-up, so the 5,270-record serial run is a multi-day job (roughly eleven days if that rate
persists). Start it as a persistent background job and inspect progress with:

```bash
bash runs/start_teacher_labels_tmux.sh clear_teacher_v5 1
bash runs/teacher_labels_status.sh clear_teacher_v5
```

The launcher refuses to create a duplicate tmux session. The foreground and background entries
write the same record-level cache and can resume each other.

The run is offline, fail-closed, and resumable at every stage of every record. Rerun the identical
command after an interruption; a configuration fingerprint prevents an incompatible run from
reusing the cache. The one-click entry pins the already cached
`kernels-community/finegrained-fp8` version-1 snapshot through `LOCAL_KERNELS`; this avoids a
network trust lookup during the first FP8 forward pass and records the local kernel path and
metadata hash in the generation manifest. Inspect selection and paths without loading the 35B
model:

```bash
bash runs/generate_teacher_labels.sh \
  ./um7 \
  ./hf_cache/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989 \
  /tmp/clear_teacher_dry \
  --max-samples 2 \
  --dry-run
```

For an actual two-record model smoke test, omit `--dry-run` and use a fresh output directory.
The default full output directory is
`data/targets/qwen36_35b_session_seeds42_43_44_v5`, containing:

- `generic_targets.pending_review.jsonl`: label plus first-pass generic description;
- `grounded_targets.pending_review.jsonl`: constrained commercial-event training target;
- `automatic_audit.jsonl`: schema, positive-entailment, and counterfactual checks;
- `teacher_records.jsonl`: raw and parsed outputs for all three passes;
- `human_review.tsv`: immutable paths and blank controlled review fields;
- `generation_manifest.json`: hashes and frozen generation configuration.

Automatic passage never changes the supervision tier to `human_audited`. After reviewers fill
every decision in `human_review.tsv` with `accept`, `correct`, or `reject`, validate and finalize
the ledger with:

```bash
bash runs/finalize_teacher_labels.sh
```

Finalization rejects incomplete rows, changed immutable fields, missing corrections, rejected
targets still present in the frozen training manifest, and missing reviewer identities. Successful
outputs can be passed to the official suite as
`generic_targets.human_audited.jsonl` and `grounded_targets.human_audited.jsonl`.

## What to run next

The immediate reproducible sequence is:

1. Run the three-seed development matrix on `session_disjoint`.
2. Review the generated comparison intervals for context, crop, and paired inputs.
3. Replace crop-teacher positives and proxy counterfactuals with human-audited targets.
4. Complete the provenance gate and freeze the official manifests.
5. Run official evaluation once across the three registered protocols.
6. Add the still-manual assets listed under "Not automatable from current data".

Inspect the complete plan without using a GPU:

```bash
bash runs/run_all_experiments.sh --profile development --dry-run
```

Run a small end-to-end code-path check:

```bash
bash runs/run_all_experiments.sh \
  --profile smoke \
  --seeds 42 \
  --skip-zero-shot
```

Run the current three-seed development matrix:

```bash
bash runs/run_all_experiments.sh \
  --profile development \
  --protocols session_disjoint \
  --seeds 42 43 44 \
  --resume
```

For a shorter first pass, select only the corrected core view baselines:

```bash
bash runs/run_all_experiments.sh \
  --profile development \
  --experiments label_context label_evidence label_pair \
  --seeds 42 43 44 \
  --resume
```

Omit `--experiments` to run the full matrix.

Run the new per-crop teacher-caption baseline alone with:

```bash
bash runs/run_all_experiments.sh \
  --profile development \
  --protocols session_disjoint \
  --experiments grounded_caption \
  --cropped-captions-root description \
  --seeds 42 43 44 \
  --skip-zero-shot \
  --resume
```

On a machine without a visible GPU, prepare and validate all three seed target files first:

```bash
bash runs/run_all_experiments.sh \
  --profile development \
  --protocols session_disjoint \
  --experiments grounded_caption \
  --cropped-captions-root description \
  --seeds 42 43 44 \
  --skip-zero-shot \
  --prepare-only \
  --resume
```

For each seed, the suite maps
`description/<original-crop-relative-path>.json` into a grounded-caption target JSONL under
`outputs/paper_suite/<protocol>/cropped_caption_targets/`. The student receives the context
view, matching the registered grounded-caption row; the crop is the source of the teacher
target. These generated targets are explicitly non-human-audited and therefore belong only in
development results until review is complete.

To run every currently executable paper-table row across all three protocols and three seeds,
then export the five registered table schemas, use:

```bash
bash runs/30_run_paper_tables.sh
```

This launcher adopts the explicit development assumption that the existing crop captions are the
grounded-caption targets. It writes generated tables to
`results/paper_tables/paper_tables/` and a `table_manifest.json` explaining every still-unfilled
cell. It does not overwrite `paper/table/*.tex` and does not relabel crop captions as human-audited.

For the 49,140 MiB card, ordinary LoRA keeps train batch 2 / accumulation 8, while multi-loss
random/graph/CLEAR runs use batch 1 / accumulation 16. Closed-set candidate batch is 2. The launcher
refuses to start below 40,000 MiB free memory, enables expandable CUDA segments, omits unused
model-internal loss computation, and projects logits only for answer tokens. Overall steps and
long-running inner loops expose `tqdm` progress bars; closed-set progress also reports allocated and
peak GPU memory. The same command resumes the failed step safely.

Development random-negative and graph-neighbor rows reuse the same crop-caption positives as the
grounded baseline. CLEAR rows merge those positives with a separate ontology-derived proxy
counterfactual file for each protocol and seed; only those CLEAR result names retain the `proxy_`
prefix. They must not be presented as the full audited CLEAR method or as official test results.

## OpenCLIP same-backbone comparison

Create the minimal OpenCLIP environment once, then run the complete 24 GB comparison:

```bash
bash runs/setup_openclip_env.sh
CUDA_VISIBLE_DEVICES=0 bash runs/34_run_openclip_full_suite_20g.sh
```

The suite evaluates both zero-shot prompts and trains linear-probe/full-visual-fine-tuning runs
with seeds 42, 43, and 44 on all three protocols. All variants use the same local
`./hf_cache/openclip` checkpoint, `context` input, Core-18 labels, validation metrics, and grouped
bootstrap analysis. Full fine-tuning updates the complete vision encoder, visual projection, and
classification head; the unused text encoder stays frozen. Results and compatible paper-table
exports are written below `results/openclip_full_suite_e20_20g/`.

For heterogeneous multi-GPU execution, use:

```bash
GPU_IDS=0,1,2 bash runs/33_run_openclip_multi_gpu.sh
```

This does not pretend that VRAM is additive. It distributes independent protocol/seed shards,
uses the same fixed 20 GB-safe micro-batches on every GPU with full-fine-tuning effective batch 16,
keeps one job per GPU, retries CUDA OOM with a smaller micro-batch, and merges all
completed shard plans into `results/openclip_multi_gpu_e20_20g/suite_summary.json`.

## Official run

Official mode deliberately refuses to start unless all of the following are supplied:

- a provenance JSON whose top-level `status`, `scientific_gate`, `gate_status`, or
  `release_gate.status` is `READY`;
- a structured target JSONL covering every selected training record, with
  `supervision_tier: human_audited`;
- a generic-caption target JSONL covering every selected training record;
- at least three distinct seeds;
- `--acknowledge-test`, confirming that private test labels may be opened.

After the manifests and annotations are frozen:

```bash
bash runs/run_all_experiments.sh \
  --profile official \
  --provenance-ready-file data/provenance/gate.json \
  --audited-targets data/targets/core18_human_audited.jsonl \
  --generic-targets data/targets/core18_generic_captions.jsonl \
  --protocols forward_temporal session_disjoint unseen_site \
  --seeds 42 43 44 \
  --acknowledge-test \
  --resume
```

`--resume` skips a step only when all of that step's declared outputs exist; an incomplete
training step resumes from its highest numbered checkpoint. The live plan and step status are
stored in `results/paper_suite/suite_plan.json`; logs are under
`results/paper_suite/logs/`.

The final outputs are:

- `suite_summary.json`: raw per-seed metrics, mean/std summaries, paired comparisons, and
  training metadata;
- `suite_summary.csv`: flat metric table;
- `suite_results.tex`: automatically generated LaTeX result table.

Use a different GPU selection with `--cuda-devices`, and change output locations with
`--output-root` and `--results-root`.

## Experiment matrix

Development mode runs:

- Qwen3-VL direct and definition zero-shot prompts;
- OpenCLIP direct and definition prompts;
- label-only LoRA with context, crop, and ordered context+crop;
- LLM-only versus projector+LLM LoRA and the label-token-weighting ablation;
- crop-teacher grounded-caption positives for grounded-caption, random-negative, and
  graph-neighbor rows, plus explicitly marked proxy-counterfactual CLEAR diagnostics;
- free-generation and closed-set validation;
- log-sum-exp and max set aggregation;
- assigned-evidence versus size-matched non-assigned deletion from cached edge scores;
- same-adapter context-only/evidence-only interventions and single-view-insufficiency rates;
- grouped bootstrap analysis and prespecified paired comparisons.

Official mode replaces every proxy row with generic or human-audited supervision and evaluates
validation-selected thresholds on the private test split for all selected protocols.

## Not automatable from current data

The runner records these limitations in every suite plan:

- GeoChat needs its upstream custom inference stack; the repository does not emulate it with
  an incompatible Transformers path.
- Linear probe, projector-only, QLoRA, and full fine-tuning efficiency rows require additional
  architecture-specific implementations and, for the latter modes, substantially different
  hardware budgets. LLM-only and projector+LLM LoRA are automated; their runs record trainable
  parameters, runtime, and peak GPU memory in `run_metadata.json`.
- Caption-quality scoring needs blinded human-rating ledgers.
- Proposal-crop, spatial-jitter, irrelevant-crop, small-evidence, adverse-capture, and
  complex-background results need the corresponding proposal images or reviewed slice
  annotations. Assigned/non-assigned deletion from the existing context graph is automated.
- Site/flight generalization cannot be claimed until acquisition provenance is verified.

These are input or scientific-design blockers, not values that should be inferred from the
available validation predictions.

The paper-only, table-by-table execution order and the complete missing-experiment inventory are
in `EXPERIMENT_PLAN.md`. Its companion `configs/paper_experiment_paths.json` keeps all machine
roots blank and records only assumed relative paths.
