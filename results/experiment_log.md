# CLEAR-UAV experiment log

**Status: development-only.** These measurements do not satisfy the manuscript's `READY`
acquisition-provenance gate and must not be copied into the registered submission main table.
They validate the implementation and quantify the current 18-class complete subset only.

All reported values use the `session_disjoint` protocol and the 18-class complete subset in
`configs/core18_complete.txt`. The excluded supported classes (`cargo_ship`,
`project_construction`, `rural_house_construction`, and `waterside_garbage`) still have missing
2025-06 image pairs and are not silently filtered row by row.

## Environment

- Date: 2026-07-23
- GPUs: 2 x NVIDIA GeForce RTX 4090, 48 GB each
- Base model: local `models/qwen3-vl` (Qwen3-VL-8B-Instruct)
- PyTorch 2.6.0+cu124, Transformers 5.8.0, PEFT 0.19.1
- Visual budget: at most 262,144 pixels per image
- Seed: 42

## Data gate

The complete 18-class subset passes strict file and leakage validation:

| Split | Pairs |
|---|---:|
| Train | 16,597 |
| Validation | 2,384 |
| Private test | 4,778 |

Training caps every class at 250 deterministically selected pairs, leaving 3,569 training pairs.
Validation and test are not capped or duplicated.

## Smoke and throughput checks

| Run | Samples | Batch | Steps | Result |
|---|---:|---:|---:|---|
| LoRA smoke | 4 | 1 | 4 | loss 3.715, 6.14 s, passed |
| LoRA batch check | 32 | 2 | 16 | step-10 loss 1.856, completed without OOM |

## Zero-shot validation

| Method | Prompt | View | Macro-F1 (95% CI) | Micro-F1 (95% CI) | mAP | Hard-neg. | AURC ↓ | Exact acc. | Worst recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCLIP ViT-L/14 | direct | context | 0.0886 [0.0727, 0.1029] | 0.2592 [0.2146, 0.3059] | **0.1353** | 0.4103 | 0.6948 | 0.2592 | 0.0000 |
| OpenCLIP ViT-L/14 | definition | context | **0.1130 [0.0974, 0.1267]** | **0.3788 [0.3355, 0.4210]** | 0.1337 | **0.5248** | **0.5650** | **0.3788** | 0.0000 |
| Qwen3-VL-8B | direct | context | 0.0817 [0.0652, 0.0979] | 0.2818 [0.2356, 0.3294] | -- | -- | -- | 0.1963 | 0.0000 |
| Qwen3-VL-8B | definition | context | 0.0814 [0.0673, 0.0946] | 0.2808 [0.2392, 0.3243] | -- | -- | -- | 0.1854 | 0.0000 |

The definition prompt improves macro-F1 by 0.0244, micro-F1 by 0.1195, micro AP by 0.0688, and
hard-negative pairwise accuracy by 0.1145, but label-macro mAP decreases slightly by 0.0016. It
improves the final class choice without a uniform per-class ranking gain. Absolute macro-F1 remains
low because prediction mass collapses
onto a few visually broad classes. Under the better definition prompt, eight classes have zero
recall; the largest confusion is 337
`crop_lodging -> grain_drying_road` errors. This supports the paper's claim that generic aerial
vision-language alignment is insufficient for operational event boundaries.

Confidence intervals use 10,000 bootstrap resamples over the 329 validation session groups.
AURC is the area under cumulative top-1 error versus coverage, ordered by the maximum class score.
Hard-negative accuracy uses 1,170 available confusion-edge comparisons and covers 8/18 labels;
neighbors outside the complete subset are not scored.

Qwen initially returned the consistent but incompatible schema `{"event": "..."}` because the
zero-shot prompt did not state the required `events` array. After explicitly specifying the output
schema, an 18-row smoke test and both full runs had zero parse failures. Definition minus direct
Qwen macro-F1 is -0.0003 (paired 95% CI [-0.0134, 0.0122]); definitions provide no measurable
benefit. Qwen direct versus OpenCLIP direct is also inconclusive: macro-F1 difference -0.0069
(95% CI [-0.0277, 0.0154]).

## Adapted generation validation

All three adapters produced valid JSON for all 2,384 validation pairs.

| Method | View | Macro-F1 (95% CI) | Micro-F1 (95% CI) | Exact acc. | Worst recall | Set-union macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Label-only Qwen3-VL LoRA | context | 0.3043 [0.2717, 0.3293] | 0.5474 [0.5009, 0.5916] | 0.5474 | 0.0500 | 0.3319 |
| Label-only Qwen3-VL LoRA | evidence | 0.6665 [0.6226, 0.6976] | 0.8486 [0.8243, 0.8701] | 0.8486 | 0.3125 | 0.6761 |
| Two-view Qwen3-VL LoRA | context + evidence | **0.7009 [0.6587, 0.7303]** | **0.8633 [0.8394, 0.8844]** | **0.8633** | **0.3000** | **0.7216** |

Using the same 10,000 paired session-group bootstrap samples, two-view minus context is +0.3966
macro-F1 (95% CI [0.3515, 0.4349]) and +0.3159 micro-F1 (95% CI [0.2835, 0.3495]).
These intervals support a development-set paired-view benefit, but not an official acquisition-level
claim because the `READY` provenance gate remains unmet.

Two-view improves per-class F1 for all 18/18 evaluated labels. The largest gains are
`glass_greenhouse` (+0.812), `grain_drying_road` (+0.731), and `bare_soil_netting` (+0.643).
Its weakest class remains `building_demolition` (F1 0.353), and the largest residual confusion is
103 `green_algae_duckweed -> unclean_water_surface` errors.

Evidence-only minus context-only is +0.3622 macro-F1 (95% CI [0.3157, 0.4018]) and +0.3012
micro-F1 ([0.2692, 0.3357]). The crop is therefore the dominant view. Two-view minus evidence-only
is a smaller but still positive +0.0344 macro-F1 ([0.0044, 0.0659]) and +0.0147 micro-F1
([0.0029, 0.0270]). Two-view improves 14/18 per-class F1 values; its largest gain is
`piled_materials_factory` (+0.243), while `building_demolition` decreases by 0.099. This supports
a modest complementary-context effect rather than attributing the full two-view score to fusion.

## Adapted training runs

All three runs use LoRA rank 16, alpha 32, dropout 0.05, AdamW, three epochs, batch 2, gradient
accumulation 8, class cap 250, and seed 42.

| Output | View | Optimizer steps | 15-minute check | 30-minute check | ~45-minute check | 60-minute check | 75-minute check | 90-minute check |
|---|---|---:|---|---|---|---|---|---|
| `outputs/context_label_seed42` | context | 672 | step 106, stable, ~10.0 s/step | step 178, stable, ~9.7 s/step | step 261, stable, ~9.9 s/step | step 370, stable, ~9.5 s/step | step 465, stable, ~9.5 s/step | step 558, stable, ~9.5 s/step |
| `outputs/two_view_seed42` | context + evidence | 672 | step 87, stable, ~11.2 s/step | step 152, stable, ~11.1 s/step | step 227, stable, ~11.0 s/step | step 325, stable, ~10.8 s/step | step 408, stable, ~10.9 s/step | step 491, stable, ~10.9 s/step |
| `outputs/evidence_label_seed42` | evidence | 672 | step 153, stable, ~5.7 s/step | step 315, stable, ~5.8 s/step | step 472, stable, ~5.8 s/step | step 630, stable, ~5.7 s/step | completed | completed |

At the 30-minute check, memory use was 20.9 GiB on GPU 0 and 23.4 GiB on GPU 1; no OOM,
NaN, or process exit was observed. The full source compile, test suite, and whitespace check also
passed (5 tests).

The context run wrote a complete resumable checkpoint at step 500 (epoch 2.233). Logged training
loss decreased from 1.8373 at the first log point to 0.07975 at step 500. Validation is required
before interpreting this decrease as useful generalization.

The two-view run also wrote a complete checkpoint at step 500 (epoch 2.233). Its logged loss fell
from 1.8051 to 0.02840 (minimum logged 0.02596). The lower training loss may reflect easier crop
memorization and is not treated as an improvement without validation.

Both three-epoch runs completed successfully: context runtime 6,471 s with aggregate train loss
0.1441; two-view runtime 7,285 s with aggregate train loss 0.08243. Final adapters were saved under
each run's `final/` directory. Eighteen-class generation and closed-set smoke evaluations produced
valid outputs for all 18 samples. Full 2,384-pair generation evaluation was healthy at its first
15-minute check (context 642 rows, two-view 610 rows; about 19.8 GiB per GPU).

The evidence-only run used the identical optimizer, cap, and seed. It completed 672/672 steps in
3,832 s with aggregate train loss 0.08278. Its first and final logged losses were 1.7417 and
0.01881 (minimum 0.01844); the final adapter is in
`outputs/evidence_label_seed42/final/`. Checks at 15, 30, 45, and 60 minutes found stable
5.6--5.9 s/step throughput, about 21.6 GiB memory, and no OOM or NaN.

## Closed-set validation

Validation thresholds are fitted and scored on the same development validation split here. These
numbers are useful diagnostics but are optimistic estimates, not test results.

The context-only run completed all 2,384 pairs and 1,551 content sets:

| Level / aggregation | Macro-F1 | Micro-F1 | Exact acc. | mAP | AURC ↓ |
|---|---:|---:|---:|---:|---:|
| Pair | 0.3045 | 0.5658 | 0.4056 | 0.2347 | 0.3460 |
| Independent pair union | 0.3227 | 0.5493 | 0.3656 | -- | -- |
| Log-sum-exp | 0.2231 | 0.3420 | 0.2141 | 0.1681 | 0.3898 |
| Max | **0.3281** | 0.5398 | 0.3385 | **0.2610** | **0.2486** |

Session-group bootstrap gives pair macro-F1 0.3045 (95% CI [0.2724, 0.3317]) and micro-F1
0.5658 (95% CI [0.5309, 0.5975]). Unnormalized log-sum-exp performs poorly because its score
contains a crop-count term. The preregistered max-pooling comparison removes that term and recovers
the best set macro-F1, although its micro-F1 and exact-set accuracy remain below independent union.
This is mixed evidence, not support for a blanket set-aggregation superiority claim.

The context-plus-evidence run also completed all 2,384 pairs and 1,551 content sets:

| Level / aggregation | Macro-F1 | Micro-F1 | Exact acc. | mAP | AURC ↓ |
|---|---:|---:|---:|---:|---:|
| Pair | 0.6069 | **0.8475** | **0.7735** | 0.5788 | 0.2514 |
| Independent pair union | 0.6177 | 0.8339 | 0.7369 | -- | -- |
| Log-sum-exp | 0.3019 | 0.4937 | 0.4784 | 0.2316 | 0.3311 |
| Max | **0.6280** | 0.8198 | 0.7240 | **0.6022** | **0.1754** |

Pair macro-F1 has a session-bootstrap 95% CI of [0.5653, 0.6354], and micro-F1 has
[0.8254, 0.8673]. Relative to context-only closed-set scoring, the paired view improves macro-F1
by 0.3024 (paired 95% CI [0.2597, 0.3399]) and micro-F1 by 0.2817
([0.2608, 0.3047]). Max pooling slightly improves set macro-F1 over independent union, but loses
micro-F1 and exact-set accuracy; unnormalized log-sum-exp is consistently poor.

The evidence-only run completed the same closed-set validation:

| Level / aggregation | Macro-F1 | Micro-F1 | Exact acc. | mAP | AURC ↓ |
|---|---:|---:|---:|---:|---:|
| Pair | 0.5753 | **0.8258** | **0.7538** | 0.5410 | 0.2146 |
| Independent pair union | 0.5886 | 0.8113 | 0.7092 | -- | -- |
| Log-sum-exp | 0.3499 | 0.5672 | 0.5319 | 0.2743 | 0.2213 |
| Max | **0.5973** | 0.8139 | 0.7118 | **0.5654** | **0.1474** |

Pair macro-F1 has a 95% CI of [0.5440, 0.5983], and micro-F1 has
[0.7997, 0.8494]. Two-view minus evidence-only is +0.0317 macro-F1, but its paired CI
[-0.0101, 0.0681] crosses zero; the +0.0217 micro-F1 gain has a positive CI
[0.0097, 0.0342]. Evidence-only minus context-only remains clearly positive:
+0.2707 macro-F1 ([0.2356, 0.3042]) and +0.2600 micro-F1 ([0.2416, 0.2798]).
Thus joint-view macro improvement depends on decoding, while its micro improvement is stable.
Evidence-only also has the strongest available hard-negative accuracy (0.7342 versus 0.6427 for
two-view and 0.4521 for context), although this diagnostic covers only 8/18 labels.
