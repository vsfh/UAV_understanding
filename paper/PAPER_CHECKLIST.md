# ICRA 2027 submission gates

## P0 — fail any item, do not submit

- [x] Generate deterministic review ledgers for 269 multi-label contexts, 223 non-exact near-duplicate groups, the exact cross-label pair conflict, and all 7,279 provenance contexts; validate case counts and image paths.
- [x] Generate isolated reviewer A/B assignments, offline visual evidence interfaces, a strict merger, and synthetic end-to-end validation for all 493 visual cases / 1,269 rows.
- [x] Recover all 1,549 class-level `.downloaded.txt` record IDs, prove that deterministic image joins are currently zero, and generate the R5 provenance ledger.
- [x] Generate a date/exact/near proxy split with zero cross-split hits and a frozen hash; mark it pipeline-only because 42 coarse components leave only four concepts above the weak 30/10/10 support diagnostic.
- [x] Freeze the R6 source-record, source-media, and exact-context schemas; validate 1,549/0/7,279-row structural completeness and emit a machine-readable `BLOCKED` gate.
- [x] Implement a fail-closed official split builder and prove both paths: refusal on the real blocked data and zero audited cross-split leakage on a synthetic READY fixture.
- [x] Freeze a machine-readable experiment registry whose main-result rows depend on the R6 provenance gate.
- [ ] Reconcile the 26,749 catalog records against the completed 9,093-row canonical pair manifest; explain the `crop_lodging` 8,204-record vs 11-visible-pair mismatch.
- [ ] Adjudicate the 269 multi-label contexts, 28 filename-parse exemplars, and the one exact pair reused across labels; record keep/relabel/merge/exclude decisions.
- [ ] Freeze two independent reviewer exports with distinct IDs, run the R4 merger, and obtain adjudicator-signed final fields; reviewer agreement alone is not ground truth.
- [ ] Review the 223 non-exact dHash candidate groups, then freeze exact-hash, perceptual/embedding-neighbor groups and a source/flight/location-disjoint split.
- [ ] Restore or explicitly declare unavailable the source/flight/site/time provenance; inferred file dates cannot support an unseen-location claim.
- [ ] Join `source_record_id -> source_media_id -> flight_id -> site_id -> verified capture time/permission`, then rebuild and re-hash the official split; do not promote the R5 proxy split.
- [ ] Achieve `official_split_gate=READY`: disposition all 1,549 source records and all 7,279 contexts, register at least four verified sites and eight verified flights, and retain review evidence.
- [ ] Report unique groups, sites, flights, time windows, and per-split class support; retain only classes that pass the preregistered Core gate.
- [ ] Replace every red `TBD`; remove all TODOs and placeholder claims.
- [ ] Run every reported model with three seeds; report mean, standard deviation, paired group-bootstrap 95% CIs, and prespecified tests.
- [ ] Complete data permission, privacy, sensitive-location, face/plate, and release-license review.
- [ ] Confirm double anonymity and remove identifying metadata, paths, acknowledgments, repository owners, and PDF metadata.
- [ ] Confirm Letter paper, official `ieeeconf.cls`, and no more than 8 total pages including references.
- [ ] Review ICRA 2027 AI-use disclosure requirements and add the required acknowledgment/disclosure.
- [ ] Compile with `latexmk -g -pdf`; resolve all undefined references, overfull boxes, and layout warnings; visually inspect every page.

## P1 — acceptance-strength evidence

- [ ] Zero-shot: OpenCLIP, GeoChat, Qwen2.5-VL, Qwen3-VL, all with direct and definition prompts.
- [ ] Controlled adaptations on the same Qwen3-VL backbone: label-only, generic-caption, grounded-caption, two-view concat, CLEAR.
- [ ] Context-only, crop-only, joint oracle-crop, and joint proposed-crop evaluations.
- [ ] View-necessity audit: single-view-insufficiency success, assigned-vs-random evidence deletion, and crop-budget F1 curve.
- [ ] Prompt-paraphrase robustness: freeze six meaning-preserving prompts and report maximum/standard-deviation spread without choosing the best prompt on test.
- [ ] Evidence-sensitivity diagnostic: prediction change under assigned-evidence deletion, alongside accuracy, so stable-but-wrong outputs are not rewarded.
- [ ] Pair-level multi-label macro/micro-F1 and mAP, using thresholds frozen on validation groups.
- [ ] Context-level set macro/micro-F1, exact-set accuracy, and crop-to-event evidence-assignment accuracy on adjudicated sample graphs.
- [ ] CLEAR-Set aggregation: independent-pair union, max, log-sum-exp, and attention under the same backbone, crop set, and budget.
- [ ] Taxonomy-neighbor vs frequency-matched random-negative ablation.
- [ ] Counterfactual, view-dropout, calibration, and abstention ablations.
- [ ] Unseen-location and small-evidence/adverse-capture/complex-background/proposal-crop stress tests.
- [ ] Human evaluation of evidence support, contradiction, hallucination, inter-rater agreement, and adjudication.
- [ ] Fixed qualitative gallery: success, failure, abstention, unsupported rationale, and leakage-like near duplicate.

## Go / no-go scientific decision

- [ ] CLEAR improves the prespecified primary comparison (grounded-caption Qwen3-VL) on Core macro-F1 with a positive paired 95% CI; otherwise pivot the paper to a dataset/protocol contribution and remove superiority language.
- [ ] Joint context+evidence outperforms the best single view; otherwise the paired-view thesis is not empirically supported.
- [ ] CLEAR-Set outperforms independent-pair union on set F1 and evidence assignment; otherwise the sample graph remains a dataset structure rather than a method contribution.
- [ ] Graph neighbors outperform frequency-matched random negatives on hard-negative pair accuracy; otherwise the graph-aware novelty claim is removed.
- [ ] The detector/retriever proposal gap is reported; oracle crops are never presented as deployable end-to-end performance.
