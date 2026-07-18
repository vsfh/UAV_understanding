# UTS-UAV — ICRA 2027 Revision 6

This is the upload-ready LaTeX project for the double-blind ICRA 2027 draft. It uses the PaperCept `ieeeconf.cls`, US Letter paper, 10 pt, two columns, and an eight-page total limit including references.

## Scientific status

UTS-UAV is a development snapshot toward a context–evidence benchmark, not yet a release-ready 67-class dataset. The content audit verifies 9,093 pair annotations across 45 present concepts, 7,279 unique wide contexts, 9,056 unique crops, and 9,082 unique context–crop identities. It also recovers 1,389 multi-evidence contexts and 269 multi-label contexts.

Revision 6 adds a normalized, fail-closed provenance release gate:

1. `source_record_resolution_v1.csv` dispositions all 1,549 business records;
2. `source_media_registry_v1.csv` records canonical media, flight, site, capture time, permission, privacy, and review evidence;
3. `context_source_link_v1.csv` links or excludes all 7,279 exact contexts.

The templates are structurally valid, but the scientific gate is currently `BLOCKED`: 0/1,549 source records are resolved, 0 media rows are verified, and 0/7,279 contexts are linked or excluded. Therefore no official split, Core classes, main scores, flight/site-disjoint claim, or unseen-location claim exists. The split builder refuses to emit an official manifest until the gate is `READY`.

`../data/provenance_r5/` contains only the date/content proxy split for parser, dataloader, and training smoke tests. `../data/provenance_r6/` contains the normalized release templates, schema, experiment registry, and current machine-readable gate result. The synthetic READY-path test is software validation only and is never dataset evidence.

## Compile

Choose pdfLaTeX and `main.tex` in Overleaf, or run:

```bash
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Important paths

- `sections/`, `figure/`, and `table/`: paper source;
- `PAPER_CHECKLIST.md`: submission hard gates;
- `../data/audit_v2/`: canonical content manifest and audit;
- `../data/review_r4/`: isolated A/B review, strict merger, and tests;
- `../data/provenance_r5/`: proxy split pressure test only;
- `../data/provenance_r6/`: normalized provenance release gate and experiment registry;
- `../analysis/recent_dataset_paper_scan_2026-07-17.md`: 2026 benchmark-design scan.

Internal example images must not be published before permission and privacy review. Red `TBD` cells are experimental slots, not results; only outputs traceable to the frozen official manifest, configuration, seed, and raw prediction files may enter the submission.
