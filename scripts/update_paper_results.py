#!/usr/bin/env python3
"""Audit saved predictions and refresh Tables V/VI. Never train or change results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from audit_table4_results import normalize, compute
from paper_result_reference import build_expected_rows, PROTOCOLS

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "full_cls": "Full matched control",
    "no_heatmap_cls": "Without heatmap guidance",
    "single_scale_cls": "Single-scale features",
    "fixed_classifier": "Linear instead of prototype classifier",
    "no_roi": "Without ROI features",
    "no_global": "Without global features",
    "no_curriculum": "Without box curriculum",
}
SYSTEMS = {"qwen": "Direct Qwen3-VL", "tiling": "Qwen adaptive tiling",
           "ground_cls": "Qwen-Ground (matched initialization)"}
METRICS = ("p_ap", "ap50", "c_f1", "g_map50", "n_fpr", "p_recall")


def load_result(path, protocol, expected_epoch=None, expected_experiment=None,
                allow_legacy_missing_seed=False):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != protocol:
        raise ValueError(f"Wrong protocol: {path}")
    legacy_seed_exception = (allow_legacy_missing_seed and protocol == "session_disjoint"
                             and expected_experiment in ("table4_qwen3vl_t4", "table4_qwen3vl_adaptive_tiling"))
    if payload.get("seed") != 43 and not ("seed" not in payload and legacy_seed_exception):
        raise ValueError(f"Wrong seed: {path}")
    if expected_experiment is not None and payload.get("experiment") != expected_experiment:
        raise ValueError(f"Wrong experiment: {path}")
    if expected_epoch is not None and payload.get("checkpoint_epoch") != expected_epoch:
        raise ValueError(f"Incomplete checkpoint in result: {path}")
    rows = [normalize(row) for row in payload["rows"]]
    ids = [r["record_uid"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate record IDs: {path}")
    for row in rows:
        target, pred = row["target"], row["prediction"]
        if not isinstance(target["presence"], bool):
            raise ValueError(f"Invalid target presence: {path}")
        if not target["presence"] and (target["category"] is not None or target["bbox_1000"] is not None):
            raise ValueError(f"Negative target must have no class or bbox: {path}")
        if not math.isfinite(pred["presence_score"]) or not 0 <= pred["presence_score"] <= 1:
            raise ValueError(f"Invalid score: {path}")
        for box in (target["bbox_1000"], pred["bbox_1000"]):
            if box is not None and (len(box) != 4 or not all(math.isfinite(v) for v in box)
                                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError(f"Invalid box: {path}")
    return payload, rows


def check_targets(rows, reference):
    lookup = {r["record_uid"]: r for r in reference}
    ids = [r["record_uid"] for r in rows]
    if len(ids) != len(set(ids)) or len(reference) != len(lookup):
        raise ValueError("Duplicate test record IDs")
    if set(ids) != set(lookup):
        raise ValueError("Test record IDs differ from frozen source reference")
    for row in rows:
        ref = lookup[row["record_uid"]]
        a, b = row["target"], ref["target"]
        if row.get("group_id") != ref.get("group_id") or (a["presence"], a["category"]) != (b["presence"], b["category"]):
            raise ValueError("Group or target label differs from frozen source reference")
        if not a["presence"] and (a["bbox_1000"] is not None or a["category"] is not None):
            raise ValueError("Negative target must have no class or bbox")
        if a["presence"] and (a["bbox_1000"] is None or b["bbox_1000"] is None
                              or len(a["bbox_1000"]) != 4 or len(b["bbox_1000"]) != 4
                              or any(abs(x-y) >= .001 for x, y in zip(a["bbox_1000"], b["bbox_1000"]))):
            raise ValueError("Target box differs from frozen source reference")


def summarize(path, payload, rows, labels):
    stored = payload["metrics"].get("table4", payload["metrics"])
    for row in rows:
        if row["prediction"]["category"] not in (None, *labels):
            raise ValueError(f"Unknown predicted category: {path}")
    threshold = stored["threshold"]
    if not math.isfinite(threshold):
        raise ValueError(f"Invalid threshold: {path}")
    metrics = compute(rows, labels, threshold)
    for key in METRICS:
        if stored.get(key) is None or not math.isclose(metrics[key], stored[key], abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError(f"Stored {key} disagrees with predictions: {path}")
    positive_rows = [r for r in rows if r["target"]["presence"]]
    positive = compute(positive_rows, labels, threshold)
    return {"source": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(rows), "groups": len({r.get("group_id") for r in rows}),
            "uid_sha256": hashlib.sha256("\n".join(sorted(r["record_uid"] for r in rows)).encode()).hexdigest(),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "experiment": payload.get("experiment"), "protocol": payload.get("protocol"),
            "seed": payload.get("seed"), "missing_seed_metadata": "seed" not in payload,
            "seed_evidence": ("registered seed43 path/config only; absent from legacy result JSON"
                              if "seed" not in payload else "explicit result JSON seed=43"),
            "metrics": metrics, "positive_only": {k: positive[k] for k in ("c_f1", "g_map50", "ap50")}}


def audit(root, negative_inventory=None):
    root = Path(root)
    references = {protocol: build_expected_rows(root, protocol, negative_inventory)
                  for protocol in PROTOCOLS}
    labels = references["session_disjoint"]["labels"]
    if len(labels) != 18 or any(ref["labels"] != labels for ref in references.values()):
        raise ValueError("Expected one frozen core18 label subset for every protocol")
    reference = references["session_disjoint"]["rows"]
    matched = {}
    for name in VARIANTS:
        path = root / f"results/table4_matched_ablations/{name}/session_disjoint/seed43_test.json"
        if not path.exists():
            matched[name] = None
            continue
        payload, rows = load_result(path, "session_disjoint", expected_epoch=12,
                                    expected_experiment="table4_matched_" + name)
        check_targets(rows, reference)
        matched[name] = summarize(path, payload, rows, labels)
    domains = {}
    for protocol in PROTOCOLS:
        domains[protocol] = {}
        domain_reference = references[protocol]["rows"]
        for system in SYSTEMS:
            if protocol == "session_disjoint" and system == "ground_cls":
                domains[protocol][system] = matched["full_cls"]
                continue
            model = {"qwen": "qwen3vl_t4", "tiling": "qwen3vl_adaptive_tiling", "ground_cls": "ground_cls"}[system]
            directory = "table4" if protocol == "session_disjoint" else "table4_shifts"
            path = root / f"results/{directory}/{model}/{protocol}/seed43_test.json"
            if not path.exists():
                domains[protocol][system] = None
                continue
            experiment = ({"qwen": "table4_qwen3vl_t4", "tiling": "table4_qwen3vl_adaptive_tiling"}[system]
                          if protocol == "session_disjoint" else
                          {"qwen": "paper_shift_qwen", "tiling": "paper_shift_tiling",
                           "ground_cls": "paper_shift_ground_cls"}[system])
            payload, rows = load_result(path, protocol, 12 if system == "ground_cls" else None,
                                        expected_experiment=experiment,
                                        allow_legacy_missing_seed=protocol == "session_disjoint"
                                        and system in ("qwen", "tiling"))
            check_targets(rows, domain_reference)
            domains[protocol][system] = summarize(path, payload, rows, labels)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 43,
            "labels": labels, "matched": matched, "domains": domains,
            "reference_provenance": {p: ref["provenance"] for p, ref in references.items()},
            "notes": ["Single-seed point estimates, not significance tests.",
                      "UIDs/groups/targets come independently from frozen split CSVs, core18, COCO and no-event source filenames, never a result file.",
                      "Only the registered legacy session Qwen/tiling entries may lack seed metadata; missing_seed_metadata flags this explicitly.",
                      "G+ removes all negative records before AP ranking; it is not all-record G-mAP50.",
                      "Validation thresholds are read, never selected from test.",
                      "Checkpoint/config receipt validation is separate from this prediction-only audit."]}


def score(entry, key, positive=False):
    return r"\pending" if entry is None else f"{100*entry['positive_only' if positive else 'metrics'][key]:.2f}"


def matched_table(report):
    text = r"""\begin{table*}[t]
\centering
\caption{Matched Qwen-Ground ablations (session-disjoint, seed 43; percent).
The independent full control uses 12 MS and 12 CLS epochs; geometric controls
change both stages, while other controls share full-MS initialization.
Red entries are unfinished. All rows use the same metric definitions as Table~\ref{tab:main}.}
\label{tab:matched_ablation}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccccc}
\toprule
Variant & AP50 & C-F1 & G-mAP50 & N-FPR & P-R \\
\midrule
"""
    for name, label in VARIANTS.items():
        text += label + " & " + " & ".join(score(report["matched"][name], k) for k in ("ap50", "c_f1", "g_map50", "n_fpr", "p_recall")) + " \\\\\n"
    return text + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def domain_table(report):
    text = r"""\begin{table*}[t]
\centering
\caption{Cross-domain evaluation, seed 43 (percent). Each model is trained
on its protocol's own train split. G$^+$ recomputes G-mAP50 after removing
no-event records from AP ranking; it is not Table~\ref{tab:main}'s all-record G-mAP50.
No-event operating points are evaluated separately; red entries are unfinished.}
\label{tab:cross_domain_slots}
\small
\setlength{\tabcolsep}{7pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Session-disjoint} & \multicolumn{2}{c}{Unseen site} & \multicolumn{2}{c}{Forward time} \\
System & C-F1 & G$^+$ & C-F1 & G$^+$ & C-F1 & G$^+$ \\
\midrule
"""
    for system, label in SYSTEMS.items():
        values = [score(report["domains"][p][system], key, positive=True)
                  for p in ("session_disjoint", "unseen_site", "forward_temporal")
                  for key in ("c_f1", "g_map50")]
        text += label + " & " + " & ".join(values) + " \\\\\n"
    return text + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--negative-inventory", type=Path,
                        help="Offline full no-event PNG filename inventory JSON (path relative to --root)")
    args = parser.parse_args()
    report = audit(args.root, negative_inventory=args.negative_inventory)
    paper = args.paper or args.root / "paper"
    output = args.report or args.root / "reports/paper_results_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (paper / "table").mkdir(parents=True, exist_ok=True)
    for name, content in (("matched_ablation_slots.tex", matched_table(report)),
                          ("cross_domain_slots.tex", domain_table(report))):
        (paper / "table" / name).write_text(content, encoding="utf-8")
    for name, entry in report["matched"].items():
        print(name, "pending" if entry is None else {k: round(100*entry["metrics"][k], 2) for k in METRICS})
    print(f"Updated Tables V/VI and {output}. Narrative/PDF still require review after new results.")


if __name__ == "__main__":
    main()
