"""Paired ROI-branch intervention summary, with frozen cached text and confidence."""
import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clear_uav.experiment_config import project_path
from clear_uav.table3 import labels_from_config
from clear_uav.table4 import box_iou, table4_metrics

CONDITIONS = ("full", "no_spatial", "shuffle_xy", "global_memory")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    """Only an unfinished final append may be ignored; interior corruption fails."""
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise
    return rows


def scores(rows, condition):
    values = {}
    for row in rows:
        if not row["target"]["presence"]:
            continue
        prediction = row["cached_prediction"] if condition == "cached" else row["predictions"][condition]
        box = prediction["bbox_1000"]
        valid = (prediction["valid"] and box is not None and len(box) == 4
                 and all(math.isfinite(value) for value in box)
                 and box[0] < box[2] and box[1] < box[3])
        iou = box_iou(row["target"]["bbox_1000"], box) if valid else 0.0
        values[row["record_uid"]] = (iou, iou >= .5 and prediction["category"] == row["target"]["category"])
    return values


def bootstrap(rows, full, alternative, draws, seed):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        uid = row["record_uid"]
        if uid in full:
            group = groups[row.get("group_id") or uid]
            group[0] += int(full[uid][1]) - int(alternative[uid][1])
            group[1] += 1
    if len(groups) < 2 or draws < 2:
        return None
    rng, blocks = random.Random(seed), list(groups.values())
    estimates = []
    for _ in range(draws):
        sample = rng.choices(blocks, k=len(blocks))
        estimates.append(sum(block[0] for block in sample) / sum(block[1] for block in sample))
    estimates.sort()
    return [estimates[int(.025 * (draws - 1))], estimates[int(.975 * (draws - 1))]]


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yaml/spatial_evidence.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source, output = project_path(config["source_dir"]), project_path(config["output"])
    trained = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    split = config.get("split", "test")
    saved, plan = read_json(source / f"{split}_results.json"), read_json(output / "plan.json")
    if plan.get("source_sha256") and hashlib.sha256((source / f"{split}_results.json").read_bytes()).hexdigest() != plan["source_sha256"]:
        raise ValueError("Cached source predictions changed since the intervention run")
    if saved["protocol"] != trained["protocol"] or saved["seed"] != trained["seed"]:
        raise ValueError("Cached results disagree with checkpoint protocol/seed")
    reference = {row["record_uid"]: row for row in saved["rows"]}
    planned = plan["record_uids"]
    if (len(reference) != len(saved["rows"]) or len(set(planned)) != len(planned)
            or plan["total_records"] != len(planned) or set(planned) != set(reference)):
        raise ValueError("Plan must contain every unique cached split record exactly once")
    rows = read_rows(output / "paired_predictions.jsonl")
    seen = set()
    for row in rows:
        uid = row["record_uid"]
        if uid in seen or uid not in reference or set(row["predictions"]) != set(CONDITIONS):
            raise ValueError(f"Duplicate/unplanned ID or unpaired conditions: {uid}")
        seen.add(uid)
        cached = reference[uid]
        if (row["target"] != cached["target"] or row.get("group_id") != cached.get("group_id")
                or row["cached_prediction"] != cached["prediction"]):
            raise ValueError(f"Cached prediction or ground truth changed: {uid}")
        frozen = {key: value for key, value in cached["prediction"].items() if key != "bbox_1000"}
        for prediction in row["predictions"].values():
            if "bbox_1000" not in prediction or {key: value for key, value in prediction.items() if key != "bbox_1000"} != frozen:
                raise ValueError(f"Intervention changed more than ROI coordinates: {uid}")
    indexed = {row["record_uid"]: row for row in rows}
    # Keep original cached order for stable AP tie-breaking, not processing order.
    rows = [indexed[uid] for uid in reference if uid in indexed]
    complete = seen == set(planned)
    state = "COMPLETE" if complete else "PARTIAL"
    metrics = saved["metrics"].get("table4", saved["metrics"])
    threshold = float(metrics["threshold"])
    labels, results = labels_from_config(trained), {}
    samples = [SimpleNamespace(presence=row["target"]["presence"], label=row["target"]["category"],
                               bbox_1000=row["target"]["bbox_1000"]) for row in rows]
    all_scores = {name: scores(rows, name) for name in (*CONDITIONS, "cached")}
    for name in CONDITIONS:
        values = list(all_scores[name].values())
        benchmark = table4_metrics(samples, [row["predictions"][name] for row in rows], labels, threshold,
                                   any(sample.presence for sample in samples)) if rows else None
        if benchmark:
            for key in ("median_ms", "mean_calls", "max_calls", "timing_scopes"):
                benchmark.pop(key, None)  # Cached timing is not intervention runtime.
        results[name] = {"benchmark": benchmark,
                        "mean_iou": statistics.fmean(value[0] for value in values) if values else None,
                        "l50": statistics.fmean(value[0] >= .5 for value in values) if values else None,
                        "j50": statistics.fmean(value[1] for value in values) if values else None}
    if rows:
        for name in CONDITIONS[1:]:
            for key in ("c_f1", "n_fpr", "p_recall"):
                if results[name]["benchmark"][key] != results["full"]["benchmark"][key]:
                    raise ValueError("Frozen text/presence metrics unexpectedly changed")
    for name in CONDITIONS[1:]:
        full, alternative = all_scores["full"], all_scores[name]
        results[name]["change_vs_full"] = {key: results[name][key] - results["full"][key] if full else None
                                          for key in ("mean_iou", "l50", "j50")}
        results[name]["wins_vs_full"] = sum(alternative[uid][1] and not full[uid][1] for uid in full)
        results[name]["losses_vs_full"] = sum(full[uid][1] and not alternative[uid][1] for uid in full)
        results[name]["j50_drop_full_minus_condition_ci95"] = bootstrap(rows, full, alternative, config.get("bootstrap_draws", 1000), trained["seed"])
    shifts, missing = [], 0
    for row in rows:
        first, second = row["cached_prediction"]["bbox_1000"], row["predictions"]["full"]["bbox_1000"]
        if first is not None and second is not None:
            shifts.append(max(abs(a - b) for a, b in zip(first, second)))
        else:
            missing += (first is None) != (second is None)
    drift = {"comparable_boxes": len(shifts), "missing_box_disagreements": missing,
             "median_max_coordinate_shift_1000": statistics.median(shifts) if shifts else None,
             "max_coordinate_shift_1000": max(shifts) if shifts else None,
             "j50_disagreements": sum(all_scores["cached"][uid][1] != value[1] for uid, value in all_scores["full"].items())}
    status = read_json(output / "status.json") if (output / "status.json").is_file() else None
    notes = ["ROI-branch reliance only; not a training ablation or whole-Qwen faithfulness test.",
             "Category, confidence and all non-box fields are frozen; C-F1/N-FPR cannot improve here.",
             "meanIoU/L50/J50 use all processed positives before presence gating; missing/invalid boxes fail.",
             "A weak no_spatial effect is valid evidence of low branch reliance, not a failed experiment.",
             "Cached timing/call counts are not intervention runtime; see worker_status for wall time.",
             "CI resamples paired content groups; records without a group use singleton record IDs."]
    result = {"state": state, "processed_records": len(rows), "planned_records": len(planned),
              "checkpoint": plan.get("checkpoint", str((source / "best").resolve())),
              "checkpoint_sha256": plan.get("checkpoint_sha256"), "residual_gate": plan.get("residual_gate"),
              "protocol": trained["protocol"], "seed": trained["seed"], "split": split,
              "full_benchmark_claim_allowed": complete, "frozen_threshold": threshold,
              "conditions": results, "cache_vs_full_replay": drift, "worker_status": status, "notes": notes}
    lines = [f"{state}: {len(rows)}/{len(planned)} paired records", f"Checkpoint: {result['checkpoint']}",
             "G-mAP50 / meanIoU / L50 / J50 (all conditions use exactly the same processed IDs)"]
    for name, value in results.items():
        lines.append(f"{name}: " + " / ".join(pct(item) for item in [value["benchmark"]["g_map50"] if value["benchmark"] else None, value["mean_iou"], value["l50"], value["j50"]]))
        if name != "full":
            lines.append(f"  J50 change={pct(value['change_vs_full']['j50'])}; wins/losses vs full={value['wins_vs_full']}/{value['losses_vs_full']}; drop CI95={value['j50_drop_full_minus_condition_ci95']}")
    lines.extend([f"Cache/full replay drift: {json.dumps(drift)}", f"Worker status: {json.dumps(status)}", *notes])
    if not complete:
        lines.append("PARTIAL pilot only; these are not complete benchmark results.")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
