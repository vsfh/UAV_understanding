#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from clear_uav.metrics import pairwise_metrics, ranking_metrics
from clear_uav.ontology import load_ontology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class metrics and grouped bootstrap CIs")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--group-field", default="session_id")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["predictions"]


def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)


def main() -> None:
    args = parse_args()
    labels = [
        line.strip()
        for line in args.labels_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    label_index = {label: index for index, label in enumerate(labels)}
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["record_uid"]: row for row in csv.DictReader(handle)}
    rows = load_prediction_rows(args.predictions)
    groups = sorted({manifest[row["record_uid"]][args.group_field] for row in rows})
    group_index = {group: index for index, group in enumerate(groups)}

    shape = (len(groups), len(labels))
    tp = np.zeros(shape, dtype=np.int64)
    fp = np.zeros(shape, dtype=np.int64)
    fn = np.zeros(shape, dtype=np.int64)
    for row in rows:
        group = group_index[manifest[row["record_uid"]][args.group_field]]
        target = row["target"]
        prediction = row["prediction"]
        predicted = {prediction} if isinstance(prediction, str) else set(prediction)
        for label, index in label_index.items():
            tp[group, index] += label == target and label in predicted
            fp[group, index] += label != target and label in predicted
            fn[group, index] += label == target and label not in predicted

    total_tp, total_fp, total_fn = tp.sum(0), fp.sum(0), fn.sum(0)
    confusions = Counter()
    for row in rows:
        prediction = row["prediction"]
        predicted = {prediction} if isinstance(prediction, str) else set(prediction)
        for label in predicted - {row["target"]}:
            confusions[(row["target"], label)] += 1
    per_class = {}
    for label, index in label_index.items():
        precision_denominator = total_tp[index] + total_fp[index]
        recall_denominator = total_tp[index] + total_fn[index]
        precision = total_tp[index] / precision_denominator if precision_denominator else 0.0
        recall = total_tp[index] / recall_denominator if recall_denominator else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": float(f1_from_counts(total_tp[index], total_fp[index], total_fn[index])),
            "support": int(recall_denominator),
        }

    rng = np.random.default_rng(args.seed)
    macro_values = []
    micro_values = []
    for start in tqdm(
        range(0, args.resamples, 250),
        total=(args.resamples + 249) // 250,
        desc="group bootstrap",
        unit="chunk",
        dynamic_ncols=True,
    ):
        count = min(250, args.resamples - start)
        weights = rng.multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=count
        )
        sample_tp = weights @ tp
        sample_fp = weights @ fp
        sample_fn = weights @ fn
        macro_values.extend(f1_from_counts(sample_tp, sample_fp, sample_fn).mean(1))
        micro_values.extend(
            f1_from_counts(sample_tp.sum(1), sample_fp.sum(1), sample_fn.sum(1))
        )

    macro = float(f1_from_counts(total_tp, total_fp, total_fn).mean())
    micro = float(f1_from_counts(total_tp.sum(), total_fp.sum(), total_fn.sum()))
    result = {
        "num_rows": len(rows),
        "num_groups": len(groups),
        "group_field": args.group_field,
        "macro_f1": macro,
        "macro_f1_ci95": np.percentile(macro_values, [2.5, 97.5]).tolist(),
        "micro_f1": micro,
        "micro_f1_ci95": np.percentile(micro_values, [2.5, 97.5]).tolist(),
        "top_confusions": [
            {"target": target, "prediction": prediction, "count": count}
            for (target, prediction), count in confusions.most_common(20)
        ],
        "per_class": per_class,
    }
    if all("scores" in row for row in rows):
        ontology = load_ontology(args.ontology)
        result.update(
            ranking_metrics(
                [{row["target"]} for row in rows],
                [row["scores"] for row in rows],
                labels,
            )
        )
        result.update(
            pairwise_metrics(
                [{row["target"]} for row in rows],
                [row["scores"] for row in rows],
                {label: ontology.neighbors(label) for label in labels},
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"per_class", "top_confusions"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
