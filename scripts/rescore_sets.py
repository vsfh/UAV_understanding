#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from clear_uav.metrics import classification_metrics, ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore cached closed-set edges with a different set aggregator"
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--aggregator", choices=["max", "logsumexp"], required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--fit-thresholds",
        action="store_true",
        help="Fit set thresholds on this file and write OUTPUT.thresholds.json",
    )
    threshold_group.add_argument(
        "--thresholds",
        type=Path,
        help="Apply thresholds previously fitted by this script",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def best_threshold(scores: list[float], targets: list[bool]) -> float:
    candidates = [max(scores) + 1e-6, *sorted(set(scores), reverse=True)]
    best_f1, best_value = -1.0, candidates[0]
    for value in candidates:
        tp = sum(score >= value and target for score, target in zip(scores, targets))
        fp = sum(score >= value and not target for score, target in zip(scores, targets))
        fn = sum(score < value and target for score, target in zip(scores, targets))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1, best_value = f1, value
    return best_value


def main() -> None:
    args = parse_args()
    # Preserve the original validation-only CLI: omitting both flags fits thresholds.
    fit_thresholds = args.fit_thresholds or args.thresholds is None
    source = json.loads(args.scores.read_text(encoding="utf-8"))
    labels = source["labels"]
    grouped_scores = defaultdict(list)
    grouped_targets = defaultdict(set)
    for row in source["predictions"]:
        grouped_scores[row["group_id"]].append(row["scores"])
        grouped_targets[row["group_id"]].add(row["target"])

    group_ids = sorted(grouped_scores)
    score_rows = []
    for group_id in group_ids:
        rows = grouped_scores[group_id]
        if args.aggregator == "max":
            score_rows.append(
                {label: max(row[label] for row in rows) for label in labels}
            )
        else:
            score_rows.append(
                {
                    label: float(
                        args.temperature
                        * torch.logsumexp(
                            torch.tensor([row[label] for row in rows])
                            / args.temperature,
                            dim=0,
                        )
                    )
                    for label in labels
                }
            )

    targets = [grouped_targets[group_id] for group_id in group_ids]
    if fit_thresholds:
        thresholds = {
            label: best_threshold(
                [row[label] for row in score_rows],
                [label in target for target in targets],
            )
            for label in labels
        }
        threshold_path = args.output.with_suffix(".thresholds.json")
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(
            json.dumps(
                {
                    "aggregator": args.aggregator,
                    "temperature": args.temperature,
                    "labels": labels,
                    "thresholds": thresholds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if threshold_payload["aggregator"] != args.aggregator:
            raise ValueError("Aggregator differs from the fitted set thresholds")
        if threshold_payload["temperature"] != args.temperature:
            raise ValueError("Temperature differs from the fitted set thresholds")
        if threshold_payload["labels"] != labels:
            raise ValueError("Labels differ from the fitted set thresholds")
        thresholds = threshold_payload["thresholds"]
    predictions = [
        {label for label in labels if row[label] >= thresholds[label]}
        for row in score_rows
    ]
    metrics = classification_metrics(targets, predictions, labels)
    metrics.update(ranking_metrics(targets, score_rows, labels))
    result = {
        "source": str(args.scores),
        "aggregator": args.aggregator,
        "temperature": args.temperature,
        "threshold_source": (
            str(args.output.with_suffix(".thresholds.json"))
            if fit_thresholds
            else str(args.thresholds)
        ),
        "thresholds": thresholds,
        "metrics": metrics,
        "sets": [
            {
                "group_id": group_id,
                "targets": sorted(target),
                "predictions": sorted(prediction),
                "scores": scores,
            }
            for group_id, target, prediction, scores in zip(
                group_ids, targets, predictions, score_rows
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
