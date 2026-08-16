from __future__ import annotations

import json
from pathlib import Path

from clear_uav.data import cap_per_class, read_samples
from clear_uav.experiment_config import project_path
from clear_uav.metrics import classification_metrics, pairwise_metrics, ranking_metrics


def evaluation_samples(config: dict, protocol: str, labels: list[str]):
    data = config["data"]
    root = project_path(data["root"])
    samples = read_samples(
        root / protocol / f"{data['split']}.csv",
        root,
        include_labels=set(labels),
    )
    maximum = data.get("max_per_class")
    return cap_per_class(samples, maximum, 0) if maximum else samples


def scored_metrics(samples, predictions, scores, labels, ontology) -> dict:
    targets = [{sample.label} for sample in samples]
    metrics = classification_metrics(targets, predictions, labels)
    metrics.update(ranking_metrics(targets, scores, labels))
    metrics.update(
        pairwise_metrics(
            targets,
            scores,
            {label: ontology.neighbors(label) for label in labels},
        )
    )
    return metrics


def save_result(path: Path, config: dict, samples, predictions, scores, metrics) -> None:
    rows = [
        {
            "record_uid": sample.record_uid,
            "target": sample.label,
            "prediction": sorted(prediction),
            "scores": score,
        }
        for sample, prediction, score in zip(samples, predictions, scores)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"config": config, "num_samples": len(samples), "metrics": metrics, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
