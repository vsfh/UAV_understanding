#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from clear_uav.data import cap_per_class, read_private_test_samples, read_samples
from clear_uav.metrics import classification_metrics, pairwise_metrics, ranking_metrics
from clear_uav.modeling import require_local_model
from clear_uav.ontology import load_label_subset, load_ontology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OpenCLIP zero-shot baseline")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-labels", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--prompt", choices=["direct", "definition"], default="definition")
    parser.add_argument("--view", choices=["context", "evidence"], default="context")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = require_local_model(args.model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ontology = load_ontology(args.ontology)
    labels = list(
        load_label_subset(args.labels_file, ontology)
        if args.labels_file
        else ontology.labels
    )
    included_labels = set(labels)
    if args.private_labels:
        samples = read_private_test_samples(
            args.csv,
            args.private_labels,
            args.data_root,
            limit=args.max_samples,
            include_labels=included_labels,
        )
    else:
        samples = read_samples(
            args.csv,
            args.data_root,
            limit=args.max_samples,
            include_labels=included_labels,
        )
    if args.max_per_class:
        samples = cap_per_class(samples, args.max_per_class, seed=0)

    texts = [
        f"a UAV image of {label.replace('_', ' ')}"
        if args.prompt == "direct"
        else f"a UAV image showing {ontology.definitions[label]}"
        for label in labels
    ]
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    model.eval()
    text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs).pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    predictions = []
    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            paths = [
                sample.context_path if args.view == "context" else sample.evidence_path
                for sample in batch
            ]
            images = [Image.open(path).convert("RGB") for path in paths]
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs).pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = image_features @ text_features.T
            indices = scores.argmax(dim=-1).tolist()
            for sample, index, sample_scores in zip(batch, indices, scores):
                prediction = labels[index]
                score_values = sample_scores.tolist()
                predictions.append({prediction})
                rows.append(
                    {
                        "record_uid": sample.record_uid,
                        "target": sample.label,
                        "prediction": prediction,
                        "score": score_values[index],
                        "scores": {
                            label: score for label, score in zip(labels, score_values)
                        },
                    }
                )

    evaluated_labels = sorted({sample.label for sample in samples})
    metrics = classification_metrics(
        [{sample.label} for sample in samples], predictions, evaluated_labels
    )
    metrics.update(
        ranking_metrics(
            [{sample.label} for sample in samples],
            [row["scores"] for row in rows],
            evaluated_labels,
        )
    )
    metrics.update(
        pairwise_metrics(
            [{sample.label} for sample in samples],
            [row["scores"] for row in rows],
            {label: ontology.neighbors(label) for label in evaluated_labels},
        )
    )
    result = {
        "config": {
            "model_path": str(model_path),
            "prompt": args.prompt,
            "view": args.view,
            "labels": labels,
        },
        "num_pairs": len(samples),
        "metrics": metrics,
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"num_pairs": len(samples), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
