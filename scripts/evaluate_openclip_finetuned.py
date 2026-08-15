#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from clear_uav.data import cap_per_class, read_private_test_samples, read_samples
from clear_uav.metrics import classification_metrics, pairwise_metrics, ranking_metrics
from clear_uav.modeling import require_local_model
from clear_uav.openclip_finetune import load_finetuned_checkpoint, pooled_features
from clear_uav.ontology import load_label_subset, load_ontology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OpenCLIP linear probe or visual fine-tune"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-labels", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--view", choices=["context", "evidence"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OpenCLIP evaluation requires CUDA")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda")
    model_path = require_local_model(args.model_path)
    ontology = load_ontology(args.ontology)
    labels = list(load_label_subset(args.labels_file, ontology))
    included = set(labels)
    if args.private_labels:
        samples = read_private_test_samples(
            args.csv,
            args.private_labels,
            args.data_root,
            limit=args.max_samples,
            include_labels=included,
        )
    else:
        samples = read_samples(
            args.csv,
            args.data_root,
            limit=args.max_samples,
            include_labels=included,
        )
    if args.max_per_class:
        samples = cap_per_class(samples, args.max_per_class, seed=0)
    if not samples:
        raise ValueError("No evaluation samples selected")

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    classifier, checkpoint = load_finetuned_checkpoint(
        args.checkpoint, model=model, device=device
    )
    checkpoint_labels = checkpoint["labels"]
    if checkpoint_labels != labels:
        raise ValueError("Checkpoint labels do not exactly match --labels-file order")
    view = args.view or checkpoint["view"]
    model.eval()
    classifier.eval()

    rows = []
    targets = []
    predictions = []
    score_rows = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(samples), args.batch_size),
            total=(len(samples) + args.batch_size - 1) // args.batch_size,
            desc=f"OpenCLIP {checkpoint['mode']} {view}",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch = samples[start : start + args.batch_size]
            paths = [
                sample.context_path if view == "context" else sample.evidence_path
                for sample in batch
            ]
            images = [load_rgb(path) for path in paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = pooled_features(model.get_image_features(**inputs))
                logits = classifier(features)
            logits_cpu = logits.float().cpu()
            indices = logits_cpu.argmax(-1).tolist()
            for sample, index, sample_logits in zip(batch, indices, logits_cpu):
                prediction = labels[index]
                scores = {
                    label: float(score)
                    for label, score in zip(labels, sample_logits.tolist())
                }
                targets.append({sample.label})
                predictions.append({prediction})
                score_rows.append(scores)
                rows.append(
                    {
                        "record_uid": sample.record_uid,
                        "target": sample.label,
                        "prediction": prediction,
                        "score": scores[prediction],
                        "scores": scores,
                    }
                )

    metrics = classification_metrics(targets, predictions, labels)
    metrics.update(ranking_metrics(targets, score_rows, labels))
    metrics.update(
        pairwise_metrics(
            targets,
            score_rows,
            {label: ontology.neighbors(label) for label in labels},
        )
    )
    result = {
        "config": {
            "model_path": str(model_path),
            "checkpoint": str(args.checkpoint),
            "mode": checkpoint["mode"],
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_best_macro_f1": checkpoint["best_macro_f1"],
            "view": view,
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
