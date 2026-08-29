#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import List, Optional

import torch
from torch._library.infer_schema import SUPPORTED_PARAM_TYPES
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from clear_uav.experiment_config import load_yaml, project_path

# Hugging Face's FP8 kernel uses PEP 585 annotations unsupported by torch 2.6.
SUPPORTED_PARAM_TYPES[list[int]] = SUPPORTED_PARAM_TYPES[List[int]]
SUPPORTED_PARAM_TYPES[list[int] | None] = SUPPORTED_PARAM_TYPES[Optional[List[int]]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_detection(text: str, categories: set[str]) -> dict:
    value = json.loads(text)
    bbox = value["bbox_1000"]
    if set(value) != {"bbox_1000", "category", "confidence"} or len(bbox) != 4:
        raise ValueError("Invalid detection JSON")
    bbox = [float(number) for number in bbox]
    confidence = float(value["confidence"])
    if not all(0 <= number <= 1000 for number in bbox):
        raise ValueError("bbox_1000 is outside [0, 1000]")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("Invalid bbox corners")
    if value["category"] not in categories or not 0 <= confidence <= 1:
        raise ValueError("Invalid category or confidence")
    return {"bbox_1000": bbox, "category": value["category"], "confidence": confidence}


def iou(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def center_error(first: list[float], second: list[float]) -> float:
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return math.dist(first_center, second_center) / math.hypot(1000, 1000)


def summarize(ious: list[float], errors: list[float]) -> dict[str, float]:
    return {
        "median_iou": statistics.median(ious),
        "recall_at_iou_0.25": sum(value >= 0.25 for value in ious) / len(ious),
        "recall_at_iou_0.50": sum(value >= 0.50 for value in ious) / len(ious),
        "recall_at_iou_0.75": sum(value >= 0.75 for value in ious) / len(ious),
        "normalized_center_error": statistics.fmean(errors),
    }


def classification_metrics(
    targets: list[str], predictions: list[str | None], labels: list[str]
) -> dict:
    matrix = [[0 for _ in labels] for _ in labels]
    label_index = {label: index for index, label in enumerate(labels)}
    for target, prediction in zip(targets, predictions):
        if prediction is not None:
            matrix[label_index[target]][label_index[prediction]] += 1

    per_class = {}
    for label in labels:
        true_positive = sum(
            target == label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        false_positive = sum(
            target != label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        false_negative = sum(
            target == label and prediction != label
            for target, prediction in zip(targets, predictions)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(target == label for target in targets),
        }

    return {
        "accuracy": sum(target == prediction for target, prediction in zip(targets, predictions))
        / len(targets),
        "macro_f1": statistics.fmean(values["f1"] for values in per_class.values()),
        "weighted_f1": sum(
            values["f1"] * values["support"] for values in per_class.values()
        )
        / len(targets),
        "per_class": per_class,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix,
        "invalid_prediction_count": sum(prediction is None for prediction in predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen full-image detection evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    data_root = project_path(config["data"]["root"])
    definitions = json.loads(
        project_path(config["data"]["definitions"]).read_text(encoding="utf-8")
    )
    label_order = list(definitions)
    category_names = set(label_order)
    categories = "\n".join(
        f"- {label}: {definition}" for label, definition in definitions.items()
    )
    prompt = config["prompt"]["user"].replace("{categories}", categories)

    coco = json.loads(
        project_path(config["data"]["bbox_annotations"]).read_text(encoding="utf-8")
    )
    images = {image["id"]: image for image in coco["images"]}
    targets = {}
    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        targets[image["file_name"]] = [
            1000 * x / image["width"],
            1000 * y / image["height"],
            1000 * (x + width) / image["width"],
            1000 * (y + height) / image["height"],
        ]

    model_root = project_path(config["model"]["path"])
    model_path = next((model_root / "snapshots").glob("*/config.json")).parent
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=config["model"]["device_map"],
        local_files_only=True,
    )
    model.eval()

    for protocol in config["data"]["protocols"]:
        labels = {
            row["record_uid"]: row["source_class"]
            for row in read_csv(data_root / protocol / "test_labels_private.csv")
        }
        samples = []
        skipped = 0
        for row in read_csv(data_root / protocol / "test_inputs.csv"):
            image_file = row["context_path"].removeprefix("data/")
            image_path = data_root / image_file
            if not image_path.is_file() or image_file not in targets:
                skipped += 1
                continue
            samples.append(
                {
                    "record_uid": row["record_uid"],
                    "label": labels[row["record_uid"]],
                    "image_file": image_file,
                    "image_path": image_path,
                    "target_bbox": targets[image_file],
                }
            )
            maximum = config["data"].get("max_samples")
            if maximum and len(samples) == maximum:
                break

        rows = []
        detection_ious = []
        detection_errors = []
        bbox_ious = []
        bbox_errors = []
        valid_json = []
        target_categories = []
        predicted_categories = []
        with torch.inference_mode():
            for sample in tqdm(samples, desc=f"detection {protocol}", unit="image"):
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": config["prompt"]["system"]}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": str(sample["image_path"])},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                    processor_kwargs={
                        "size": {
                            "longest_edge": config["input"]["max_pixels"],
                            "shortest_edge": config["input"]["min_pixels"],
                        }
                    },
                ).to(model.device)
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=config["generation"]["max_new_tokens"],
                )
                raw = processor.decode(
                    generated[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                try:
                    prediction = parse_detection(raw, category_names)
                    bbox_iou = iou(prediction["bbox_1000"], sample["target_bbox"])
                    bbox_error = center_error(
                        prediction["bbox_1000"], sample["target_bbox"]
                    )
                    correct = prediction["category"] == sample["label"]
                    detection_iou = bbox_iou if correct else 0.0
                    detection_error = bbox_error if correct else 1.0
                    valid = True
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    prediction = None
                    bbox_iou = 0.0
                    bbox_error = 1.0
                    detection_iou = 0.0
                    detection_error = 1.0
                    correct = False
                    valid = False
                detection_ious.append(detection_iou)
                detection_errors.append(detection_error)
                bbox_ious.append(bbox_iou)
                bbox_errors.append(bbox_error)
                valid_json.append(valid)
                target_categories.append(sample["label"])
                predicted_categories.append(
                    prediction["category"] if prediction is not None else None
                )
                rows.append(
                    {
                        "record_uid": sample["record_uid"],
                        "image_file": sample["image_file"],
                        "target": {
                            "bbox_1000": sample["target_bbox"],
                            "category": sample["label"],
                        },
                        "raw_response": raw,
                        "prediction": prediction,
                        "valid_json": valid,
                        "category_correct": correct,
                        "iou": detection_iou,
                        "bbox_only_iou": bbox_iou,
                        "normalized_center_error": detection_error,
                    }
                )

        metrics = summarize(detection_ious, detection_errors)
        metrics["valid_json_rate"] = sum(valid_json) / len(valid_json)
        metrics["bbox_only"] = summarize(bbox_ious, bbox_errors)
        metrics.update(
            classification_metrics(target_categories, predicted_categories, label_order)
        )
        result = {
            "experiment": config["experiment"],
            "protocol": protocol,
            "primary_metric": "macro_f1",
            "num_samples": len(samples),
            "skipped_missing_images": skipped,
            "metrics": metrics,
            "rows": rows,
        }
        output = project_path(config["output"]["path"], protocol=protocol)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{protocol}: {metrics}")


if __name__ == "__main__":
    main()
