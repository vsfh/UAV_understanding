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


def parse_bbox(text: str) -> list[float]:
    value = json.loads(text)
    bbox = value["bbox_1000"]
    if set(value) != {"bbox_1000"} or len(bbox) != 4:
        raise ValueError("Invalid localization JSON")
    bbox = [float(number) for number in bbox]
    if not all(0 <= number <= 1000 for number in bbox):
        raise ValueError("bbox_1000 is outside [0, 1000]")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("Invalid bbox corners")
    return bbox


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen full-image localization evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    data_root = project_path(config["data"]["root"])
    definitions = json.loads(
        project_path(config["data"]["definitions"]).read_text(encoding="utf-8")
    )
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
        ious = []
        errors = []
        with torch.inference_mode():
            for sample in tqdm(samples, desc=f"localization {protocol}", unit="image"):
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
                    prediction = parse_bbox(raw)
                    sample_iou = iou(prediction, sample["target_bbox"])
                    sample_error = center_error(prediction, sample["target_bbox"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    prediction = None
                    sample_iou = 0.0
                    sample_error = 1.0
                ious.append(sample_iou)
                errors.append(sample_error)
                rows.append(
                    {
                        "record_uid": sample["record_uid"],
                        "image_file": sample["image_file"],
                        "target_bbox_1000": sample["target_bbox"],
                        "raw_response": raw,
                        "prediction_bbox_1000": prediction,
                        "iou": sample_iou,
                        "normalized_center_error": sample_error,
                    }
                )

        result = {
            "experiment": config["experiment"],
            "protocol": protocol,
            "num_samples": len(samples),
            "skipped_missing_images": skipped,
            "metrics": summarize(ious, errors),
            "rows": rows,
        }
        output = project_path(config["output"]["path"], protocol=protocol)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{protocol}: {result['metrics']}")


if __name__ == "__main__":
    main()
