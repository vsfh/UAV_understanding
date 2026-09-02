from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForMultimodalLM,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    DFineForObjectDetection,
)

from clear_uav.experiment_config import load_yaml, project_path
from clear_uav.generation_constraints import (
    grounding_prefix_allowed_tokens,
    label_prefix_allowed_tokens,
    location_token_strings,
)
from clear_uav.modeling import LORA_PATTERNS, assistant_only_labels, load_qwen
from clear_uav.table3 import (
    EncoderClassifier,
    class_sampler,
    cosine_scheduler,
    encoder_hidden_dim,
    labels_from_config,
    seed_everything,
)


@dataclass(frozen=True)
class DiscoverySample:
    record_uid: str
    label: str | None
    presence: bool
    image_path: Path
    evidence_path: Path | None
    bbox_1000: tuple[float, float, float, float] | None
    group_id: str | None = None
    negative_subtype: str | None = None


NO_EVENT_TIMESTAMP = re.compile(
    r"photo - (\d{4}-\d{2}-\d{2}T\d{6}\.\d+)\.png$"
)


def ensure_hf_model(model_config: dict) -> Path:
    destination = project_path(model_config["path"])
    has_weights = any(destination.glob("*.safetensors")) or any(
        destination.glob("pytorch_model*.bin")
    )
    if (destination / "config.json").is_file() and has_weights:
        print(f"[download skip] {destination}")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_config["repo_id"],
        revision=model_config.get("revision", "main"),
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.model",
            "*.py",
            "*.safetensors",
            "*.tiktoken",
            "*.txt",
        ],
    )
    return destination


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def definitions_from_config(config: dict) -> tuple[list[str], dict[str, str]]:
    labels = labels_from_config(config)
    definitions = json.loads(
        project_path(config["data"]["definitions"]).read_text(encoding="utf-8")
    )
    return labels, {label: definitions[label] for label in labels}


def definition_prompts(config: dict) -> tuple[list[str], list[str]]:
    labels, definitions = definitions_from_config(config)
    template = config["prompt"]["open_vocab_template"]
    return labels, [
        template.format(label=label, definition=definitions[label]) for label in labels
    ]


def bbox_targets(path: Path) -> dict[str, tuple[float, float, float, float]]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in coco["images"]}
    result = {}
    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        result[image["file_name"]] = (
            1000 * x / image["width"],
            1000 * y / image["height"],
            1000 * (x + width) / image["width"],
            1000 * (y + height) / image["height"],
        )
    return result


def manifest_image(data_root: Path, value: str) -> tuple[Path, str]:
    name = value.removeprefix("data/")
    return data_root / name, name


def row_presence(row: dict[str, str], label: str | None, negative_label: str) -> bool:
    if "presence" in row and row["presence"] != "":
        return row["presence"].strip().lower() in {"1", "true", "yes", "positive"}
    return bool(label and label != negative_label)


def no_event_groups(root: Path, gap_seconds: float) -> list[list[Path]]:
    timestamped = []
    untimestamped = []
    for path in sorted(root.glob("*.png")):
        match = NO_EVENT_TIMESTAMP.fullmatch(path.name)
        if match:
            timestamped.append(
                (
                    datetime.strptime(match.group(1), "%Y-%m-%dT%H%M%S.%f"),
                    path,
                )
            )
        else:
            untimestamped.append(path)

    groups = []
    current = []
    previous = None
    for captured_at, path in sorted(timestamped):
        if previous is not None and (captured_at - previous).total_seconds() > gap_seconds:
            groups.append(current)
            current = []
        current.append(path)
        previous = captured_at
    if current:
        groups.append(current)
    if untimestamped:
        groups.append(untimestamped)
    return groups


def no_event_split(config: dict, protocol: str) -> dict[str, list[tuple[Path, str]]]:
    settings = config["data"].get("no_event")
    if not settings:
        return {"train": [], "val": [], "test": []}

    root = project_path(settings["root"])
    groups = no_event_groups(root, settings["group_gap_seconds"])
    ratios = settings["split_ratios"]
    split_names = ("train", "val", "test")
    total = sum(len(group) for group in groups)
    ratio_total = sum(ratios[name] for name in split_names)
    targets = {
        name: total * ratios[name] / ratio_total for name in split_names
    }
    counts = {name: 0 for name in split_names}
    result = {name: [] for name in split_names}
    seed = settings.get("seed", 43)

    def digest(group):
        names = "\n".join(path.name for path in group)
        return hashlib.sha256(f"{seed}:{protocol}:{names}".encode()).digest()

    for group in sorted(groups, key=lambda value: (-len(value), digest(value))):
        split = max(
            split_names,
            key=lambda name: (targets[name] - counts[name]) / targets[name],
        )
        group_names = "\n".join(path.name for path in group)
        group_id = f"neggrp_{hashlib.sha256(group_names.encode()).hexdigest()[:16]}"
        result[split].extend((path, group_id) for path in group)
        counts[split] += len(group)
    return result


def read_no_event_samples(config: dict, protocol: str, split: str) -> list[DiscoverySample]:
    samples = []
    for path, group_id in no_event_split(config, protocol)[split]:
        uid = hashlib.sha256(path.name.encode()).hexdigest()[:20]
        samples.append(
            DiscoverySample(
                record_uid=f"neg_{uid}",
                label=None,
                presence=False,
                image_path=path,
                evidence_path=None,
                bbox_1000=None,
                group_id=group_id,
                negative_subtype="ordinary",
            )
        )
    return samples


def read_discovery_samples(config: dict, protocol: str, split: str) -> list[DiscoverySample]:
    data_root = project_path(config["data"]["root"])
    supported = set(labels_from_config(config))
    negative_label = config["data"].get("negative_label", "no_event")
    if split == "test":
        rows = read_csv(data_root / protocol / "test_inputs.csv")
        private = {
            row["record_uid"]: row for row in read_csv(
                data_root / protocol / "test_labels_private.csv"
            )
        }
    else:
        rows = read_csv(data_root / protocol / f"{split}.csv")
        private = {}
    boxes = bbox_targets(project_path(config["data"]["bbox_annotations"]))
    samples = []
    for row in rows:
        label_row = private.get(row["record_uid"], row)
        label = label_row.get("source_class") or None
        presence = row_presence(label_row | row, label, negative_label)
        if presence and label not in supported:
            continue
        image_path, image_name = manifest_image(data_root, row["context_path"])
        evidence_value = row.get("evidence_path", "")
        evidence_path = manifest_image(data_root, evidence_value)[0] if evidence_value else None
        samples.append(
            DiscoverySample(
                record_uid=row["record_uid"],
                label=label if presence else None,
                presence=presence,
                image_path=image_path,
                evidence_path=evidence_path if presence else None,
                bbox_1000=boxes[image_name] if presence else None,
                group_id=row.get("content_group_id") or None,
                negative_subtype=(
                    (label_row.get("negative_subtype") or None)
                    if not presence
                    else None
                ),
            )
        )
    samples.extend(read_no_event_samples(config, protocol, split))
    maximum = config["data"].get("max_samples")
    return samples[:maximum] if maximum else samples


def discovery_sampler(samples, train_config: dict, seed: int):
    labels = [sample.label or "no_event" for sample in samples]
    counts = Counter(labels)
    power = train_config["class_balance_power"]
    weights = torch.tensor(
        [counts[label] ** (-power) for label in labels], dtype=torch.double
    )
    budget = train_config.get("samples_per_epoch", "positive_records")
    if budget == "positive_records":
        budget = sum(sample.presence for sample in samples)
    elif budget is None:
        budget = len(samples)
    return WeightedRandomSampler(
        weights,
        num_samples=int(budget),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def training_data_state(config: dict, protocol: str) -> dict:
    data_root = project_path(config["data"]["root"])
    manifest = data_root / protocol / "train.csv"
    negatives = no_event_split(config, protocol)["train"]
    fingerprint_rows = [
        (path.name, path.stat().st_size, group_id)
        for path, group_id in negatives
    ]
    state = {
        "version": 1,
        "positive_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "negative_records": len(negatives),
        "negative_fingerprint": hashlib.sha256(
            json.dumps(fingerprint_rows, separators=(",", ":")).encode()
        ).hexdigest(),
        "samples_per_epoch": config["train"].get("samples_per_epoch"),
    }
    if "recipe_version" in config["train"]:
        state["recipe_version"] = config["train"]["recipe_version"]
    return state


def training_data_is_current(output_dir: Path, config: dict, protocol: str) -> bool:
    path = output_dir / "training_data.json"
    return path.is_file() and json.loads(path.read_text()) == training_data_state(
        config, protocol
    )


def save_training_data_state(output_dir: Path, config: dict, protocol: str) -> None:
    (output_dir / "training_data.json").write_text(
        json.dumps(training_data_state(config, protocol), indent=2),
        encoding="utf-8",
    )


def box_iou(first, second) -> float:
    if first is None or second is None:
        return 0.0
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def average_precision(items: list[tuple[float, bool]], positives: int) -> float | None:
    if positives == 0:
        return None
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, correct) in enumerate(
        sorted(items, key=lambda item: item[0], reverse=True), 1
    ):
        if correct:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def presence_ap(samples, predictions) -> float | None:
    positives = sum(sample.presence for sample in samples)
    negatives = len(samples) - positives
    if not positives or not negatives:
        return None
    return average_precision(
        [(prediction["presence_score"], sample.presence) for sample, prediction in zip(samples, predictions)],
        positives,
    )


def localization_ap50(samples, predictions, category: str | None = None) -> float | None:
    if category is None:
        positives = sum(sample.presence for sample in samples)
    else:
        positives = sum(sample.label == category for sample in samples)
    items = []
    for sample, prediction in zip(samples, predictions):
        if prediction["bbox_1000"] is None:
            continue
        correct = sample.presence and box_iou(sample.bbox_1000, prediction["bbox_1000"]) >= 0.5
        if category is not None:
            if prediction["category"] != category:
                continue
            correct = correct and sample.label == category
        items.append((prediction["presence_score"], correct))
    return average_precision(items, positives)


def macro_f1(samples, predictions, labels, threshold: float) -> float:
    values = []
    for label in labels:
        tp = fp = fn = 0
        for sample, prediction in zip(samples, predictions):
            if not sample.presence:
                continue
            predicted = (
                prediction["category"]
                if prediction["presence_score"] >= threshold
                else None
            )
            tp += sample.label == label and predicted == label
            fp += sample.label != label and predicted == label
            fn += sample.label == label and predicted != label
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.fmean(values)


def select_threshold(samples, predictions, fallback: float) -> float:
    positives = sum(sample.presence for sample in samples)
    negatives = len(samples) - positives
    if not positives or not negatives:
        return fallback
    candidates = sorted({prediction["presence_score"] for prediction in predictions})
    best_threshold, best_f1 = fallback, -1.0
    for threshold in candidates:
        tp = sum(s.presence and p["presence_score"] >= threshold for s, p in zip(samples, predictions))
        fp = sum(not s.presence and p["presence_score"] >= threshold for s, p in zip(samples, predictions))
        fn = sum(s.presence and p["presence_score"] < threshold for s, p in zip(samples, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


def table4_metrics(samples, predictions, labels, threshold: float, classification: bool) -> dict:
    negatives = [
        prediction["presence_score"] >= threshold
        for sample, prediction in zip(samples, predictions)
        if not sample.presence
    ]
    per_class_ap = [localization_ap50(samples, predictions, label) for label in labels]
    return {
        "p_ap": presence_ap(samples, predictions),
        "n_fpr": sum(negatives) / len(negatives) if negatives else None,
        "ap50": localization_ap50(samples, predictions),
        "c_f1": macro_f1(samples, predictions, labels, threshold) if classification else None,
        "g_map50": (
            statistics.fmean(value for value in per_class_ap if value is not None)
            if classification else None
        ),
        "valid_rate": sum(prediction["valid"] for prediction in predictions) / len(predictions),
        "median_ms": statistics.median(prediction["latency_ms"] for prediction in predictions),
        "threshold": threshold,
        "positive_records": sum(sample.presence for sample in samples),
        "negative_records": sum(not sample.presence for sample in samples),
    }


def save_results(path: Path, config: dict, protocol: str, samples, predictions, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, prediction in zip(samples, predictions):
        rows.append(
            {
                "record_uid": sample.record_uid,
                "group_id": sample.group_id,
                "negative_subtype": sample.negative_subtype,
                "target": {
                    "presence": sample.presence,
                    "bbox_1000": sample.bbox_1000,
                    "category": sample.label,
                },
                "prediction": prediction,
                "iou": box_iou(sample.bbox_1000, prediction["bbox_1000"]),
            }
        )
    path.write_text(
        json.dumps(
            {
                "experiment": config["experiment"],
                "protocol": protocol,
                "metrics": metrics,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def calibration_path(config: dict, protocol: str, seed: int) -> Path:
    return project_path(config["output"]["calibration"], protocol=protocol, seed=seed)


def save_calibration(config, protocol, seed, samples, predictions, classification=False):
    threshold = select_threshold(samples, predictions, config["validation"]["fallback_threshold"])
    labels = labels_from_config(config)
    metrics = table4_metrics(samples, predictions, labels, threshold, classification)
    path = calibration_path(config, protocol, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"threshold": threshold, "metrics": metrics}, indent=2), encoding="utf-8")
    return threshold, metrics


def read_threshold(config, protocol, seed) -> float:
    return json.loads(calibration_path(config, protocol, seed).read_text())["threshold"]


def ensure_yolo_model(model_config: dict) -> Path:
    destination = project_path(model_config["path"])
    if destination.is_file():
        print(f"[download skip] {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(model_config["url"], destination)
    return destination


def empty_prediction(latency_ms=0.0, raw_output=None, valid=True):
    return {
        "presence_score": 0.0,
        "bbox_1000": None,
        "category": None,
        "valid": valid,
        "latency_ms": latency_ms,
        "raw_output": raw_output,
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class DetectionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class DFineCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        images = []
        labels = []
        for sample in samples:
            with Image.open(sample.image_path) as source:
                images.append(source.convert("RGB"))
            if sample.presence:
                x1, y1, x2, y2 = sample.bbox_1000
                boxes = torch.tensor(
                    [[(x1 + x2) / 2000, (y1 + y2) / 2000, (x2 - x1) / 1000, (y2 - y1) / 1000]],
                    dtype=torch.float32,
                )
                classes = torch.tensor([0], dtype=torch.long)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                classes = torch.empty((0,), dtype=torch.long)
            labels.append({"class_labels": classes, "boxes": boxes})
        batch = self.processor(images=images, return_tensors="pt")
        return dict(batch), labels


def detection_loader(samples, collator, batch_size, workers, shuffle=False, sampler=None):
    return DataLoader(
        DetectionDataset(samples),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def load_dfine(path: Path, device, trained: bool):
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = DFineForObjectDetection.from_pretrained(
        path,
        local_files_only=True,
        num_labels=1,
        id2label={0: "event"},
        label2id={"event": 0},
        ignore_mismatched_sizes=not trained,
    ).to(device)
    return model, processor


@torch.inference_mode()
def predict_dfine(model, processor, samples, device, description):
    model.eval()
    predictions = []
    for sample in tqdm(samples, desc=description, unit="image"):
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        synchronize(device)
        started = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs,
            target_sizes=[(image.height, image.width)],
            threshold=0.0,
        )[0]
        synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        if len(result["scores"]) == 0:
            predictions.append(empty_prediction(latency))
            continue
        index = int(result["scores"].argmax())
        x1, y1, x2, y2 = result["boxes"][index].float().cpu().tolist()
        predictions.append(
            {
                "presence_score": float(result["scores"][index]),
                "bbox_1000": [
                    1000 * x1 / image.width,
                    1000 * y1 / image.height,
                    1000 * x2 / image.width,
                    1000 * y2 / image.height,
                ],
                "category": None,
                "valid": True,
                "latency_ms": latency,
                "raw_output": None,
            }
        )
    return predictions


def train_dfine(config: dict) -> None:
    ensure_hf_model(config["model"])
    device = torch.device(config["runtime"]["device"])
    train_config = config["train"]
    for protocol in config["data"]["protocols"]:
        for seed in train_config["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            output_dir = project_path(config["output"]["root"], **values)
            model_dir = project_path(config["output"]["model"], **values)
            if (
                config["output"].get("skip_existing")
                and (model_dir / "config.json").exists()
                and training_data_is_current(output_dir, config, protocol)
            ):
                print(f"[skip] {model_dir}")
            else:
                seed_everything(seed)
                train_samples = read_discovery_samples(config, protocol, "train")
                model, processor = load_dfine(project_path(config["model"]["path"]), device, False)
                sampler = discovery_sampler(train_samples, train_config, seed)
                train_batches = detection_loader(
                    train_samples,
                    DFineCollator(processor),
                    train_config["batch_size"],
                    train_config["num_workers"],
                    sampler=sampler,
                )
                optimizer = AdamW(
                    model.parameters(),
                    lr=train_config["learning_rate"],
                    weight_decay=train_config["weight_decay"],
                )
                updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
                scheduler = cosine_scheduler(
                    optimizer,
                    updates * train_config["epochs"],
                    train_config["warmup_ratio"],
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(output_dir / "tensorboard")
                global_step = 0
                for epoch in range(1, train_config["epochs"] + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    for batch_index, (batch, labels) in enumerate(
                        tqdm(train_batches, desc=f"{config['experiment']} epoch {epoch}"), 1
                    ):
                        batch = {key: value.to(device) for key, value in batch.items()}
                        labels = [
                            {key: value.to(device) for key, value in row.items()} for row in labels
                        ]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model(**batch, labels=labels).loss
                        (loss / train_config["gradient_accumulation"]).backward()
                        if (
                            batch_index % train_config["gradient_accumulation"] == 0
                            or batch_index == len(train_batches)
                        ):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.flush()
                model.save_pretrained(model_dir)
                processor.save_pretrained(model_dir)
                save_training_data_state(output_dir, config, protocol)
                writer.close()
                del model, optimizer
                torch.cuda.empty_cache()
            model, processor = load_dfine(model_dir, device, True)
            val_samples = read_discovery_samples(config, protocol, "val")
            predictions = predict_dfine(model, processor, val_samples, device, "dfine validation")
            save_calibration(config, protocol, seed, val_samples, predictions)
            del model
            torch.cuda.empty_cache()


def test_dfine(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            model, processor = load_dfine(
                project_path(config["output"]["model"], **values), device, True
            )
            samples = read_discovery_samples(config, protocol, "test")
            predictions = predict_dfine(model, processor, samples, device, "dfine test")
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, False)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)
            del model
            torch.cuda.empty_cache()


def load_grounding_dino(config, device):
    path = project_path(config["model"]["path"])
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        path, local_files_only=True
    ).to(device).eval()
    return model, processor


@torch.inference_mode()
def predict_grounding_dino(model, processor, samples, config, device, description):
    _, prompts = definition_prompts(config)
    chunk_size = config["inference"]["prompt_chunk_size"]
    prompt_chunks = [
        ". ".join(prompt.rstrip(". ") for prompt in prompts[start : start + chunk_size])
        + "."
        for start in range(0, len(prompts), chunk_size)
    ]
    predictions = []
    for sample in tqdm(samples, desc=description, unit="image"):
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        best = None
        synchronize(device)
        started = time.perf_counter()
        for text_prompt in prompt_chunks:
            inputs = processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=config["inference"]["minimum_score"],
                text_threshold=config["inference"]["text_threshold"],
                target_sizes=[(image.height, image.width)],
            )[0]
            if len(result["scores"]):
                index = int(result["scores"].argmax())
                candidate = (float(result["scores"][index]), result["boxes"][index])
                if best is None or candidate[0] > best[0]:
                    best = candidate
        synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        if best is None:
            predictions.append(empty_prediction(latency))
            continue
        score, box = best
        x1, y1, x2, y2 = box.float().cpu().tolist()
        predictions.append(
            {
                "presence_score": score,
                "bbox_1000": [
                    1000 * x1 / image.width,
                    1000 * y1 / image.height,
                    1000 * x2 / image.width,
                    1000 * y2 / image.height,
                ],
                "category": None,
                "valid": True,
                "latency_ms": latency,
                "raw_output": None,
            }
        )
    return predictions


def train_grounding_dino(config: dict) -> None:
    ensure_hf_model(config["model"])
    device = torch.device(config["runtime"]["device"])
    model, processor = load_grounding_dino(config, device)
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            samples = read_discovery_samples(config, protocol, "val")
            predictions = predict_grounding_dino(
                model, processor, samples, config, device, "grounding dino validation"
            )
            save_calibration(config, protocol, seed, samples, predictions)


def test_grounding_dino(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    model, processor = load_grounding_dino(config, device)
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            samples = read_discovery_samples(config, protocol, "test")
            predictions = predict_grounding_dino(
                model, processor, samples, config, device, "grounding dino test"
            )
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, False)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)


def load_yolo_world(config):
    from ultralytics import YOLOWorld

    model = YOLOWorld(str(project_path(config["model"]["path"])))
    _, prompts = definition_prompts(config)
    model.set_classes(prompts)
    return model


def predict_yolo_world(model, samples, config, description):
    predictions = []
    for sample in tqdm(samples, desc=description, unit="image"):
        started = time.perf_counter()
        result = model.predict(
            source=str(sample.image_path),
            imgsz=config["inference"]["image_size"],
            conf=config["inference"]["minimum_score"],
            verbose=False,
        )[0]
        latency = (time.perf_counter() - started) * 1000
        if result.boxes is None or len(result.boxes) == 0:
            predictions.append(empty_prediction(latency))
            continue
        index = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = result.boxes.xyxy[index].float().cpu().tolist()
        height, width = result.orig_shape
        predictions.append(
            {
                "presence_score": float(result.boxes.conf[index]),
                "bbox_1000": [
                    1000 * x1 / width,
                    1000 * y1 / height,
                    1000 * x2 / width,
                    1000 * y2 / height,
                ],
                "category": None,
                "valid": True,
                "latency_ms": latency,
                "raw_output": None,
            }
        )
    return predictions


def train_yolo_world(config: dict) -> None:
    ensure_yolo_model(config["model"])
    model = load_yolo_world(config)
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            samples = read_discovery_samples(config, protocol, "val")
            predictions = predict_yolo_world(model, samples, config, "yolo-world validation")
            save_calibration(config, protocol, seed, samples, predictions)


def test_yolo_world(config: dict) -> None:
    model = load_yolo_world(config)
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            samples = read_discovery_samples(config, protocol, "test")
            predictions = predict_yolo_world(model, samples, config, "yolo-world test")
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, False)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)


def category_block(config: dict) -> str:
    labels, definitions = definitions_from_config(config)
    return "\n".join(f"- {label}: {definitions[label]}" for label in labels)


def add_qwen_location_tokens(model, processor, count: int = 1000) -> list[int]:
    tokens = location_token_strings(count)
    processor.tokenizer.add_tokens(tokens, special_tokens=True)
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    return processor.tokenizer.convert_tokens_to_ids(tokens)


def native_location_token_ids(tokenizer, count: int = 1000) -> list[int]:
    tokens = location_token_strings(count)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(set(ids)) != count or tokenizer.unk_token_id in ids:
        raise ValueError("tokenizer does not provide 1000 native <loc_*> tokens")
    return ids


def quantized_location_tokens(bbox_1000, count: int = 1000) -> list[str]:
    return [
        f"<loc_{max(0, min(count - 1, round(value * (count - 1) / 1000)))}>"
        for value in bbox_1000
    ]


def grounding_target(sample: DiscoverySample, count: int = 1000) -> str:
    if not sample.presence:
        return "no_event"
    return sample.label + "".join(quantized_location_tokens(sample.bbox_1000, count))


def parse_grounding_tokens(
    token_ids,
    tokenizer,
    labels,
    location_token_ids,
    negative_label: str = "no_event",
):
    ignored = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    ids = [int(token) for token in token_ids if int(token) not in ignored]
    negative = tokenizer(negative_label, add_special_tokens=False)["input_ids"]
    if ids == negative:
        return {"presence_score": 0.0, "bbox_1000": None, "category": None}

    loc_to_index = {token: index for index, token in enumerate(location_token_ids)}
    for label in labels:
        prefix = tokenizer(label, add_special_tokens=False)["input_ids"]
        coordinates = ids[len(prefix) :]
        if ids[: len(prefix)] != prefix or len(coordinates) != 4:
            continue
        if not all(token in loc_to_index for token in coordinates):
            continue
        bbox = [1000 * loc_to_index[token] / (len(location_token_ids) - 1) for token in coordinates]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("invalid generated box ordering")
        return {"presence_score": 1.0, "bbox_1000": bbox, "category": label}
    raise ValueError("output does not match grounding token grammar")


def event_probability(first_scores, tokenizer, labels, negative_label="no_event") -> float:
    probabilities = first_scores.float().softmax(dim=-1)
    event_tokens = {
        tokenizer(label, add_special_tokens=False)["input_ids"][0] for label in labels
    }
    negative_token = tokenizer(negative_label, add_special_tokens=False)["input_ids"][0]
    event_score = probabilities[list(event_tokens)].sum()
    negative_score = probabilities[negative_token]
    return float(event_score / (event_score + negative_score).clamp_min(1e-12))


class VlmDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class QwenDiscoveryCollator:
    def __init__(self, processor, config, training):
        self.processor = processor
        self.config = config
        self.training = training
        self.location_count = config["generation"]["location_tokens"]
        self.prompt = config["prompt"]["user"].replace("{categories}", category_block(config))

    def messages(self, sample, answer=None, image=None, prompt=None):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config["prompt"]["system"]}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image or str(sample.image_path)},
                    {"type": "text", "text": prompt or self.prompt},
                ],
            },
        ]
        if answer is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return messages

    def __call__(self, samples):
        conversations = [
            self.messages(
                sample,
                grounding_target(sample, self.location_count) if self.training else None,
            )
            for sample in samples
        ]
        size = {
            "longest_edge": self.config["input"]["max_pixels"],
            "shortest_edge": self.config["input"]["min_pixels"],
        }
        encoded = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=not self.training,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "size": size},
        )
        if self.training:
            encoded["labels"] = assistant_only_labels(
                encoded["input_ids"],
                encoded["attention_mask"],
                self.processor.tokenizer,
            )
        return dict(encoded)


def vlm_loader(samples, collator, batch_size, workers, sampler=None):
    return DataLoader(
        VlmDataset(samples),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def load_qwen_adapter(config, protocol, seed, device):
    model, processor = load_qwen(project_path(config["model"]["path"]))
    add_qwen_location_tokens(
        model, processor, config["generation"]["location_tokens"]
    )
    adapter = project_path(config["output"]["adapter"], protocol=protocol, seed=seed)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).to(device).eval()
    return model, processor


@torch.inference_mode()
def predict_qwen_discovery(model, processor, samples, config, device, description):
    collator = QwenDiscoveryCollator(processor, config, False)
    labels = labels_from_config(config)
    location_ids = processor.tokenizer.convert_tokens_to_ids(
        location_token_strings(config["generation"]["location_tokens"])
    )
    predictions = []
    for sample in tqdm(samples, desc=description, unit="image"):
        synchronize(device)
        started = time.perf_counter()
        batch = collator([sample])
        batch = {key: value.to(device) for key, value in batch.items()}
        input_length = batch["input_ids"].shape[1]
        constraint = grounding_prefix_allowed_tokens(
            processor.tokenizer,
            labels,
            location_token_ids=location_ids,
            prompt_length=input_length,
        )
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=config["generation"]["max_new_tokens"],
            prefix_allowed_tokens_fn=(
                constraint
                if config["generation"].get("constrained_grammar", True)
                else None
            ),
            return_dict_in_generate=True,
            output_scores=True,
        )
        synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        output_ids = generated.sequences[0, input_length:]
        raw = processor.decode(output_ids, skip_special_tokens=False).strip()
        try:
            parsed = parse_grounding_tokens(
                output_ids,
                processor.tokenizer,
                labels,
                location_ids,
            )
            parsed["presence_score"] = event_probability(
                generated.scores[0][0], processor.tokenizer, labels
            )
            predictions.append(parsed | {"valid": True, "latency_ms": latency, "raw_output": raw})
        except (KeyError, TypeError, ValueError):
            predictions.append(empty_prediction(latency, raw, False))
    return predictions


def train_qwen_discovery(config: dict) -> None:
    ensure_hf_model(config["model"])
    device = torch.device(config["runtime"]["device"])
    train_config = config["train"]
    for protocol in config["data"]["protocols"]:
        for seed in train_config["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            output_dir = project_path(config["output"]["root"], **values)
            adapter_dir = project_path(config["output"]["adapter"], **values)
            if (
                config["output"].get("skip_existing")
                and (adapter_dir / "adapter_config.json").exists()
                and training_data_is_current(output_dir, config, protocol)
            ):
                print(f"[skip] {adapter_dir}")
            else:
                seed_everything(seed)
                train_samples = read_discovery_samples(config, protocol, "train")
                model, processor = load_qwen(project_path(config["model"]["path"]))
                location_ids = add_qwen_location_tokens(
                    model,
                    processor,
                    config["generation"]["location_tokens"],
                )
                model.config.use_cache = False
                model.enable_input_require_grads()
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=train_config["lora_r"],
                        lora_alpha=train_config["lora_alpha"],
                        lora_dropout=train_config["lora_dropout"],
                        target_modules=LORA_PATTERNS[train_config["lora_scope"]],
                        bias="none",
                        task_type="CAUSAL_LM",
                        trainable_token_indices={
                            "model.language_model.embed_tokens": location_ids,
                            "lm_head": location_ids,
                        },
                    ),
                ).to(device)
                model.gradient_checkpointing_enable()
                sampler = discovery_sampler(train_samples, train_config, seed)
                train_batches = vlm_loader(
                    train_samples,
                    QwenDiscoveryCollator(processor, config, True),
                    train_config["batch_size"],
                    train_config["num_workers"],
                    sampler,
                )
                optimizer = AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    lr=train_config["learning_rate"],
                    weight_decay=train_config["weight_decay"],
                )
                updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
                scheduler = cosine_scheduler(
                    optimizer,
                    updates * train_config["epochs"],
                    train_config["warmup_ratio"],
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(output_dir / "tensorboard")
                global_step = 0
                for epoch in range(1, train_config["epochs"] + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    progress = tqdm(
                        train_batches,
                        desc=f"{config['experiment']} epoch {epoch}",
                    )
                    for batch_index, batch in enumerate(
                        progress, 1
                    ):
                        batch = {key: value.to(device) for key, value in batch.items()}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model(**batch).loss
                        running_loss += float(loss.detach())
                        (loss / train_config["gradient_accumulation"]).backward()
                        if (
                            batch_index % train_config["gradient_accumulation"] == 0
                            or batch_index == len(train_batches)
                        ):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            writer.add_scalar("train/loss", loss.item(), global_step)
                        progress.set_postfix(
                            loss=f"{float(loss.detach()):.4f}",
                            avg=f"{running_loss / batch_index:.4f}",
                        )
                    writer.flush()
                model.save_pretrained(adapter_dir)
                processor.save_pretrained(adapter_dir)
                save_training_data_state(output_dir, config, protocol)
                writer.close()
                del model, optimizer
                torch.cuda.empty_cache()
            model, processor = load_qwen_adapter(config, protocol, seed, device)
            val_samples = read_discovery_samples(config, protocol, "val")
            predictions = predict_qwen_discovery(
                model, processor, val_samples, config, device, "qwen validation"
            )
            save_calibration(config, protocol, seed, val_samples, predictions, True)
            del model
            torch.cuda.empty_cache()


def test_qwen_discovery(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            model, processor = load_qwen_adapter(config, protocol, seed, device)
            samples = read_discovery_samples(config, protocol, "test")
            predictions = predict_qwen_discovery(
                model, processor, samples, config, device, "qwen test"
            )
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, True)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)
            del model
            torch.cuda.empty_cache()


class FlorenceDiscoveryCollator:
    def __init__(self, processor, config, training):
        self.processor = processor
        self.training = training
        self.location_count = config["generation"]["location_tokens"]
        labels, definitions = definitions_from_config(config)
        template = config["prompt"]["open_vocab_template"]
        categories = "; ".join(
            template.format(label=label, definition=definitions[label]).rstrip(".")
            for label in labels
        )
        self.prompt = config["prompt"]["user"].replace("{categories}", categories)

    def __call__(self, samples):
        images = []
        for sample in samples:
            with Image.open(sample.image_path) as source:
                images.append(source.convert("RGB"))
        batch = self.processor(
            text=[self.prompt] * len(samples), images=images, padding=True, return_tensors="pt"
        )
        if batch["input_ids"].shape[1] > self.processor.tokenizer.model_max_length:
            raise ValueError(
                f"Florence encoder input has {batch['input_ids'].shape[1]} tokens, "
                f"but the model limit is {self.processor.tokenizer.model_max_length}"
            )
        if self.training:
            tokenizer = self.processor.tokenizer
            targets = [
                torch.tensor(
                    tokenizer(
                        grounding_target(sample, self.location_count),
                        add_special_tokens=False,
                    )["input_ids"]
                    + [tokenizer.eos_token_id],
                    dtype=torch.long,
                )
                for sample in samples
            ]
            batch["labels"] = nn.utils.rnn.pad_sequence(
                targets, batch_first=True, padding_value=-100
            )
        return dict(batch)


def load_florence(path: Path, device, training: bool = False):
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.float32 if training else torch.bfloat16,
    ).to(device)
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    model.generation_config.num_beams = 1
    model.generation_config.early_stopping = False
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.validate(strict=True)
    return model, processor


@torch.inference_mode()
def predict_florence(model, processor, samples, config, device, description):
    collator = FlorenceDiscoveryCollator(processor, config, False)
    labels = labels_from_config(config)
    location_ids = native_location_token_ids(
        processor.tokenizer, config["generation"]["location_tokens"]
    )
    predictions = []
    for sample in tqdm(samples, desc=description, unit="image"):
        synchronize(device)
        started = time.perf_counter()
        batch = collator([sample])
        batch = {key: value.to(device) for key, value in batch.items()}
        constraint = grounding_prefix_allowed_tokens(
            processor.tokenizer,
            labels,
            location_token_ids=location_ids,
            decoder_start_token_id=model.config.text_config.decoder_start_token_id,
        )
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=config["generation"]["max_new_tokens"],
            prefix_allowed_tokens_fn=(
                constraint
                if config["generation"].get("constrained_grammar", True)
                else None
            ),
            return_dict_in_generate=True,
            output_scores=True,
        )
        synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        output_ids = generated.sequences[0]
        raw = processor.decode(output_ids, skip_special_tokens=False).strip()
        try:
            parsed = parse_grounding_tokens(
                output_ids,
                processor.tokenizer,
                labels,
                location_ids,
            )
            parsed["presence_score"] = event_probability(
                generated.scores[0][0], processor.tokenizer, labels
            )
            predictions.append(parsed | {"valid": True, "latency_ms": latency, "raw_output": raw})
        except (KeyError, TypeError, ValueError):
            predictions.append(empty_prediction(latency, raw, False))
    return predictions


def train_florence_discovery(config: dict) -> None:
    ensure_hf_model(config["model"])
    device = torch.device(config["runtime"]["device"])
    train_config = config["train"]
    for protocol in config["data"]["protocols"]:
        for seed in train_config["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            output_dir = project_path(config["output"]["root"], **values)
            model_dir = project_path(config["output"]["model"], **values)
            if (
                config["output"].get("skip_existing")
                and (model_dir / "config.json").exists()
                and training_data_is_current(output_dir, config, protocol)
            ):
                print(f"[skip] {model_dir}")
            else:
                seed_everything(seed)
                train_samples = read_discovery_samples(config, protocol, "train")
                model, processor = load_florence(
                    project_path(config["model"]["path"]), device, training=True
                )
                sampler = discovery_sampler(train_samples, train_config, seed)
                train_batches = vlm_loader(
                    train_samples,
                    FlorenceDiscoveryCollator(processor, config, True),
                    train_config["batch_size"],
                    train_config["num_workers"],
                    sampler,
                )
                optimizer = AdamW(
                    model.parameters(),
                    lr=train_config["learning_rate"],
                    weight_decay=train_config["weight_decay"],
                )
                updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
                scheduler = cosine_scheduler(
                    optimizer,
                    updates * train_config["epochs"],
                    train_config["warmup_ratio"],
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(
                    output_dir
                    / "tensorboard"
                    / f"recipe_v{train_config['recipe_version']}"
                )
                global_step = 0
                for epoch in range(1, train_config["epochs"] + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    gradient_norm = 0.0
                    progress = tqdm(
                        train_batches,
                        desc=(
                            f"{config['experiment']} {protocol} "
                            f"epoch {epoch}/{train_config['epochs']}"
                        ),
                        unit="batch",
                    )
                    for batch_index, batch in enumerate(progress, 1):
                        batch = {key: value.to(device) for key, value in batch.items()}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model(**batch).loss
                        loss_value = float(loss.detach())
                        if not math.isfinite(loss_value):
                            raise FloatingPointError(
                                f"non-finite Florence loss at {protocol} epoch {epoch} "
                                f"batch {batch_index}: {loss_value}"
                            )
                        (loss / train_config["gradient_accumulation"]).backward()
                        running_loss += loss_value
                        if (
                            batch_index % train_config["gradient_accumulation"] == 0
                            or batch_index == len(train_batches)
                        ):
                            gradient_norm = float(
                                nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    train_config["max_grad_norm"],
                                    error_if_nonfinite=True,
                                )
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            writer.add_scalar("train/loss", loss_value, global_step)
                            writer.add_scalar(
                                "train/average_loss",
                                running_loss / batch_index,
                                global_step,
                            )
                            writer.add_scalar(
                                "train/gradient_norm", gradient_norm, global_step
                            )
                            writer.add_scalar(
                                "train/learning_rate",
                                scheduler.get_last_lr()[0],
                                global_step,
                            )
                        progress.set_postfix(
                            loss=f"{loss_value:.4f}",
                            avg=f"{running_loss / batch_index:.4f}",
                            grad=f"{gradient_norm:.3f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )
                    writer.flush()
                    print(
                        f"{config['experiment']} {protocol} epoch {epoch}: "
                        f"loss={running_loss / len(train_batches):.4f}"
                    )
                model.save_pretrained(model_dir)
                processor.save_pretrained(model_dir)
                save_training_data_state(output_dir, config, protocol)
                writer.close()
                del model, optimizer
                torch.cuda.empty_cache()
            model, processor = load_florence(model_dir, device)
            val_samples = read_discovery_samples(config, protocol, "val")
            predictions = predict_florence(
                model, processor, val_samples, config, device, "florence validation"
            )
            save_calibration(config, protocol, seed, val_samples, predictions, True)
            del model
            torch.cuda.empty_cache()


def test_florence_discovery(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            model, processor = load_florence(
                project_path(config["output"]["model"], **values), device
            )
            samples = read_discovery_samples(config, protocol, "test")
            predictions = predict_florence(
                model, processor, samples, config, device, "florence test"
            )
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, True)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)
            del model
            torch.cuda.empty_cache()


class CropClassifierDataset(Dataset):
    def __init__(self, samples, processor, labels):
        self.samples = [sample for sample in samples if sample.presence]
        self.processor = processor
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.evidence_path) as source:
            pixels = self.processor(images=source.convert("RGB"), return_tensors="pt")
        return pixels["pixel_values"][0], self.label_to_index[sample.label]


def classifier_loader(samples, processor, labels, batch_size, workers, sampler=None):
    return DataLoader(
        CropClassifierDataset(samples, processor, labels),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train_dinov2_classifier(config, protocol, seed):
    ensure_hf_model(config["classifier_model"])
    device = torch.device(config["runtime"]["device"])
    train_config = config["classifier_train"]
    labels = labels_from_config(config)
    output = project_path(config["output"]["classifier"], protocol=protocol, seed=seed)
    if config["output"].get("skip_existing") and output.exists():
        print(f"[skip] {output}")
        return
    seed_everything(seed)
    train_samples = read_discovery_samples(config, protocol, "train")
    path = project_path(config["classifier_model"]["path"])
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    backbone = AutoModel.from_pretrained(path, local_files_only=True)
    model = EncoderClassifier(backbone, encoder_hidden_dim(backbone), len(labels)).to(device)
    positive_train = [sample for sample in train_samples if sample.presence]
    sampler = class_sampler(
        [sample.label for sample in positive_train], train_config["class_balance_power"], seed
    )
    train_batches = classifier_loader(
        positive_train,
        processor,
        labels,
        train_config["batch_size"],
        train_config["num_workers"],
        sampler,
    )
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": train_config["backbone_learning_rate"]},
            {"params": model.classifier.parameters(), "lr": train_config["learning_rate"]},
        ],
        weight_decay=train_config["weight_decay"],
    )
    updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
    scheduler = cosine_scheduler(
        optimizer, updates * train_config["epochs"], train_config["warmup_ratio"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (pixels, targets) in enumerate(
            tqdm(train_batches, desc=f"dinov2 classifier epoch {epoch}"), 1
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(pixels.to(device)), targets.to(device))
            (loss / train_config["gradient_accumulation"]).backward()
            if (
                batch_index % train_config["gradient_accumulation"] == 0
                or batch_index == len(train_batches)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "labels": labels,
        },
        output,
    )
    del model, optimizer
    torch.cuda.empty_cache()


def load_dinov2_classifier(config, protocol, seed, device):
    labels = labels_from_config(config)
    path = project_path(config["classifier_model"]["path"])
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    backbone = AutoModel.from_pretrained(path, local_files_only=True)
    model = EncoderClassifier(backbone, encoder_hidden_dim(backbone), len(labels)).to(device)
    checkpoint = torch.load(
        project_path(config["output"]["classifier"], protocol=protocol, seed=seed),
        map_location="cpu",
        weights_only=False,
    )
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    return model.eval(), processor


def crop_from_box(image: Image.Image, bbox, margin=0.0) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 * image.width / 1000, x2 * image.width / 1000
    y1, y2 = y1 * image.height / 1000, y2 * image.height / 1000
    extra_x, extra_y = (x2 - x1) * margin, (y2 - y1) * margin
    return image.crop(
        (
            max(0, round(x1 - extra_x)),
            max(0, round(y1 - extra_y)),
            min(image.width, round(x2 + extra_x)),
            min(image.height, round(y2 + extra_y)),
        )
    )


@torch.inference_mode()
def classify_dinov2_crops(model, processor, samples, localizations, config, device):
    labels = labels_from_config(config)
    predictions = []
    margin = config["cascade"]["context_margin"]
    for sample, localization in tqdm(
        zip(samples, localizations), total=len(samples), desc="dinov2 cascade", unit="image"
    ):
        if localization["bbox_1000"] is None:
            predictions.append(localization)
            continue
        synchronize(device)
        started = time.perf_counter()
        with Image.open(sample.image_path) as source:
            crop = crop_from_box(source.convert("RGB"), localization["bbox_1000"], margin)
        pixels = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            index = int(model(pixels).argmax(-1)[0])
        synchronize(device)
        latency = localization["latency_ms"] + (time.perf_counter() - started) * 1000
        predictions.append(localization | {"category": labels[index], "latency_ms": latency})
    return predictions


def train_dfine_dinov2(config: dict) -> None:
    train_dfine(config)
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            train_dinov2_classifier(config, protocol, seed)
            values = {"protocol": protocol, "seed": seed}
            locator, locator_processor = load_dfine(
                project_path(config["output"]["model"], **values), device, True
            )
            classifier, classifier_processor = load_dinov2_classifier(
                config, protocol, seed, device
            )
            samples = read_discovery_samples(config, protocol, "val")
            locations = predict_dfine(
                locator, locator_processor, samples, device, "cascade validation localization"
            )
            predictions = classify_dinov2_crops(
                classifier, classifier_processor, samples, locations, config, device
            )
            save_calibration(config, protocol, seed, samples, predictions, True)
            del locator, classifier
            torch.cuda.empty_cache()


def test_dfine_dinov2(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            locator, locator_processor = load_dfine(
                project_path(config["output"]["model"], **values), device, True
            )
            classifier, classifier_processor = load_dinov2_classifier(
                config, protocol, seed, device
            )
            samples = read_discovery_samples(config, protocol, "test")
            locations = predict_dfine(
                locator, locator_processor, samples, device, "cascade test localization"
            )
            predictions = classify_dinov2_crops(
                classifier, classifier_processor, samples, locations, config, device
            )
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, True)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)
            del locator, classifier
            torch.cuda.empty_cache()


class QwenCropCollator:
    def __init__(self, processor, config, training):
        self.processor = processor
        self.config = config
        self.training = training
        self.labels = labels_from_config(config)
        self.prompt = config["classifier_prompt"]["user"].replace(
            "{categories}", category_block(config)
        )

    def messages(self, image, answer=None):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.config["classifier_prompt"]["system"]}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        if answer is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return messages

    def __call__(self, samples):
        conversations = [
            self.messages(str(sample.evidence_path), sample.label if self.training else None)
            for sample in samples
        ]
        size = {
            "longest_edge": self.config["classifier_input"]["max_pixels"],
            "shortest_edge": self.config["classifier_input"]["min_pixels"],
        }
        encoded = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=not self.training,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "size": size},
        )
        if self.training:
            encoded["labels"] = assistant_only_labels(
                encoded["input_ids"],
                encoded["attention_mask"],
                self.processor.tokenizer,
            )
        return dict(encoded)


def train_qwen_crop_classifier(config, protocol, seed):
    ensure_hf_model(config["classifier_model"])
    adapter_dir = project_path(
        config["output"]["classifier_adapter"], protocol=protocol, seed=seed
    )
    if config["output"].get("skip_existing") and (adapter_dir / "adapter_config.json").exists():
        print(f"[skip] {adapter_dir}")
        return
    device = torch.device(config["runtime"]["device"])
    train_config = config["classifier_train"]
    seed_everything(seed)
    train_samples = [
        sample for sample in read_discovery_samples(config, protocol, "train") if sample.presence
    ]
    model, processor = load_qwen(project_path(config["classifier_model"]["path"]))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=train_config["lora_r"],
            lora_alpha=train_config["lora_alpha"],
            lora_dropout=train_config["lora_dropout"],
            target_modules=LORA_PATTERNS[train_config["lora_scope"]],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    model.gradient_checkpointing_enable()
    sampler = class_sampler(
        [sample.label for sample in train_samples], train_config["class_balance_power"], seed
    )
    train_batches = vlm_loader(
        train_samples,
        QwenCropCollator(processor, config, True),
        train_config["batch_size"],
        train_config["num_workers"],
        sampler,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
    scheduler = cosine_scheduler(
        optimizer, updates * train_config["epochs"], train_config["warmup_ratio"]
    )
    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(
            tqdm(train_batches, desc=f"qwen crop classifier epoch {epoch}"), 1
        ):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / train_config["gradient_accumulation"]).backward()
            if (
                batch_index % train_config["gradient_accumulation"] == 0
                or batch_index == len(train_batches)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    del model, optimizer
    torch.cuda.empty_cache()


def load_qwen_crop_classifier(config, protocol, seed, device):
    model, processor = load_qwen(project_path(config["classifier_model"]["path"]))
    adapter = project_path(
        config["output"]["classifier_adapter"], protocol=protocol, seed=seed
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).to(device).eval()
    return model, processor


@torch.inference_mode()
def classify_qwen_crops(model, processor, samples, localizations, config, device):
    collator = QwenCropCollator(processor, config, False)
    labels = set(labels_from_config(config))
    predictions = []
    margin = config["cascade"]["context_margin"]
    for sample, localization in tqdm(
        zip(samples, localizations), total=len(samples), desc="qwen cascade", unit="image"
    ):
        if localization["bbox_1000"] is None:
            predictions.append(localization)
            continue
        synchronize(device)
        started = time.perf_counter()
        with Image.open(sample.image_path) as source:
            crop = crop_from_box(source.convert("RGB"), localization["bbox_1000"], margin)
        messages = collator.messages(crop)
        size = {
            "longest_edge": config["classifier_input"]["max_pixels"],
            "shortest_edge": config["classifier_input"]["min_pixels"],
        }
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"size": size},
        ).to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=config["classifier_generation"]["max_new_tokens"],
            prefix_allowed_tokens_fn=(
                label_prefix_allowed_tokens(
                    processor.tokenizer,
                    labels,
                    prompt_length=inputs["input_ids"].shape[1],
                )
                if config["classifier_generation"].get(
                    "constrained_decoding", False
                )
                else None
            ),
        )
        synchronize(device)
        latency = localization["latency_ms"] + (time.perf_counter() - started) * 1000
        raw = processor.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        category = raw if raw in labels else None
        predictions.append(
            localization
            | {
                "category": category,
                "valid": localization["valid"] and category is not None,
                "latency_ms": latency,
                "raw_output": {"localizer": localization["raw_output"], "classifier": raw},
            }
        )
    return predictions


def selection_path(config, protocol, seed):
    return project_path(config["output"]["selection"], protocol=protocol, seed=seed)


def select_best_localizer(config, protocol, seed):
    rows = []
    for candidate in config["selection"]["candidates"]:
        candidate_config = load_yaml(project_path(candidate["config"]))
        candidate_seed = candidate_config["train"]["seeds"][0]
        calibration = json.loads(
            calibration_path(candidate_config, protocol, candidate_seed).read_text()
        )
        rows.append(
            {
                "name": candidate["name"],
                "config": candidate["config"],
                "validation_ap50": calibration["metrics"]["ap50"],
            }
        )
    selected = max(rows, key=lambda row: row["validation_ap50"])
    path = selection_path(config, protocol, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"selected": selected, "candidates": rows}, indent=2))
    return selected


def read_selection(config, protocol, seed):
    return json.loads(selection_path(config, protocol, seed).read_text())["selected"]


def predict_selected_localizer(selected, protocol, samples, device):
    candidate = load_yaml(project_path(selected["config"]))
    seed = candidate["train"]["seeds"][0]
    values = {"protocol": protocol, "seed": seed}
    family = candidate["model"]["family"]
    if family == "dfine":
        model, processor = load_dfine(
            project_path(candidate["output"]["model"], **values), device, True
        )
        predictions = predict_dfine(model, processor, samples, device, "best localizer dfine")
    elif family == "grounding_dino":
        model, processor = load_grounding_dino(candidate, device)
        predictions = predict_grounding_dino(
            model, processor, samples, candidate, device, "best localizer grounding dino"
        )
    elif family == "yolo_world":
        model = load_yolo_world(candidate)
        predictions = predict_yolo_world(model, samples, candidate, "best localizer yolo-world")
    else:
        raise ValueError(f"unknown localizer family: {family}")
    del model
    torch.cuda.empty_cache()
    return predictions


def train_best_localizer_qwen(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            selected = select_best_localizer(config, protocol, seed)
            train_qwen_crop_classifier(config, protocol, seed)
            classifier, processor = load_qwen_crop_classifier(config, protocol, seed, device)
            samples = read_discovery_samples(config, protocol, "val")
            locations = predict_selected_localizer(selected, protocol, samples, device)
            predictions = classify_qwen_crops(
                classifier, processor, samples, locations, config, device
            )
            save_calibration(config, protocol, seed, samples, predictions, True)
            del classifier
            torch.cuda.empty_cache()


def test_best_localizer_qwen(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            selected = read_selection(config, protocol, seed)
            classifier, processor = load_qwen_crop_classifier(config, protocol, seed, device)
            samples = read_discovery_samples(config, protocol, "test")
            locations = predict_selected_localizer(selected, protocol, samples, device)
            predictions = classify_qwen_crops(
                classifier, processor, samples, locations, config, device
            )
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, True)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, selected, metrics)
            del classifier
            torch.cuda.empty_cache()


@torch.inference_mode()
def qwen_discovery_call(model, processor, image, config, device):
    prompt = config["prompt"]["user"].replace("{categories}", category_block(config))
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": config["prompt"]["system"]}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    synchronize(device)
    started = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "size": {
                "longest_edge": config["input"]["max_pixels"],
                "shortest_edge": config["input"]["min_pixels"],
            }
        },
    ).to(device)
    input_length = inputs["input_ids"].shape[1]
    labels = labels_from_config(config)
    location_ids = processor.tokenizer.convert_tokens_to_ids(
        location_token_strings(config["generation"]["location_tokens"])
    )
    constraint = grounding_prefix_allowed_tokens(
        processor.tokenizer,
        labels,
        location_token_ids=location_ids,
        prompt_length=input_length,
    )
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=config["generation"]["max_new_tokens"],
        prefix_allowed_tokens_fn=(
            constraint
            if config["generation"].get("constrained_grammar", True)
            else None
        ),
        return_dict_in_generate=True,
        output_scores=True,
    )
    synchronize(device)
    latency = (time.perf_counter() - started) * 1000
    output_ids = generated.sequences[0, input_length:]
    raw = processor.decode(output_ids, skip_special_tokens=False).strip()
    try:
        parsed = parse_grounding_tokens(
            output_ids, processor.tokenizer, labels, location_ids
        )
        parsed["presence_score"] = event_probability(
            generated.scores[0][0], processor.tokenizer, labels
        )
        return parsed | {"valid": True, "latency_ms": latency, "raw_output": raw}
    except (KeyError, TypeError, ValueError):
        return empty_prediction(latency, raw, False)


def sliding_tiles(image: Image.Image, grid: int, overlap: float):
    tile_width = image.width / (grid - overlap * (grid - 1))
    tile_height = image.height / (grid - overlap * (grid - 1))
    step_x, step_y = tile_width * (1 - overlap), tile_height * (1 - overlap)
    tiles = []
    for row in range(grid):
        for column in range(grid):
            left = round(min(column * step_x, image.width - tile_width))
            top = round(min(row * step_y, image.height - tile_height))
            right = round(min(image.width, left + tile_width))
            bottom = round(min(image.height, top + tile_height))
            tiles.append((image.crop((left, top, right, bottom)), (left, top, right, bottom)))
    return tiles


def tile_box_to_global(bbox, tile_region, image_size):
    left, top, right, bottom = tile_region
    width, height = image_size
    return [
        1000 * (left + bbox[0] * (right - left) / 1000) / width,
        1000 * (top + bbox[1] * (bottom - top) / 1000) / height,
        1000 * (left + bbox[2] * (right - left) / 1000) / width,
        1000 * (top + bbox[3] * (bottom - top) / 1000) / height,
    ]


@torch.inference_mode()
def predict_qwen_agent(model, processor, samples, config, device, description):
    predictions = []
    for sample in tqdm(samples, desc=description, unit="image"):
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        calls = []
        global_prediction = qwen_discovery_call(model, processor, image, config, device)
        calls.append({"view": "global", "prediction": global_prediction})
        candidates = [global_prediction]
        if global_prediction["presence_score"] < config["agent"]["early_exit_threshold"]:
            for tile, region in sliding_tiles(
                image, config["agent"]["tile_grid"], config["agent"]["overlap"]
            ):
                tile_prediction = qwen_discovery_call(model, processor, tile, config, device)
                if tile_prediction["bbox_1000"] is not None:
                    tile_prediction["bbox_1000"] = tile_box_to_global(
                        tile_prediction["bbox_1000"], region, image.size
                    )
                calls.append(
                    {"view": "tile", "region_pixels": region, "prediction": tile_prediction}
                )
                candidates.append(tile_prediction)
        best = max(candidates, key=lambda prediction: prediction["presence_score"])
        total_latency = sum(call["prediction"]["latency_ms"] for call in calls)
        if best["bbox_1000"] is not None:
            inspect_crop = crop_from_box(
                image, best["bbox_1000"], config["agent"]["inspect_context_margin"]
            )
            inspected = qwen_discovery_call(model, processor, inspect_crop, config, device)
            calls.append({"view": "source_pixel_inspection", "prediction": inspected})
            total_latency += inspected["latency_ms"]
            category = inspected["category"]
            presence_score = best["presence_score"] * inspected["presence_score"]
            valid = best["valid"] and inspected["valid"]
        else:
            category = None
            presence_score = best["presence_score"]
            valid = best["valid"]
        predictions.append(
            best
            | {
                "presence_score": presence_score,
                "category": category,
                "valid": valid,
                "latency_ms": total_latency,
                "raw_output": calls,
            }
        )
    return predictions


def train_qwen_agent(config: dict) -> None:
    train_qwen_discovery(config)
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            model, processor = load_qwen_adapter(config, protocol, seed, device)
            samples = read_discovery_samples(config, protocol, "val")
            predictions = predict_qwen_agent(
                model, processor, samples, config, device, "qwen agent validation"
            )
            save_calibration(config, protocol, seed, samples, predictions, True)
            del model
            torch.cuda.empty_cache()


def test_qwen_agent(config: dict) -> None:
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            model, processor = load_qwen_adapter(config, protocol, seed, device)
            samples = read_discovery_samples(config, protocol, "test")
            predictions = predict_qwen_agent(
                model, processor, samples, config, device, "qwen agent test"
            )
            threshold = read_threshold(config, protocol, seed)
            metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, True)
            save_results(
                project_path(config["output"]["results"], **values),
                config,
                protocol,
                samples,
                predictions,
                metrics,
            )
            print(protocol, seed, metrics)
            del model
            torch.cuda.empty_cache()
