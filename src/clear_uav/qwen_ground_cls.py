from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.ops import roi_align
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from clear_uav.experiment_config import load_yaml_with_base, project_path
from clear_uav.ontology import load_ontology
from clear_uav.qwen_ground_ms import GroundCollator, QwenGroundMS, cxcywh_to_xyxy


NO_EVENT_TIMESTAMP = re.compile(
    r"photo - (\d{4}-\d{2}-\d{2}T\d{6}\.\d+)\.png$"
)


def load_class_labels(ontology_path: Path, labels_path: Path | None = None) -> list[str]:
    if labels_path is not None:
        return [
            line.strip()
            for line in labels_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return list(load_ontology(ontology_path).labels)


@torch.inference_mode()
def load_definition_prototypes(config: dict, labels: list[str]) -> torch.Tensor | None:
    if config["mode"] == "fixed":
        return None
    cache = project_path(config["prototype_cache"])
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload["labels"] == labels:
            return payload["embeddings"]

    definitions = json.loads(project_path(config["definitions"]).read_text())
    texts = [f"{label.replace('_', ' ')}: {definitions[label]}" for label in labels]
    model_path = project_path(config["text_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text_model = CLIPTextModelWithProjection.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    embeddings = F.normalize(text_model(**inputs).text_embeds.float(), dim=-1).cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": labels, "embeddings": embeddings}, cache)
    return embeddings


def load_cls_config(path: str | Path) -> dict:
    return load_yaml_with_base(path)


def no_event_split(
    root: Path,
    protocol: str,
    split_ratios: dict[str, float],
    group_gap_seconds: float,
    seed: int,
) -> dict[str, list[tuple[Path, str]]]:
    timestamped = []
    untimestamped = []
    for path in sorted(root.glob("*.png")):
        match = NO_EVENT_TIMESTAMP.fullmatch(path.name)
        if match:
            timestamped.append(
                (datetime.strptime(match.group(1), "%Y-%m-%dT%H%M%S.%f"), path)
            )
        else:
            untimestamped.append(path)

    groups = []
    current = []
    previous = None
    for captured_at, path in sorted(timestamped):
        if previous is not None and (captured_at - previous).total_seconds() > group_gap_seconds:
            groups.append(current)
            current = []
        current.append(path)
        previous = captured_at
    if current:
        groups.append(current)
    if untimestamped:
        groups.append(untimestamped)

    split_names = ("train", "val", "test")
    total = sum(map(len, groups))
    ratio_total = sum(split_ratios[name] for name in split_names)
    targets = {
        name: total * split_ratios[name] / ratio_total for name in split_names
    }
    counts = Counter({name: 0 for name in split_names})
    result = {name: [] for name in split_names}

    def digest(group: list[Path]) -> bytes:
        names = "\n".join(path.name for path in group)
        return hashlib.sha256(f"{seed}:{protocol}:{names}".encode()).digest()

    for group in sorted(groups, key=lambda value: (-len(value), digest(value))):
        split = max(
            split_names,
            key=lambda name: (targets[name] - counts[name]) / targets[name],
        )
        names = "\n".join(path.name for path in group)
        group_id = f"neggrp_{hashlib.sha256(names.encode()).hexdigest()[:16]}"
        result[split].extend((path, group_id) for path in group)
        counts[split] += len(group)
    return result


def read_no_event_images(
    data_config: dict,
    protocol: str,
    split: str,
) -> list[tuple[Path, str]]:
    settings = data_config.get("no_event")
    if not settings:
        return []
    return no_event_split(
        project_path(settings["root"]),
        protocol,
        settings["split_ratios"],
        settings["group_gap_seconds"],
        settings.get("seed", 43),
    )[split]


class GroundClassificationDataset(Dataset):
    def __init__(
        self,
        samples,
        data_root: Path,
        targets: dict[str, torch.Tensor],
        labels: list[str],
        negative_images: list[tuple[Path, str]] | None = None,
    ) -> None:
        label_to_index = {label: index for index, label in enumerate(labels)}
        self.rows = []
        self.labels = []
        for sample in samples:
            if sample.label not in label_to_index:
                continue
            image_file = sample.context_path.relative_to(data_root).as_posix()
            if image_file not in targets:
                continue
            self.rows.append(
                {
                    "record_uid": sample.record_uid,
                    "image_file": image_file,
                    "image_path": sample.context_path,
                    "target": targets[image_file],
                    "presence": True,
                    "class_target": label_to_index[sample.label],
                    "class_label": sample.label,
                    "group_id": sample.content_group_id,
                }
            )
            self.labels.append(sample.label)

        for image_path, group_id in negative_images or []:
            uid = hashlib.sha256(image_path.name.encode()).hexdigest()[:20]
            self.rows.append(
                {
                    "record_uid": f"neg_{uid}",
                    "image_file": image_path.relative_to(data_root).as_posix(),
                    "image_path": image_path,
                    "target": torch.tensor([0.5, 0.5, 1.0, 1.0]),
                    "presence": False,
                    "class_target": -100,
                    "class_label": None,
                    "group_id": group_id,
                }
            )
            self.labels.append("no_event")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class GroundClassificationCollator(GroundCollator):
    def __call__(self, rows: list[dict]) -> dict:
        batch = super().__call__(rows)
        batch["presence_targets"] = torch.tensor(
            [row["presence"] for row in rows], dtype=torch.float32
        )
        batch["class_targets"] = torch.tensor(
            [row["class_target"] for row in rows], dtype=torch.long
        )
        batch["class_labels"] = [row["class_label"] for row in rows]
        batch["group_ids"] = [row["group_id"] for row in rows]
        return batch


def classification_sampler(
    dataset: GroundClassificationDataset,
    train_config: dict,
    seed: int,
) -> WeightedRandomSampler:
    counts = Counter(dataset.labels)
    power = train_config["class_balance_power"]
    weights = torch.tensor(
        [counts[label] ** (-power) for label in dataset.labels], dtype=torch.double
    )
    negative_indices = [i for i, label in enumerate(dataset.labels) if label == "no_event"]
    if negative_indices:
        negative_fraction = train_config["negative_fraction"]
        positive_mass = weights.sum() - weights[negative_indices].sum()
        negative_weight = (
            negative_fraction * positive_mass
            / ((1 - negative_fraction) * len(negative_indices))
        )
        weights[negative_indices] = negative_weight
    budget = train_config.get("samples_per_epoch")
    positive_records = sum(label != "no_event" for label in dataset.labels)
    if budget in (None, "positive_records"):
        budget = positive_records
    elif budget == "positive_records_preserved":
        budget = math.ceil(positive_records / (1 - train_config["negative_fraction"]))
    return WeightedRandomSampler(
        weights,
        num_samples=int(budget),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def roi_align_feature_maps(
    feature_maps: list[torch.Tensor],
    boxes_cxcywh: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    aligned = []
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh.float()).clamp(0, 1)
    for feature_map, box in zip(feature_maps, boxes_xyxy):
        height, width = feature_map.shape[-2:]
        scale = box.new_tensor([width, height, width, height])
        aligned.append(
            roi_align(
                feature_map.unsqueeze(0).float(),
                [(box * scale).unsqueeze(0)],
                output_size=output_size,
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )[0]
        )
    return torch.stack(aligned)


class QwenGroundCLS(QwenGroundMS):
    def __init__(
        self,
        vision_encoder: nn.Module,
        model_config: dict,
        classifier_config: dict,
        num_classes: int,
        class_prototypes: torch.Tensor | None = None,
    ) -> None:
        super().__init__(vision_encoder, model_config)
        fusion_dim = model_config["hidden_dim"]
        classifier_dim = classifier_config.get("hidden_dim", fusion_dim)
        components = classifier_config["components"]
        self.use_roi = components["roi"]
        self.use_global_context = components["global_context"]
        self.roi_output_size = tuple(classifier_config["roi_output_size"])
        self.classification_mode = classifier_config["mode"]

        self.presence_head = nn.Linear(fusion_dim, 1)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.zeros_(self.presence_head.bias)

        self.classifier_fusion = nn.Sequential(
            nn.Linear(2 * fusion_dim, classifier_dim),
            nn.LayerNorm(classifier_dim),
            nn.GELU(),
            nn.Dropout(classifier_config["dropout"]),
            nn.Linear(classifier_dim, classifier_dim),
            nn.LayerNorm(classifier_dim),
            nn.GELU(),
            nn.Dropout(classifier_config["dropout"]),
        )
        if self.classification_mode == "definition":
            self.register_buffer("class_prototypes", class_prototypes, persistent=False)
            self.visual_to_text = nn.Linear(
                classifier_dim, class_prototypes.shape[-1], bias=False
            )
            self.class_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        else:
            self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        gt_boxes: torch.Tensor | None = None,
        positive_mask: torch.Tensor | None = None,
        gt_box_probability: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(pixel_values, image_grid_thw)
        event_feature = output["event_feature"].float()
        presence_logits = self.presence_head(event_feature).squeeze(-1)

        predicted_boxes = output["bbox_cxcywh"].detach()
        if positive_mask is None:
            positive_mask = torch.ones(
                len(predicted_boxes), dtype=torch.bool, device=predicted_boxes.device
            )
        else:
            positive_mask = positive_mask.bool()
        if gt_boxes is None or gt_box_probability <= 0:
            gt_mask = torch.zeros_like(positive_mask)
        elif gt_box_probability >= 1:
            gt_mask = positive_mask
        else:
            gt_mask = positive_mask & (
                torch.rand(len(predicted_boxes), device=predicted_boxes.device)
                < gt_box_probability
            )
        conditioning_boxes = torch.where(
            gt_mask[:, None], gt_boxes if gt_boxes is not None else predicted_boxes, predicted_boxes
        )

        if self.use_roi:
            roi_grid = roi_align_feature_maps(
                output["fine_features"],
                conditioning_boxes,
                self.roi_output_size,
            )
            roi_feature = roi_grid.mean((-2, -1))
        else:
            roi_feature = torch.zeros_like(event_feature)

        global_feature = (
            event_feature
            if self.use_global_context
            else torch.zeros_like(event_feature)
        )
        classification_feature = self.classifier_fusion(
            torch.cat((global_feature, roi_feature), dim=-1)
        )
        if self.classification_mode == "definition":
            visual = F.normalize(self.visual_to_text(classification_feature), dim=-1)
            class_logits = (
                self.class_logit_scale.exp().clamp(max=100)
                * visual
                @ self.class_prototypes.T
            )
        else:
            class_logits = self.classifier(classification_feature)
        positive_count = positive_mask.sum().clamp_min(1)
        output.update(
            {
                "presence_logits": presence_logits,
                "class_logits": class_logits,
                "classification_feature": classification_feature,
                "roi_feature": roi_feature,
                "classification_boxes": conditioning_boxes,
                "gt_box_fraction": (gt_mask & positive_mask).float().sum()
                / positive_count,
            }
        )
        return output


def gt_box_probability(epoch: int, schedule: dict) -> float:
    if not schedule.get("enabled", True):
        return 0.0
    gt_only_epochs = schedule["gt_only_epochs"]
    transition_epochs = schedule["transition_epochs"]
    if epoch <= gt_only_epochs:
        return 1.0
    if transition_epochs <= 0:
        return 0.0
    return max(0.0, 1.0 - (epoch - gt_only_epochs) / transition_epochs)


def focal_presence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    correct_probability = torch.where(targets.bool(), probabilities, 1 - probabilities)
    alpha = torch.where(
        targets.bool(),
        targets.new_tensor(config["positive_alpha"]),
        targets.new_tensor(1 - config["positive_alpha"]),
    )
    losses = alpha * (1 - correct_probability).pow(config["focal_gamma"]) * cross_entropy
    return losses.mean()


def classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    labels: list[str],
    include_indices: set[int] | None = None,
    supported_only: bool = False,
) -> dict:
    targets = targets.long().cpu()
    predictions = predictions.long().cpu()
    if not len(targets):
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "evaluated_labels": [],
            "per_class": {},
            "confusion_matrix_labels": labels,
            "confusion_matrix": [],
        }
    matrix = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1

    indices = list(range(len(labels))) if include_indices is None else sorted(include_indices)
    if supported_only:
        indices = [index for index in indices if matrix[index].sum()]
    per_class = {}
    f1_values = []
    weighted_f1 = 0.0
    for index in indices:
        label = labels[index]
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum()) - true_positive
        false_negative = int(matrix[index].sum()) - true_positive
        support = int(matrix[index].sum())
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
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        weighted_f1 += support * f1

    total = len(targets)
    return {
        "accuracy": float((targets == predictions).float().mean()),
        "macro_f1": sum(f1_values) / max(1, len(f1_values)),
        "weighted_f1": weighted_f1 / max(1, total),
        "evaluated_labels": [labels[index] for index in indices],
        "per_class": per_class,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
    }
