from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.ops import roi_align

from clear_uav.ontology import load_ontology
from clear_uav.qwen_ground_ms import (
    GroundCollator,
    GroundDataset,
    QwenGroundMS,
    cxcywh_to_xyxy,
)


def load_class_labels(path: Path) -> list[str]:
    return list(load_ontology(path).labels)


class GroundClassificationDataset(GroundDataset):
    def __init__(self, samples, data_root, targets, labels: list[str]) -> None:
        super().__init__(samples, data_root, targets)
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __getitem__(self, index: int) -> dict:
        row = super().__getitem__(index)
        label = self.labels[index]
        row["class_target"] = self.label_to_index[label]
        row["class_label"] = label
        return row


class GroundClassificationCollator(GroundCollator):
    def __call__(self, rows: list[dict]) -> dict:
        batch = super().__call__(rows)
        batch["class_targets"] = torch.tensor(
            [row["class_target"] for row in rows], dtype=torch.long
        )
        batch["class_labels"] = [row["class_label"] for row in rows]
        return batch


def roi_align_fused_features(
    fused: torch.Tensor,
    pooled_shapes: list[tuple[int, int]],
    boxes_cxcywh: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    aligned = []
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh.float()).clamp(0, 1)
    for batch_index, (height, width) in enumerate(pooled_shapes):
        feature_map = (
            fused[batch_index, : height * width]
            .transpose(0, 1)
            .reshape(1, fused.shape[-1], height, width)
            .float()
        )
        scale = boxes_xyxy.new_tensor([width, height, width, height])
        spatial_box = (boxes_xyxy[batch_index] * scale).unsqueeze(0)
        aligned.append(
            roi_align(
                feature_map,
                [spatial_box],
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
    ) -> None:
        super().__init__(vision_encoder, model_config)
        fusion_dim = model_config["hidden_dim"]
        classifier_dim = classifier_config.get("hidden_dim", fusion_dim)
        self.roi_output_size = tuple(classifier_config["roi_output_size"])
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
        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        gt_boxes: torch.Tensor | None = None,
        gt_box_probability: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(pixel_values, image_grid_thw)
        predicted_boxes = output["bbox_cxcywh"].detach()
        if gt_boxes is None or gt_box_probability <= 0:
            conditioning_boxes = predicted_boxes
            gt_mask = torch.zeros(
                len(predicted_boxes), dtype=torch.bool, device=predicted_boxes.device
            )
        elif gt_box_probability >= 1:
            conditioning_boxes = gt_boxes
            gt_mask = torch.ones(
                len(predicted_boxes), dtype=torch.bool, device=predicted_boxes.device
            )
        else:
            gt_mask = torch.rand(len(predicted_boxes), device=predicted_boxes.device) < gt_box_probability
            conditioning_boxes = torch.where(
                gt_mask[:, None], gt_boxes, predicted_boxes
            )

        roi_grid = roi_align_fused_features(
            output["f_fused"],
            output["pooled_shapes"],
            conditioning_boxes,
            self.roi_output_size,
        )
        roi_feature = roi_grid.mean((-2, -1))
        classification_feature = self.classifier_fusion(
            torch.cat((output["event_feature"].float(), roi_feature), dim=-1)
        )
        output.update(
            {
                "class_logits": self.classifier(classification_feature),
                "classification_feature": classification_feature,
                "roi_feature": roi_feature,
                "classification_boxes": conditioning_boxes,
                "gt_box_fraction": gt_mask.float().mean(),
            }
        )
        return output


def gt_box_probability(epoch: int, schedule: dict) -> float:
    gt_only_epochs = schedule["gt_only_epochs"]
    transition_epochs = schedule["transition_epochs"]
    if epoch <= gt_only_epochs:
        return 1.0
    if transition_epochs <= 0:
        return 0.0
    return max(0.0, 1.0 - (epoch - gt_only_epochs) / transition_epochs)


def classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    labels: list[str],
) -> dict:
    targets = targets.long().cpu()
    predictions = predictions.long().cpu()
    matrix = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1

    per_class = {}
    f1_values = []
    weighted_f1 = 0.0
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum()) - true_positive
        false_negative = int(matrix[index].sum()) - true_positive
        support = int(matrix[index].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
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
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1 / total,
        "per_class": per_class,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
    }
