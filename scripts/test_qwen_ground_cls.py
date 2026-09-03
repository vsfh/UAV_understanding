#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clear_uav.experiment_config import project_path
from clear_uav.qwen_ground_cls import (
    GroundClassificationCollator,
    GroundClassificationDataset,
    QwenGroundCLS,
    classification_metrics,
    load_cls_config,
    load_class_labels,
    load_definition_prototypes,
    read_no_event_images,
)
from clear_uav.qwen_ground_ms import (
    cxcywh_to_xyxy,
    load_bbox_targets,
    load_ground_checkpoint,
    load_qwen_vision,
    localization_metrics,
    per_box_metrics,
    read_ground_samples,
    seed_everything,
)


CONFIG = "configs/yaml/qwen_ground_cls.yaml"


def move_inputs(batch: dict, device: torch.device) -> dict:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("pixel_values", "image_grid_thw")
    }


def average_precision(items: list[tuple[float, bool]], positives: int) -> float:
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, correct) in enumerate(
        sorted(items, key=lambda item: item[0], reverse=True), 1
    ):
        if correct:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def presence_metrics(
    targets: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    predictions = scores >= threshold
    targets = targets.bool()
    true_positive = int((predictions & targets).sum())
    false_positive = int((predictions & ~targets).sum())
    false_negative = int((~predictions & targets).sum())
    true_negative = int((~predictions & ~targets).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    ap_items = list(zip(scores.tolist(), targets.tolist()))
    return {
        "accuracy": (true_positive + true_negative) / len(targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p_ap": average_precision(ap_items, int(targets.sum())),
        "n_fpr": false_positive / max(1, int((~targets).sum())),
        "threshold": threshold,
    }


def grounded_map50(
    presence_targets: torch.Tensor,
    class_targets: torch.Tensor,
    presence_scores: torch.Tensor,
    class_probabilities: torch.Tensor,
    ious: torch.Tensor,
    labels: list[str],
) -> float:
    joint_scores = presence_scores[:, None] * class_probabilities
    values = []
    for class_index in range(len(labels)):
        positives = int(
            (presence_targets & (class_targets == class_index)).sum()
        )
        if positives == 0:
            continue
        items = []
        for sample_index in range(len(presence_targets)):
            correct = bool(
                presence_targets[sample_index]
                and class_targets[sample_index] == class_index
                and ious[sample_index] >= 0.5
            )
            items.append((float(joint_scores[sample_index, class_index]), correct))
        values.append(average_precision(items, positives))
    return statistics.fmean(values)


def best_presence_threshold(
    targets: torch.Tensor,
    scores: torch.Tensor,
    max_n_fpr: float,
) -> float:
    upper = float(
        torch.nextafter(scores.max(), scores.new_tensor(float("inf")))
    )
    candidates = sorted(
        set(scores.tolist()) | {upper}
    )
    feasible = [
        threshold
        for threshold in candidates
        if presence_metrics(targets, scores, threshold)["n_fpr"] <= max_n_fpr
    ]
    return max(
        feasible,
        key=lambda threshold: (
            presence_metrics(targets, scores, threshold)["recall"],
            presence_metrics(targets, scores, threshold)["precision"],
            threshold,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    config = load_cls_config(args.config)
    device = torch.device(config["runtime"]["device"])
    data_root = project_path(config["data"]["root"])
    labels_path = config["data"].get("labels")
    labels = load_class_labels(
        project_path(config["data"]["ontology"]),
        project_path(labels_path) if labels_path else None,
    )
    targets = load_bbox_targets(project_path(config["data"]["bbox_annotations"]))
    class_prototypes = load_definition_prototypes(config["classification"], labels)

    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            seed_everything(seed)
            samples = read_ground_samples(
                data_root / protocol / "test_inputs.csv",
                data_root,
                labels_csv=data_root / protocol / "test_labels_private.csv",
                limit=config["data"].get("max_test_samples"),
            )
            negative_images = read_no_event_images(config["data"], protocol, "test")
            train_labels = {
                sample.label
                for sample in read_ground_samples(
                    data_root / protocol / "train.csv", data_root
                )
                if sample.label in labels
            }
            vision, processor = load_qwen_vision(
                project_path(config["model"]["path"]), device
            )
            model = QwenGroundCLS(
                vision,
                config["model"],
                config["classification"],
                len(labels),
                class_prototypes,
            ).to(device)
            checkpoint = load_ground_checkpoint(
                model,
                project_path(config["output"]["checkpoint"], **values),
            )
            model.eval()
            threshold = config["presence"]["threshold"]
            if config["presence"].get("calibrate_on_val"):
                calibration_samples = read_ground_samples(
                    data_root / protocol / "val.csv",
                    data_root,
                    limit=config["presence"].get("calibration_samples"),
                )
                calibration_dataset = GroundClassificationDataset(
                    calibration_samples,
                    data_root,
                    targets,
                    labels,
                    read_no_event_images(config["data"], protocol, "val"),
                )
                calibration_batches = DataLoader(
                    calibration_dataset,
                    batch_size=config["test"]["batch_size"],
                    shuffle=False,
                    num_workers=config["test"]["num_workers"],
                    collate_fn=GroundClassificationCollator(processor, config["input"]),
                    pin_memory=True,
                )
                calibration_scores, calibration_targets = [], []
                with torch.inference_mode():
                    for batch in tqdm(calibration_batches, desc="presence calibration"):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = model(**move_inputs(batch, device))
                        calibration_scores.append(output["presence_logits"].float().sigmoid().cpu())
                        calibration_targets.append(batch["presence_targets"].bool())
                threshold = best_presence_threshold(
                    torch.cat(calibration_targets),
                    torch.cat(calibration_scores),
                    config["presence"]["max_n_fpr"],
                )
                print(f"calibrated presence threshold: {threshold:.2f}")
            dataset = GroundClassificationDataset(
                samples,
                data_root,
                targets,
                labels,
                negative_images,
            )
            batches = DataLoader(
                dataset,
                batch_size=config["test"]["batch_size"],
                shuffle=False,
                num_workers=config["test"]["num_workers"],
                collate_fn=GroundClassificationCollator(processor, config["input"]),
                pin_memory=True,
                persistent_workers=config["test"]["num_workers"] > 0,
            )

            all_presence_scores = []
            all_presence_targets = []
            all_class_probabilities = []
            all_class_predictions = []
            all_class_targets = []
            all_ious = []
            positive_predictions = []
            positive_targets = []
            positive_class_predictions = []
            positive_class_targets = []
            rows = []
            with torch.inference_mode():
                for batch in tqdm(
                    batches,
                    desc=f"qwen_ground_cls {protocol} seed{seed}",
                    unit="batch",
                ):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**move_inputs(batch, device))
                    torch.cuda.synchronize()
                    latency_ms = 1000 * (time.perf_counter() - started) / len(batch["record_uids"])

                    predictions = output["bbox_cxcywh"].float().cpu()
                    batch_targets = batch["targets"]
                    presence_targets = batch["presence_targets"].bool()
                    presence_scores = output["presence_logits"].float().sigmoid().cpu()
                    probabilities = output["class_logits"].float().softmax(-1).cpu()
                    class_predictions = probabilities.argmax(-1)
                    class_targets = batch["class_targets"]
                    predicted_xyxy = cxcywh_to_xyxy(predictions).clamp(0, 1)
                    target_xyxy = cxcywh_to_xyxy(batch_targets).clamp(0, 1)
                    batch_ious = torch.zeros(len(predictions))
                    batch_errors = torch.zeros(len(predictions))
                    if presence_targets.any():
                        positive_ious, positive_errors = per_box_metrics(
                            predictions[presence_targets],
                            batch_targets[presence_targets],
                        )
                        batch_ious[presence_targets] = positive_ious
                        batch_errors[presence_targets] = positive_errors
                        positive_predictions.append(predictions[presence_targets])
                        positive_targets.append(batch_targets[presence_targets])
                        positive_class_predictions.append(class_predictions[presence_targets])
                        positive_class_targets.append(class_targets[presence_targets])

                    all_presence_scores.append(presence_scores)
                    all_presence_targets.append(presence_targets)
                    all_class_probabilities.append(probabilities)
                    all_class_predictions.append(class_predictions)
                    all_class_targets.append(class_targets)
                    all_ious.append(batch_ious)
                    for sample_index, uid in enumerate(batch["record_uids"]):
                        is_positive = bool(presence_targets[sample_index])
                        predicts_event = presence_scores[sample_index] >= threshold
                        predicted_class = int(class_predictions[sample_index])
                        row = {
                            "record_uid": uid,
                            "group_id": batch["group_ids"][sample_index],
                            "image_file": batch["image_files"][sample_index],
                            "target_presence": is_positive,
                            "target_bbox_1000": (
                                (target_xyxy[sample_index] * 1000).tolist()
                                if is_positive else None
                            ),
                            "target_category": (
                                labels[int(class_targets[sample_index])]
                                if is_positive else None
                            ),
                            "prediction_presence": bool(predicts_event),
                            "presence_score": float(presence_scores[sample_index]),
                            "prediction_bbox_1000": (
                                (predicted_xyxy[sample_index] * 1000).tolist()
                                if predicts_event else None
                            ),
                            "prediction_category": (
                                labels[predicted_class] if predicts_event else None
                            ),
                            "candidate_bbox_1000": (
                                predicted_xyxy[sample_index] * 1000
                            ).tolist(),
                            "candidate_category": labels[predicted_class],
                            "category_confidence": float(
                                probabilities[sample_index, predicted_class]
                            ),
                            "iou": float(batch_ious[sample_index]) if is_positive else None,
                            "normalized_center_error": (
                                float(batch_errors[sample_index]) if is_positive else None
                            ),
                            "latency_ms": latency_ms,
                        }
                        rows.append(row)

            presence_scores = torch.cat(all_presence_scores)
            presence_targets = torch.cat(all_presence_targets)
            class_probabilities = torch.cat(all_class_probabilities)
            class_predictions = torch.cat(all_class_predictions)
            class_targets = torch.cat(all_class_targets)
            ious = torch.cat(all_ious)
            box_predictions = torch.cat(positive_predictions)
            box_targets = torch.cat(positive_targets)
            positive_class_predictions = torch.cat(positive_class_predictions)
            positive_class_targets = torch.cat(positive_class_targets)
            positive_scores = presence_scores[presence_targets]
            positive_ious = ious[presence_targets]
            class_correct = positive_class_predictions == positive_class_targets
            predicted_present = positive_scores >= threshold
            label_to_index = {label: index for index, label in enumerate(labels)}
            seen_indices = {label_to_index[label] for label in train_labels}
            unseen_indices = set(range(len(labels))) - seen_indices
            seen_mask = torch.tensor(
                [int(target) in seen_indices for target in positive_class_targets]
            )
            unseen_mask = ~seen_mask
            metrics = {
                "presence": presence_metrics(
                    presence_targets, presence_scores, threshold
                ),
                "localization_positive_only": localization_metrics(
                    box_predictions, box_targets
                ),
                "classification_positive_only": classification_metrics(
                    positive_class_targets, positive_class_predictions, labels
                ),
                "classification_seen_only": classification_metrics(
                    positive_class_targets[seen_mask],
                    positive_class_predictions[seen_mask],
                    labels,
                    seen_indices,
                    supported_only=True,
                ),
                "classification_unseen_only": classification_metrics(
                    positive_class_targets[unseen_mask],
                    positive_class_predictions[unseen_mask],
                    labels,
                    unseen_indices,
                    supported_only=True,
                ),
                "g_map50": grounded_map50(
                    presence_targets,
                    class_targets,
                    presence_scores,
                    class_probabilities,
                    ious,
                    labels,
                ),
                "grounded_accuracy_at_iou_0.25": float(
                    (predicted_present & class_correct & (positive_ious >= 0.25)).float().mean()
                ),
                "grounded_accuracy_at_iou_0.50": float(
                    (predicted_present & class_correct & (positive_ious >= 0.50)).float().mean()
                ),
                "grounded_accuracy_at_iou_0.75": float(
                    (predicted_present & class_correct & (positive_ious >= 0.75)).float().mean()
                ),
                "median_latency_ms": statistics.median(row["latency_ms"] for row in rows),
            }
            metrics["table4"] = {
                "p_ap": metrics["presence"]["p_ap"],
                "n_fpr": metrics["presence"]["n_fpr"],
                "p_precision": metrics["presence"]["precision"],
                "p_recall": metrics["presence"]["recall"],
                "p_f1": metrics["presence"]["f1"],
                "ap50": average_precision(
                    list(
                        zip(
                            presence_scores.tolist(),
                            (presence_targets & (ious >= 0.5)).tolist(),
                        )
                    ),
                    int(presence_targets.sum()),
                ),
                "c_f1": metrics["classification_positive_only"]["macro_f1"],
                "g_map50": metrics["g_map50"],
                "valid_rate": 1.0,
                "median_ms": metrics["median_latency_ms"],
                "mean_calls": 1.0,
                "max_calls": 1,
                "threshold": threshold,
                "positive_records": int(presence_targets.sum()),
                "negative_records": int((~presence_targets).sum()),
            }
            result = {
                "experiment": config["experiment"],
                "protocol": protocol,
                "seed": seed,
                "checkpoint_epoch": checkpoint["epoch"],
                "num_samples": len(dataset),
                "num_positive": int(presence_targets.sum()),
                "num_no_event": int((~presence_targets).sum()),
                "components": config["classification"]["components"],
                "classification_mode": config["classification"]["mode"],
                "labels": labels,
                "seen_labels": sorted(train_labels),
                "unseen_labels": sorted(set(labels) - train_labels),
                "metrics": metrics,
                "rows": rows,
            }
            output_path = project_path(config["output"]["test_results"], **values)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"{protocol} seed{seed}: {metrics}")
            del model, vision
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
