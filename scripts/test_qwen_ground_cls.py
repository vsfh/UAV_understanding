#!/usr/bin/env python3
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clear_uav.experiment_config import load_yaml, project_path
from clear_uav.qwen_ground_cls import (
    GroundClassificationCollator,
    GroundClassificationDataset,
    QwenGroundCLS,
    classification_metrics,
    load_class_labels,
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


def main() -> None:
    config = load_yaml(CONFIG)
    device = torch.device(config["runtime"]["device"])
    data_root = project_path(config["data"]["root"])
    labels = load_class_labels(project_path(config["data"]["ontology"]))
    targets = load_bbox_targets(project_path(config["data"]["bbox_annotations"]))

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
            vision, processor = load_qwen_vision(
                project_path(config["model"]["path"]), device
            )
            model = QwenGroundCLS(
                vision,
                config["model"],
                config["classification"],
                len(labels),
            ).to(device)
            checkpoint = load_ground_checkpoint(
                model,
                project_path(config["output"]["checkpoint"], **values),
            )
            model.eval()
            dataset = GroundClassificationDataset(
                samples, data_root, targets, labels
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

            all_predictions = []
            all_targets = []
            all_class_predictions = []
            all_class_targets = []
            rows = []
            with torch.inference_mode():
                for batch in tqdm(
                    batches,
                    desc=f"qwen_ground_cls {protocol} seed{seed}",
                    unit="batch",
                ):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**move_inputs(batch, device))
                    predictions = output["bbox_cxcywh"].float().cpu()
                    batch_targets = batch["targets"]
                    probabilities = output["class_logits"].float().softmax(-1).cpu()
                    class_predictions = probabilities.argmax(-1)
                    class_targets = batch["class_targets"]
                    predicted_xyxy = cxcywh_to_xyxy(predictions).clamp(0, 1)
                    target_xyxy = cxcywh_to_xyxy(batch_targets).clamp(0, 1)
                    batch_ious, batch_errors = per_box_metrics(predictions, batch_targets)

                    all_predictions.append(predictions)
                    all_targets.append(batch_targets)
                    all_class_predictions.append(class_predictions)
                    all_class_targets.append(class_targets)
                    for (
                        uid,
                        image_file,
                        prediction,
                        target,
                        predicted_class,
                        target_class,
                        class_probability,
                        sample_iou,
                        sample_error,
                    ) in zip(
                        batch["record_uids"],
                        batch["image_files"],
                        predicted_xyxy,
                        target_xyxy,
                        class_predictions,
                        class_targets,
                        probabilities,
                        batch_ious,
                        batch_errors,
                    ):
                        rows.append(
                            {
                                "record_uid": uid,
                                "image_file": image_file,
                                "prediction_bbox_1000": (prediction * 1000).tolist(),
                                "target_bbox_1000": (target * 1000).tolist(),
                                "prediction_category": labels[int(predicted_class)],
                                "target_category": labels[int(target_class)],
                                "category_confidence": float(class_probability[predicted_class]),
                                "iou": float(sample_iou),
                                "normalized_center_error": float(sample_error),
                            }
                        )

            box_predictions = torch.cat(all_predictions)
            box_targets = torch.cat(all_targets)
            class_predictions = torch.cat(all_class_predictions)
            class_targets = torch.cat(all_class_targets)
            ious, _ = per_box_metrics(box_predictions, box_targets)
            class_correct = class_predictions == class_targets
            metrics = {
                "localization": localization_metrics(box_predictions, box_targets),
                "classification": classification_metrics(
                    class_targets, class_predictions, labels
                ),
                "grounded_accuracy_at_iou_0.25": float(
                    (class_correct & (ious >= 0.25)).float().mean()
                ),
                "grounded_accuracy_at_iou_0.50": float(
                    (class_correct & (ious >= 0.50)).float().mean()
                ),
                "grounded_accuracy_at_iou_0.75": float(
                    (class_correct & (ious >= 0.75)).float().mean()
                ),
            }
            result = {
                "experiment": config["experiment"],
                "protocol": protocol,
                "seed": seed,
                "checkpoint_epoch": checkpoint["epoch"],
                "num_samples": len(dataset),
                "labels": labels,
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
