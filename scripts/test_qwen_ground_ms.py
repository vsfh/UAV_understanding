#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clear_uav.experiment_config import load_yaml, project_path
from clear_uav.qwen_ground_ms import (
    GroundCollator,
    GroundDataset,
    QwenGroundMS,
    cxcywh_to_xyxy,
    load_bbox_targets,
    load_ground_checkpoint,
    load_qwen_vision,
    localization_metrics,
    per_box_metrics,
    read_ground_samples,
    seed_everything,
)


def move_inputs(batch: dict, device: torch.device) -> dict:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("pixel_values", "image_grid_thw")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Qwen single-view grounding")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    device = torch.device(config["runtime"]["device"])
    data_root = project_path(config["data"]["root"])
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
            model = QwenGroundMS(vision, config["model"]).to(device)
            checkpoint = load_ground_checkpoint(
                model,
                project_path(config["output"]["checkpoint"], **values),
            )
            model.eval()
            dataset = GroundDataset(samples, data_root, targets)
            batches = DataLoader(
                dataset,
                batch_size=config["test"]["batch_size"],
                shuffle=False,
                num_workers=config["test"]["num_workers"],
                collate_fn=GroundCollator(processor, config["input"]),
                pin_memory=True,
                persistent_workers=config["test"]["num_workers"] > 0,
            )

            all_predictions, all_targets, rows = [], [], []
            with torch.inference_mode():
                for batch in tqdm(
                    batches,
                    desc=f"qwen_ground_ms {protocol} seed{seed}",
                    unit="batch",
                ):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**move_inputs(batch, device))
                    predictions = output["bbox_cxcywh"].float().cpu()
                    batch_targets = batch["targets"]
                    predicted_xyxy = cxcywh_to_xyxy(predictions).clamp(0, 1)
                    target_xyxy = cxcywh_to_xyxy(batch_targets).clamp(0, 1)
                    batch_ious, batch_errors = per_box_metrics(predictions, batch_targets)
                    all_predictions.append(predictions)
                    all_targets.append(batch_targets)
                    for uid, image_file, prediction, target, sample_iou, sample_error in zip(
                        batch["record_uids"],
                        batch["image_files"],
                        predicted_xyxy,
                        target_xyxy,
                        batch_ious,
                        batch_errors,
                    ):
                        rows.append(
                            {
                                "record_uid": uid,
                                "image_file": image_file,
                                "prediction_bbox_1000": (prediction * 1000).tolist(),
                                "target_bbox_1000": (target * 1000).tolist(),
                                "iou": float(sample_iou),
                                "normalized_center_error": float(sample_error),
                            }
                        )

            metrics = localization_metrics(
                torch.cat(all_predictions),
                torch.cat(all_targets),
            )
            result = {
                "experiment": config["experiment"],
                "protocol": protocol,
                "seed": seed,
                "checkpoint_epoch": checkpoint["epoch"],
                "num_samples": len(dataset),
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
