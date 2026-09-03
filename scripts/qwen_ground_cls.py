#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from clear_uav.experiment_config import experiment_runs, project_path
from clear_uav.qwen_ground_cls import (
    GroundClassificationCollator,
    GroundClassificationDataset,
    QwenGroundCLS,
    classification_sampler,
    focal_presence_loss,
    gt_box_probability,
    load_cls_config,
    load_class_labels,
    load_definition_prototypes,
    read_no_event_images,
)
from clear_uav.qwen_ground_ms import (
    checkpoint_payload,
    cosine_scheduler,
    gaussian_heatmap,
    load_bbox_targets,
    load_ground_checkpoint,
    load_qwen_vision,
    localization_loss,
    read_ground_samples,
    save_checkpoint,
    seed_everything,
    spatial_heatmap_loss,
)


CONFIG = "configs/yaml/qwen_ground_cls.yaml"


def make_loader(
    samples,
    negative_images,
    data_root,
    targets,
    labels,
    collator,
    train_config,
    seed,
):
    dataset = GroundClassificationDataset(
        samples,
        data_root,
        targets,
        labels,
        negative_images,
    )
    sampler = classification_sampler(dataset, train_config, seed)
    return DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        sampler=sampler,
        num_workers=train_config["num_workers"],
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=train_config["num_workers"] > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def move_inputs(batch: dict, device: torch.device) -> dict:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("pixel_values", "image_grid_thw")
    }


def train_run(config: dict, protocol: str, seed: int) -> None:
    values = {"protocol": protocol, "seed": seed}
    output_dir = project_path(config["output"]["root"], **values)
    checkpoint_path = project_path(config["output"]["checkpoint"], **values)
    if config["output"].get("skip_existing") and checkpoint_path.exists():
        print(f"[skip] {checkpoint_path}")
        return

    seed_everything(seed)
    device = torch.device(config["runtime"]["device"])
    data_root = project_path(config["data"]["root"])
    labels_path = config["data"].get("labels")
    labels = load_class_labels(
        project_path(config["data"]["ontology"]),
        project_path(labels_path) if labels_path else None,
    )
    targets = load_bbox_targets(project_path(config["data"]["bbox_annotations"]))
    train_samples = read_ground_samples(data_root / protocol / "train.csv", data_root)
    negative_images = read_no_event_images(config["data"], protocol, "train")

    vision, processor = load_qwen_vision(project_path(config["model"]["path"]), device)
    class_prototypes = load_definition_prototypes(config["classification"], labels)
    model = QwenGroundCLS(
        vision,
        config["model"],
        config["classification"],
        len(labels),
        class_prototypes,
    ).to(device)
    initialization_path = project_path(
        config["initialization"]["grounding_checkpoint"], **values
    )
    initialization = load_ground_checkpoint(model, initialization_path)
    print(
        f"initialized grounding backbone from {initialization_path} "
        f"(epoch {initialization['epoch']})"
    )
    train_config = config["train"]
    if train_config["gradient_checkpointing"]:
        model.vision_encoder.gradient_checkpointing_enable()
    train_batches = make_loader(
        train_samples,
        negative_images,
        data_root,
        targets,
        labels,
        GroundClassificationCollator(processor, config["input"]),
        train_config,
        seed,
    )

    vision_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("vision_encoder.") and parameter.requires_grad
    ]
    structure_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("vision_encoder.") and parameter.requires_grad
    ]
    parameters = [*vision_parameters, *structure_parameters]
    optimizer = AdamW(
        [
            {"params": vision_parameters, "lr": train_config["vision_learning_rate"]},
            {"params": structure_parameters, "lr": train_config["learning_rate"]},
        ],
        weight_decay=train_config["weight_decay"],
    )
    steps_per_epoch = math.ceil(
        len(train_batches) / train_config["gradient_accumulation"]
    )
    scheduler = cosine_scheduler(
        optimizer,
        steps_per_epoch * train_config["epochs"],
        train_config["warmup_ratio"],
    )
    writer = SummaryWriter(output_dir / "tensorboard")
    global_step = 0
    print(f"classes: {len(labels)}")
    print(
        "trainable vision parameters: "
        f"{sum(parameter.numel() for parameter in vision_parameters):,}"
    )
    print(
        "trainable grounding/classification parameters: "
        f"{sum(parameter.numel() for parameter in structure_parameters):,}"
    )
    print(
        f"training images: {sum(label != 'no_event' for label in train_batches.dataset.labels):,} positive + "
        f"{len(negative_images):,} no-event; "
        f"samples per epoch: {len(train_batches.sampler):,}; "
        f"negative exposure: {train_config['negative_fraction']:.0%}; "
        f"class balance power: {train_config['class_balance_power']}"
    )

    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scheduled_gt_probability = gt_box_probability(
            epoch, config["box_conditioning"]
        )
        running_loss = 0.0
        running_classification_loss = 0.0
        running_accuracy = 0.0
        running_presence_loss = 0.0
        running_presence_accuracy = 0.0
        progress = tqdm(
            train_batches,
            desc=(
                f"qwen_ground_cls {protocol} seed{seed} "
                f"epoch {epoch}/{train_config['epochs']}"
            ),
            unit="batch",
        )
        for batch_index, batch in enumerate(progress, 1):
            target = batch["targets"].to(device, non_blocking=True)
            class_target = batch["class_targets"].to(device, non_blocking=True)
            presence_target = batch["presence_targets"].to(device, non_blocking=True)
            positive_mask = presence_target.bool()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    **move_inputs(batch, device),
                    gt_boxes=target,
                    positive_mask=positive_mask,
                    gt_box_probability=scheduled_gt_probability,
                )
                prediction = output["bbox_cxcywh"]
                if positive_mask.any():
                    bbox_loss, l1, giou = localization_loss(
                        prediction[positive_mask].float(),
                        target[positive_mask],
                        train_config["l1_weight"],
                        train_config["giou_weight"],
                    )
                    heatmap_losses = []
                    for sample_index in torch.where(positive_mask)[0].tolist():
                        logits = output["heatmap_logits"][sample_index]
                        sample_target = target[sample_index]
                        height, width = logits.shape[-2:]
                        heatmap_target = gaussian_heatmap(
                            sample_target.unsqueeze(0), height, width
                        )
                        heatmap_losses.append(
                            spatial_heatmap_loss(logits, heatmap_target)
                        )
                    heatmap_loss = torch.stack(heatmap_losses).mean()
                    classification_loss = F.cross_entropy(
                        output["class_logits"][positive_mask].float(),
                        class_target[positive_mask],
                        label_smoothing=train_config["label_smoothing"],
                    )
                else:
                    zero = prediction.sum() * 0
                    bbox_loss = l1 = giou = heatmap_loss = classification_loss = zero
                presence_loss = focal_presence_loss(
                    output["presence_logits"].float(),
                    presence_target,
                    config["presence"],
                )
                loss = (
                    bbox_loss
                    + train_config["heatmap_weight"] * heatmap_loss
                    + train_config["classification_weight"] * classification_loss
                    + train_config["presence_weight"] * presence_loss
                )

            (loss / train_config["gradient_accumulation"]).backward()
            if positive_mask.any():
                accuracy = (
                    output["class_logits"][positive_mask].argmax(-1)
                    == class_target[positive_mask]
                ).float().mean()
            else:
                accuracy = loss.new_zeros(())
            presence_accuracy = (
                (output["presence_logits"] >= 0) == positive_mask
            ).float().mean()
            running_loss += loss.item()
            running_classification_loss += classification_loss.item()
            running_accuracy += accuracy.item()
            running_presence_loss += presence_loss.item()
            running_presence_accuracy += presence_accuracy.item()
            if (
                batch_index % train_config["gradient_accumulation"] == 0
                or batch_index == len(train_batches)
            ):
                nn.utils.clip_grad_norm_(parameters, train_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/l1", l1.item(), global_step)
                writer.add_scalar("train/giou", giou.item(), global_step)
                writer.add_scalar("train/heatmap_kl", heatmap_loss.item(), global_step)
                writer.add_scalar(
                    "train/classification", classification_loss.item(), global_step
                )
                writer.add_scalar("train/presence", presence_loss.item(), global_step)
                writer.add_scalar("train/classification_accuracy", accuracy.item(), global_step)
                writer.add_scalar(
                    "train/presence_accuracy", presence_accuracy.item(), global_step
                )
                writer.add_scalar(
                    "train/scheduled_gt_box_probability",
                    scheduled_gt_probability,
                    global_step,
                )
                writer.add_scalar(
                    "train/actual_gt_box_fraction",
                    output["gt_box_fraction"].item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/vision_learning_rate", scheduler.get_last_lr()[0], global_step
                )
                writer.add_scalar(
                    "train/structure_learning_rate", scheduler.get_last_lr()[1], global_step
                )
            progress.set_postfix(
                loss=f"{running_loss / batch_index:.4f}",
                cls=f"{running_classification_loss / batch_index:.4f}",
                acc=f"{running_accuracy / batch_index:.3f}",
                presence=f"{running_presence_accuracy / batch_index:.3f}",
                gt_box=f"{scheduled_gt_probability:.2f}",
            )

        epoch_metrics = {
            "train_loss": running_loss / len(train_batches),
            "classification_loss": running_classification_loss / len(train_batches),
            "classification_accuracy": running_accuracy / len(train_batches),
            "presence_loss": running_presence_loss / len(train_batches),
            "presence_accuracy": running_presence_accuracy / len(train_batches),
            "gt_box_probability": scheduled_gt_probability,
        }
        for name, value in epoch_metrics.items():
            writer.add_scalar(f"epoch/{name}", value, epoch)
        writer.flush()
        print(f"{protocol} seed{seed} epoch {epoch}: {epoch_metrics}")
        payload = checkpoint_payload(model, epoch, epoch_metrics, config)
        payload["format_version"] = 8
        payload["labels"] = labels
        save_checkpoint(payload, checkpoint_path)

    writer.close()
    (output_dir / "metrics.json").write_text(
        json.dumps(epoch_metrics, indent=2), encoding="utf-8"
    )
    del model, vision, optimizer
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    config = load_cls_config(args.config)
    for protocol, seed in experiment_runs(config):
        train_run(config, protocol, seed)


if __name__ == "__main__":
    main()
