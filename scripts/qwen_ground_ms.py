#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from clear_uav.experiment_config import experiment_runs, load_yaml_with_base, project_path
from clear_uav.qwen_ground_ms import (
    GroundCollator,
    GroundDataset,
    QwenGroundMS,
    checkpoint_payload,
    cosine_scheduler,
    gaussian_heatmap,
    load_bbox_targets,
    load_qwen_vision,
    localization_loss,
    read_ground_samples,
    save_checkpoint,
    seed_everything,
    spatial_heatmap_loss,
)


def make_loader(
    samples,
    data_root,
    targets,
    collator,
    batch_size,
    workers,
    shuffle,
    seed,
    class_balance_power=0.0,
    samples_per_epoch=None,
):
    dataset = GroundDataset(samples, data_root, targets)
    sampler = None
    if shuffle and class_balance_power > 0:
        class_counts = Counter(dataset.labels)
        sample_weights = torch.tensor(
            [class_counts[label] ** (-class_balance_power) for label in dataset.labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=samples_per_epoch or len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=workers > 0,
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
    targets = load_bbox_targets(project_path(config["data"]["bbox_annotations"]))
    train_samples = read_ground_samples(data_root / protocol / "train.csv", data_root)

    vision, processor = load_qwen_vision(project_path(config["model"]["path"]), device)
    model = QwenGroundMS(vision, config["model"]).to(device)
    if config["train"]["gradient_checkpointing"]:
        model.vision_encoder.gradient_checkpointing_enable()
    collator = GroundCollator(processor, config["input"])
    train_config = config["train"]
    train_batches = make_loader(
        train_samples,
        data_root,
        targets,
        collator,
        train_config["batch_size"],
        train_config["num_workers"],
        True,
        seed,
        train_config["class_balance_power"],
        train_config.get("samples_per_epoch"),
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
            {
                "params": vision_parameters,
                "lr": train_config["vision_learning_rate"],
            },
            {
                "params": structure_parameters,
                "lr": train_config["learning_rate"],
            },
        ],
        weight_decay=train_config["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
    scheduler = cosine_scheduler(
        optimizer,
        steps_per_epoch * train_config["epochs"],
        train_config["warmup_ratio"],
    )
    writer = SummaryWriter(output_dir / "tensorboard")
    global_step = 0
    print(
        f"trainable vision parameters: "
        f"{sum(parameter.numel() for parameter in vision_parameters):,}"
    )
    print(
        f"trainable grounding parameters: "
        f"{sum(parameter.numel() for parameter in structure_parameters):,}"
    )
    print(
        f"training images: {len(train_batches.dataset):,}; "
        f"samples per epoch: {len(train_batches.sampler):,}; "
        f"class balance power: {train_config['class_balance_power']}"
    )

    for epoch in range(1, train_config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_heatmap = 0.0
        progress = tqdm(
            train_batches,
            desc=f"qwen_ground_ms {protocol} seed{seed} epoch {epoch}/{train_config['epochs']}",
            unit="batch",
        )
        for batch_index, batch in enumerate(progress, 1):
            target = batch["targets"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**move_inputs(batch, device))
                prediction = output["bbox_cxcywh"]
                bbox_loss, l1, giou = localization_loss(
                    prediction.float(),
                    target,
                    train_config["l1_weight"],
                    train_config["giou_weight"],
                )
                heatmap_losses = []
                for logits, sample_target in zip(output["heatmap_logits"], target):
                    height, width = logits.shape[-2:]
                    heatmap_target = gaussian_heatmap(
                        sample_target.unsqueeze(0), height, width
                    )
                    heatmap_losses.append(spatial_heatmap_loss(logits, heatmap_target))
                heatmap_loss = torch.stack(heatmap_losses).mean()
                loss = bbox_loss + train_config["heatmap_weight"] * heatmap_loss
            (loss / train_config["gradient_accumulation"]).backward()
            running_loss += loss.item()
            running_heatmap += heatmap_loss.item()
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
                    "train/heatmap_center_error",
                    torch.linalg.vector_norm(
                        output["heatmap_center"].float() - target[:, :2], dim=-1
                    ).mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/vision_learning_rate", scheduler.get_last_lr()[0], global_step
                )
                writer.add_scalar(
                    "train/grounding_learning_rate", scheduler.get_last_lr()[1], global_step
                )
            progress.set_postfix(
                loss=f"{running_loss / batch_index:.4f}",
                heatmap_kl=f"{running_heatmap / batch_index:.4f}",
            )

        epoch_loss = running_loss / len(train_batches)
        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
        writer.flush()
        print(f"{protocol} seed{seed} epoch {epoch}: train_loss={epoch_loss:.6f}")
        epoch_metrics = {
            "train_loss": epoch_loss,
            "heatmap_kl": running_heatmap / len(train_batches),
        }
        save_checkpoint(
            checkpoint_payload(model, epoch, epoch_metrics, config),
            checkpoint_path,
        )

    writer.close()
    (output_dir / "metrics.json").write_text(
        json.dumps(epoch_metrics, indent=2),
        encoding="utf-8",
    )
    del model, vision, optimizer
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen single-view grounding")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_with_base(args.config)
    for protocol, seed in experiment_runs(config):
        train_run(config, protocol, seed)


if __name__ == "__main__":
    main()
