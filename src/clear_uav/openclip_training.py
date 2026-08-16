from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from clear_uav.data import Sample, cap_per_class, read_samples
from clear_uav.experiment_config import experiment_runs, project_path
from clear_uav.metrics import classification_metrics
from clear_uav.openclip_finetune import (
    OpenCLIPClassifier,
    atomic_torch_save,
    checkpoint_payload,
    label_prompts,
    pooled_features,
    seed_everything,
    set_trainable_mode,
)
from clear_uav.ontology import load_label_subset, load_ontology


class ImageDataset(Dataset):
    def __init__(self, samples, processor, label_to_index, view):
        self.samples = samples
        self.processor = processor
        self.label_to_index = label_to_index
        self.view = view

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample.context_path if self.view == "context" else sample.evidence_path
        with Image.open(path) as source:
            pixels = self.processor(images=source.convert("RGB"), return_tensors="pt")
        return pixels["pixel_values"][0], self.label_to_index[sample.label]


def loader(samples, processor, label_to_index, view, batch_size, workers, shuffle, seed):
    return DataLoader(
        ImageDataset(samples, processor, label_to_index, view),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def cache_features(model, batches, device, description):
    model.eval()
    features, targets = [], []
    for pixels, labels in tqdm(batches, desc=description, unit="batch", dynamic_ncols=True):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features.append(pooled_features(model.get_image_features(pixel_values=pixels)).cpu())
        targets.append(labels)
    return torch.cat(features).float(), torch.cat(targets)


@torch.inference_mode()
def validate(model, head, batches, labels, device, mode):
    model.eval()
    head.eval()
    targets, predictions = [], []
    for inputs, batch_targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = inputs if mode == "linear_probe" else pooled_features(
                model.get_image_features(pixel_values=inputs)
            )
            indices = head(features).argmax(-1).tolist()
        targets.extend({labels[index]} for index in batch_targets.tolist())
        predictions.extend({labels[index]} for index in indices)
    return classification_metrics(targets, predictions, labels)["macro_f1"]


def scheduler_for(optimizer, steps, warmup_ratio):
    warmup = max(1, round(steps * warmup_ratio))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_openclip(config: dict, mode: str) -> None:
    model_path = project_path(config["model"]["path"])
    data_root = project_path(config["data"]["root"])
    ontology = load_ontology(project_path(config["data"]["ontology"]))
    labels = list(
        load_label_subset(project_path(config["data"]["labels"]), ontology)
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_config = config["train"]
    device = torch.device(config["runtime"]["device"])

    for protocol, seed in experiment_runs(config):
        run_values = {"protocol": protocol, "seed": seed}
        output_dir = project_path(config["output"]["root"], **run_values)
        checkpoint = output_dir / "best.pt"
        if config["output"].get("skip_existing") and checkpoint.exists():
            print(f"[skip] {checkpoint}")
            continue

        seed_everything(seed)
        train_samples = cap_per_class(
            read_samples(
                data_root / protocol / "train.csv",
                data_root,
                include_labels=set(labels),
            ),
            config["data"]["max_per_class"],
            seed,
        )
        val_samples = read_samples(
            data_root / protocol / "val.csv", data_root, include_labels=set(labels)
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16
        ).to(device)
        backbone = set_trainable_mode(model, mode)
        if mode == "full_finetune" and train_config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()

        prompts = label_prompts(labels, ontology.definitions, train_config["prompt"])
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            text_features = pooled_features(model.get_text_features(**text_inputs))
        head = OpenCLIPClassifier(text_features.shape[-1], len(labels)).to(device)
        head.initialize_from_text(text_features)

        image_batch = (
            train_config["feature_batch_size"]
            if mode == "linear_probe"
            else train_config["batch_size"]
        )
        train_batches = loader(
            train_samples,
            processor,
            label_to_index,
            train_config["view"],
            image_batch,
            train_config["num_workers"],
            mode == "full_finetune",
            seed,
        )
        val_batches = loader(
            val_samples,
            processor,
            label_to_index,
            train_config["view"],
            image_batch,
            train_config["num_workers"],
            False,
            seed,
        )
        if mode == "linear_probe":
            train_features, train_targets = cache_features(
                model, train_batches, device, f"{protocol} seed{seed} train features"
            )
            val_features, val_targets = cache_features(
                model, val_batches, device, f"{protocol} seed{seed} val features"
            )
            train_batches = DataLoader(
                TensorDataset(train_features, train_targets),
                batch_size=train_config["batch_size"],
                shuffle=True,
            )
            val_batches = DataLoader(
                TensorDataset(val_features, val_targets), batch_size=512
            )

        groups = [{"params": head.parameters(), "lr": train_config["learning_rate"]}]
        if backbone:
            groups.append(
                {"params": backbone, "lr": train_config["backbone_learning_rate"]}
            )
        optimizer = AdamW(groups, weight_decay=train_config["weight_decay"])
        updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
        schedule = scheduler_for(
            optimizer, updates * train_config["epochs"], train_config["warmup_ratio"]
        )
        writer = SummaryWriter(output_dir / "tensorboard")
        loss_fn = nn.CrossEntropyLoss()
        best_f1 = -1.0
        global_step = 0
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, train_config["epochs"] + 1):
            model.train(mode == "full_finetune")
            head.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            progress = tqdm(
                train_batches,
                desc=f"{mode} {protocol} seed{seed} epoch {epoch}/{train_config['epochs']}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch_index, (inputs, targets) in enumerate(progress, 1):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if mode == "full_finetune" and train_config["gradient_checkpointing"]:
                    inputs.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = inputs if mode == "linear_probe" else pooled_features(
                        model.get_image_features(pixel_values=inputs)
                    )
                    loss = loss_fn(head(features), targets)
                (loss / train_config["gradient_accumulation"]).backward()
                total_loss += loss.item()
                if (
                    batch_index % train_config["gradient_accumulation"] == 0
                    or batch_index == len(train_batches)
                ):
                    nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in optimizer.param_groups
                            for parameter in group["params"]
                        ],
                        1.0,
                    )
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/learning_rate", schedule.get_last_lr()[0], global_step
                    )
                progress.set_postfix(loss=f"{total_loss / batch_index:.4f}")

            macro_f1 = validate(model, head, val_batches, labels, device, mode)
            writer.add_scalar("epoch/train_loss", total_loss / len(train_batches), epoch)
            writer.add_scalar("epoch/val_macro_f1", macro_f1, epoch)
            writer.flush()
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                atomic_torch_save(
                    checkpoint_payload(
                        model=model,
                        classifier=head,
                        mode=mode,
                        labels=labels,
                        prompt=train_config["prompt"],
                        view=train_config["view"],
                        epoch=epoch,
                        best_macro_f1=best_f1,
                    ),
                    checkpoint,
                )
        writer.close()
        (output_dir / "metrics.json").write_text(
            json.dumps({"best_val_macro_f1": best_f1}, indent=2), encoding="utf-8"
        )
        del model, head, optimizer
        torch.cuda.empty_cache()
