#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from clear_uav.data import Sample, cap_per_class, read_samples
from clear_uav.metrics import classification_metrics
from clear_uav.modeling import require_local_model
from clear_uav.openclip_finetune import (
    OpenCLIPClassifier,
    atomic_torch_save,
    checkpoint_payload,
    label_prompts,
    pooled_features,
    seed_everything,
    set_trainable_mode,
    trainable_parameter_count,
)
from clear_uav.ontology import load_label_subset, load_ontology


class OpenCLIPImageDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        processor,
        label_to_index: dict[str, int],
        view: str,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.label_to_index = label_to_index
        self.view = view

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        sample = self.samples[index]
        path = sample.context_path if self.view == "context" else sample.evidence_path
        with Image.open(path) as source:
            image = source.convert("RGB")
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, self.label_to_index[sample.label]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an OpenCLIP linear probe or fully tuned visual classifier"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["linear_probe", "full_finetune"], required=True
    )
    parser.add_argument("--view", choices=["context", "evidence"], default="context")
    parser.add_argument("--prompt", choices=["direct", "definition"], default="definition")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        help="Image-encoder batch for linear-probe feature extraction",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def make_loader(
    samples: list[Sample],
    processor,
    label_to_index: dict[str, int],
    view: str,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        OpenCLIPImageDataset(samples, processor, label_to_index, view),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


@torch.inference_mode()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feature_batches = []
    label_batches = []
    for pixels, targets in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = pooled_features(model.get_image_features(pixel_values=pixels))
        feature_batches.append(features.float().cpu())
        label_batches.append(targets)
    return torch.cat(feature_batches), torch.cat(label_batches)


@torch.inference_mode()
def validation_macro_f1(
    model: nn.Module,
    classifier: OpenCLIPClassifier,
    loader: DataLoader,
    labels: list[str],
    device: torch.device,
    *,
    cached_features: torch.Tensor | None = None,
    cached_targets: torch.Tensor | None = None,
) -> float:
    model.eval()
    classifier.eval()
    predictions: list[set[str]] = []
    targets_out: list[set[str]] = []
    if cached_features is not None and cached_targets is not None:
        batches = DataLoader(
            TensorDataset(cached_features, cached_targets), batch_size=512, shuffle=False
        )
        for features, targets in batches:
            logits = classifier(features.to(device, non_blocking=True))
            predictions.extend({labels[index]} for index in logits.argmax(-1).tolist())
            targets_out.extend({labels[index]} for index in targets.tolist())
    else:
        for pixels, targets in loader:
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = pooled_features(model.get_image_features(pixel_values=pixels))
            logits = classifier(features)
            predictions.extend({labels[index]} for index in logits.argmax(-1).tolist())
            targets_out.extend({labels[index]} for index in targets.tolist())
    return classification_metrics(targets_out, predictions, labels)["macro_f1"]


def cosine_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OpenCLIP training requires CUDA")
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs < 1:
        raise ValueError("Batch size, gradient accumulation, and epochs must be positive")
    if args.feature_batch_size is not None and args.feature_batch_size < 1:
        raise ValueError("--feature-batch-size must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda")
    model_path = require_local_model(args.model_path)
    ontology = load_ontology(args.ontology)
    labels = list(load_label_subset(args.labels_file, ontology))
    label_to_index = {label: index for index, label in enumerate(labels)}
    included = set(labels)
    train_samples = read_samples(
        args.train_csv, args.data_root, limit=args.max_samples, include_labels=included
    )
    train_samples = cap_per_class(train_samples, args.max_per_class, args.seed)
    val_samples = read_samples(
        args.val_csv, args.data_root, limit=args.max_samples, include_labels=included
    )
    if not train_samples or not val_samples:
        raise ValueError("Training and validation samples must both be non-empty")

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    backbone_parameters = set_trainable_mode(model, args.mode)
    if args.mode == "full_finetune" and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    prompts = label_prompts(labels, ontology.definitions, args.prompt)
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        text_features = pooled_features(model.get_text_features(**text_inputs))
    classifier = OpenCLIPClassifier(text_features.shape[-1], len(labels)).to(device)
    classifier.initialize_from_text(text_features)

    image_batch_size = (
        args.feature_batch_size
        if args.mode == "linear_probe" and args.feature_batch_size is not None
        else args.batch_size
    )
    train_loader = make_loader(
        train_samples,
        processor,
        label_to_index,
        args.view,
        batch_size=image_batch_size,
        workers=args.num_workers,
        shuffle=args.mode == "full_finetune",
        seed=args.seed,
    )
    val_loader = make_loader(
        val_samples,
        processor,
        label_to_index,
        args.view,
        batch_size=image_batch_size,
        workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    cached_train_features = cached_train_targets = None
    cached_val_features = cached_val_targets = None
    if args.mode == "linear_probe":
        cached_train_features, cached_train_targets = collect_features(
            model, train_loader, device, "cache train features"
        )
        cached_val_features, cached_val_targets = collect_features(
            model, val_loader, device, "cache val features"
        )
        train_loader = DataLoader(
            TensorDataset(cached_train_features, cached_train_targets),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            pin_memory=True,
        )

    parameter_groups = [
        {
            "params": list(classifier.parameters()),
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = AdamW(parameter_groups)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler = cosine_scheduler(
        optimizer, updates_per_epoch * args.epochs, args.warmup_ratio
    )
    loss_function = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "final" / "openclip_classifier.pt"
    history = []
    best_macro_f1 = -1.0
    total_examples = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        if args.mode == "full_finetune":
            model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"OpenCLIP {args.mode} epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, (inputs, targets) in enumerate(progress, 1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if args.mode == "full_finetune" and args.gradient_checkpointing:
                inputs.requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = (
                    inputs
                    if args.mode == "linear_probe"
                    else pooled_features(model.get_image_features(pixel_values=inputs))
                )
                logits = classifier(features)
                loss = loss_function(logits, targets) / args.gradient_accumulation
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation
            total_examples += targets.shape[0]
            should_step = (
                batch_index % args.gradient_accumulation == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(
                loss=f"{running_loss / batch_index:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        macro_f1 = validation_macro_f1(
            model,
            classifier,
            val_loader,
            labels,
            device,
            cached_features=cached_val_features,
            cached_targets=cached_val_targets,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_macro_f1": macro_f1,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    classifier=classifier,
                    mode=args.mode,
                    labels=labels,
                    prompt=args.prompt,
                    view=args.view,
                    epoch=epoch,
                    best_macro_f1=best_macro_f1,
                ),
                checkpoint_path,
            )

    elapsed = time.monotonic() - started
    metadata = {
        "mode": args.mode,
        "model_path": str(model_path),
        "labels": labels,
        "view": args.view,
        "prompt": args.prompt,
        "classifier_batch_size": args.batch_size,
        "feature_batch_size": image_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "trainable_parameters": trainable_parameter_count(model, classifier),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters())
        + sum(parameter.numel() for parameter in classifier.parameters()),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "history": history,
        "best_val_macro_f1": best_macro_f1,
        "train_metrics": {
            "train_runtime": elapsed,
            "train_samples_per_second": total_examples / elapsed,
        },
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
