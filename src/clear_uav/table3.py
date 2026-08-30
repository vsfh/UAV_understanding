from __future__ import annotations

import json
import math
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from clear_uav.data import read_private_test_samples, read_samples
from clear_uav.experiment_config import experiment_runs, project_path
from clear_uav.modeling import LORA_PATTERNS, load_qwen
from clear_uav.ontology import load_label_subset, load_ontology


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_hf_model(model_config: dict) -> Path:
    destination = project_path(model_config["path"])
    if (destination / "config.json").is_file():
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
            "pytorch_model.bin",
        ],
    )
    return destination


def labels_from_config(config: dict) -> list[str]:
    ontology = load_ontology(project_path(config["data"]["ontology"]))
    return list(load_label_subset(project_path(config["data"]["labels"]), ontology))


def bbox_records(path: Path) -> dict[str, tuple[float, float, float, float]]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    images = {row["id"]: row["file_name"] for row in coco["images"]}
    return {
        images[row["image_id"]]: tuple(row["bbox"])
        for row in coco["annotations"]
    }


def sample_bbox(sample, data_root: Path, boxes) -> tuple[float, float, float, float]:
    name = sample.context_path.relative_to(data_root).as_posix()
    return boxes[name]


def expanded_box(box, image_size, margin: float) -> tuple[int, int, int, int]:
    x, y, width, height = box
    image_width, image_height = image_size
    extra_x, extra_y = width * margin, height * margin
    return (
        max(0, round(x - extra_x)),
        max(0, round(y - extra_y)),
        min(image_width, round(x + width + extra_x)),
        min(image_height, round(y + height + extra_y)),
    )


def render_view(sample, data_root: Path, boxes, view: str, context_margin: float):
    if view == "gt_crop":
        with Image.open(sample.evidence_path) as source:
            return source.convert("RGB")

    with Image.open(sample.context_path) as source:
        image = source.convert("RGB")
    box = sample_bbox(sample, data_root, boxes)
    roi = expanded_box(box, image.size, 0.0)
    if view == "drawn_roi":
        drawn = image.copy()
        width = max(4, round(max(image.size) / 300))
        ImageDraw.Draw(drawn).rectangle(roi, outline=(255, 0, 0), width=width)
        return drawn
    if view == "coordinates":
        return image
    if view == "masked_background":
        masked = Image.new("RGB", image.size, (127, 127, 127))
        masked.paste(image.crop(roi), roi)
        return masked
    if view == "roi_context":
        return image.crop(expanded_box(box, image.size, context_margin))
    raise ValueError(view)


def coordinate_text(sample, data_root: Path, boxes) -> str:
    x, y, width, height = sample_bbox(sample, data_root, boxes)
    with Image.open(sample.context_path) as image:
        image_width, image_height = image.size
    values = [
        round(1000 * x / image_width),
        round(1000 * y / image_height),
        round(1000 * (x + width) / image_width),
        round(1000 * (y + height) / image_height),
    ]
    return f"The GT ROI coordinates are bbox_1000={values}."


def class_sampler(labels: list[str], power: float, seed: int):
    counts = Counter(labels)
    weights = torch.tensor([counts[label] ** -power for label in labels], dtype=torch.double)
    return WeightedRandomSampler(
        weights,
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def table3_metrics(targets, predictions, valid, labels):
    per_class = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": sum(t == p for t, p in zip(targets, predictions)) / len(targets),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels),
        "valid_rate": sum(valid) / len(valid),
        "per_class": per_class,
    }


def save_predictions(
    path: Path,
    experiment: str,
    protocol: str,
    seed: int,
    samples,
    predictions,
    valid,
    metrics,
    raw_outputs=None,
):
    if raw_outputs is None:
        raw_outputs = [None] * len(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "protocol": protocol,
                "seed": seed,
                "num_samples": len(samples),
                "metrics": metrics,
                "rows": [
                    {
                        "record_uid": sample.record_uid,
                        "target": sample.label,
                        "prediction": prediction,
                        "valid": is_valid,
                    }
                    | {"raw_output": raw_output}
                    for sample, prediction, is_valid, raw_output in zip(
                        samples, predictions, valid, raw_outputs
                    )
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, round(total_steps * warmup_ratio))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EncoderDataset(Dataset):
    def __init__(self, samples, processor, labels):
        self.samples = samples
        self.processor = processor
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.evidence_path) as source:
            pixels = self.processor(images=source.convert("RGB"), return_tensors="pt")
        return pixels["pixel_values"][0], self.label_to_index[sample.label], sample.record_uid


class EncoderClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim: int, classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, pixels):
        output = self.backbone(pixel_values=pixels)
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        return self.classifier(features.float())


def encoder_hidden_dim(backbone) -> int:
    hidden_sizes = getattr(backbone.config, "hidden_sizes", None)
    return hidden_sizes[-1] if hidden_sizes else backbone.config.hidden_size


@torch.inference_mode()
def evaluate_encoder(model, batches, labels, device):
    model.eval()
    targets, predictions, uids = [], [], []
    for pixels, indices, batch_uids in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(pixels.to(device, non_blocking=True)).argmax(-1).cpu().tolist()
        targets.extend(labels[index] for index in indices.tolist())
        predictions.extend(labels[index] for index in predicted)
        uids.extend(batch_uids)
    valid = [True] * len(targets)
    return targets, predictions, valid, uids, table3_metrics(targets, predictions, valid, labels)


def encoder_loader(samples, processor, labels, batch_size, workers, sampler=None):
    return DataLoader(
        EncoderDataset(samples, processor, labels),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train_encoder(config: dict) -> None:
    labels = labels_from_config(config)
    label_set = set(labels)
    root = project_path(config["data"]["root"])
    model_path = project_path(config["model"]["path"])
    device = torch.device(config["runtime"]["device"])
    train_config = config["train"]
    for protocol, seed in experiment_runs(config):
        values = {"protocol": protocol, "seed": seed}
        output_dir = project_path(config["output"]["root"], **values)
        checkpoint_path = output_dir / "best.pt"
        if config["output"].get("skip_existing") and checkpoint_path.exists():
            print(f"[skip] {checkpoint_path}")
            continue
        seed_everything(seed)
        train_samples = read_samples(root / protocol / "train.csv", root, include_labels=label_set)
        val_samples = read_samples(root / protocol / "val.csv", root, include_labels=label_set)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        model = EncoderClassifier(backbone, encoder_hidden_dim(backbone), len(labels)).to(device)
        sampler = class_sampler(
            [sample.label for sample in train_samples],
            train_config["class_balance_power"],
            seed,
        )
        train_batches = encoder_loader(
            train_samples, processor, labels, train_config["batch_size"],
            train_config["num_workers"], sampler
        )
        val_batches = encoder_loader(
            val_samples, processor, labels, train_config["batch_size"],
            train_config["num_workers"]
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
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir / "tensorboard")
        best_f1 = -1.0
        global_step = 0
        for epoch in range(1, train_config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for batch_index, (pixels, targets, _) in enumerate(
                tqdm(train_batches, desc=f"{config['experiment']} {protocol} epoch {epoch}"), 1
            ):
                pixels, targets = pixels.to(device), targets.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nn.functional.cross_entropy(model(pixels), targets)
                (loss / train_config["gradient_accumulation"]).backward()
                total_loss += loss.item()
                if batch_index % train_config["gradient_accumulation"] == 0 or batch_index == len(train_batches):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    writer.add_scalar("train/loss", loss.item(), global_step)
            metrics = evaluate_encoder(model, val_batches, labels, device)[-1]
            writer.add_scalar("epoch/train_loss", total_loss / len(train_batches), epoch)
            writer.add_scalar("epoch/val_macro_f1", metrics["macro_f1"], epoch)
            writer.flush()
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(
                    {
                        "epoch": epoch,
                        "labels": labels,
                        "backbone": model.backbone.state_dict(),
                        "classifier": model.classifier.state_dict(),
                        "metrics": metrics,
                    },
                    temporary,
                )
                temporary.replace(checkpoint_path)
        writer.close()
        del model, optimizer
        torch.cuda.empty_cache()


def test_encoder(config: dict) -> None:
    labels = labels_from_config(config)
    label_set = set(labels)
    root = project_path(config["data"]["root"])
    model_path = project_path(config["model"]["path"])
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            samples = read_private_test_samples(
                root / protocol / "test_inputs.csv",
                root / protocol / "test_labels_private.csv",
                root,
                include_labels=label_set,
            )
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
            model = EncoderClassifier(backbone, encoder_hidden_dim(backbone), len(labels)).to(device)
            checkpoint = torch.load(
                project_path(config["output"]["checkpoint"], **values),
                map_location="cpu",
                weights_only=False,
            )
            model.backbone.load_state_dict(checkpoint["backbone"])
            model.classifier.load_state_dict(checkpoint["classifier"])
            batches = encoder_loader(
                samples, processor, labels, config["test"]["batch_size"],
                config["test"]["num_workers"]
            )
            targets, predictions, valid, _, metrics = evaluate_encoder(model, batches, labels, device)
            save_predictions(
                project_path(config["output"]["results"], **values),
                config["experiment"], protocol, seed, samples, predictions, valid, metrics
            )
            print(protocol, seed, metrics)
            del model
            torch.cuda.empty_cache()


class VlmDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class QwenCollator:
    def __init__(
        self, processor, labels, data_root, boxes, view, margin, prompt, max_pixels, training
    ):
        self.processor = processor
        self.labels = labels
        self.data_root = data_root
        self.boxes = boxes
        self.view = view
        self.margin = margin
        self.prompt = prompt
        self.max_pixels = max_pixels
        self.training = training

    def messages(self, sample, answer=None):
        image = render_view(sample, self.data_root, self.boxes, self.view, self.margin)
        coordinates = ""
        if "{coordinates}" in self.prompt["user"]:
            coordinates = coordinate_text(sample, self.data_root, self.boxes)
        instruction = self.prompt["user"].format(
            categories=", ".join(self.labels),
            coordinates=coordinates,
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.prompt["system"]}],
            },
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def __call__(self, samples):
        conversations = [self.messages(sample, sample.label if self.training else None) for sample in samples]
        encoded = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=not self.training,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "size": {"longest_edge": self.max_pixels, "shortest_edge": min(65536, self.max_pixels)},
            },
        )
        if self.training:
            prompts = self.processor.apply_chat_template(
                [messages[:-1] for messages in conversations],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "padding": True,
                    "size": {"longest_edge": self.max_pixels, "shortest_edge": min(65536, self.max_pixels)},
                },
            )
            labels = torch.full_like(encoded["input_ids"], -100)
            for row in range(len(samples)):
                full = encoded["attention_mask"][row].bool().nonzero().flatten()
                prompt = prompts["attention_mask"][row].bool().nonzero().flatten()
                labels[row, full[len(prompt):]] = encoded["input_ids"][row, full[len(prompt):]]
            encoded["labels"] = labels
        return dict(encoded)


def vlm_loader(samples, collator, batch_size, workers, sampler=None):
    return DataLoader(
        VlmDataset(samples), batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=workers, collate_fn=collator, pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def qwen_val_loss(model, batches, device):
    model.eval()
    losses = []
    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(model(**batch).loss.float().item())
    return sum(losses) / len(losses)


def train_qwen(config: dict) -> None:
    labels = labels_from_config(config)
    label_set = set(labels)
    root = project_path(config["data"]["root"])
    boxes = bbox_records(project_path(config["data"]["bbox_annotations"]))
    train_config = config["train"]
    device = torch.device(config["runtime"]["device"])
    for protocol, seed in experiment_runs(config):
        values = {"protocol": protocol, "seed": seed}
        output_dir = project_path(config["output"]["root"], **values)
        adapter_dir = output_dir / "best"
        if config["output"].get("skip_existing") and (adapter_dir / "adapter_config.json").exists():
            print(f"[skip] {adapter_dir}")
            continue
        seed_everything(seed)
        train_samples = read_samples(root / protocol / "train.csv", root, include_labels=label_set)
        val_samples = read_samples(root / protocol / "val.csv", root, include_labels=label_set)
        model, processor = load_qwen(project_path(config["model"]["path"]))
        model.config.use_cache = False
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=train_config["lora_r"],
                lora_alpha=train_config["lora_alpha"],
                lora_dropout=train_config["lora_dropout"],
                target_modules=LORA_PATTERNS[train_config["lora_scope"]],
                bias="none", task_type="CAUSAL_LM",
            ),
        ).to(device)
        model.gradient_checkpointing_enable()
        train_collator = QwenCollator(
            processor, labels, root, boxes, config["representation"]["view"],
            config["representation"].get("context_margin", 0.0),
            config["prompt"], train_config["max_pixels"], True,
        )
        val_collator = QwenCollator(
            processor, labels, root, boxes, config["representation"]["view"],
            config["representation"].get("context_margin", 0.0),
            config["prompt"], train_config["max_pixels"], True,
        )
        sampler = class_sampler(
            [sample.label for sample in train_samples], train_config["class_balance_power"], seed
        )
        train_batches = vlm_loader(
            train_samples, train_collator, train_config["batch_size"],
            train_config["num_workers"], sampler
        )
        val_batches = vlm_loader(
            val_samples, val_collator, train_config["batch_size"],
            train_config["num_workers"]
        )
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=train_config["learning_rate"],
        )
        updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
        scheduler = cosine_scheduler(optimizer, updates * train_config["epochs"], train_config["warmup_ratio"])
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir / "tensorboard")
        best_loss = float("inf")
        global_step = 0
        for epoch in range(1, train_config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(
                tqdm(train_batches, desc=f"{config['experiment']} {protocol} epoch {epoch}"), 1
            ):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                (loss / train_config["gradient_accumulation"]).backward()
                if batch_index % train_config["gradient_accumulation"] == 0 or batch_index == len(train_batches):
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    writer.add_scalar("train/loss", loss.item(), global_step)
            val_loss = qwen_val_loss(model, val_batches, device)
            writer.add_scalar("epoch/val_loss", val_loss, epoch); writer.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                model.save_pretrained(adapter_dir)
                processor.save_pretrained(adapter_dir)
        writer.close()
        del model, optimizer
        torch.cuda.empty_cache()


def parse_label(text: str, labels: list[str]):
    value = text.strip()
    return value if value in set(labels) else None


def test_qwen(config: dict) -> None:
    labels = labels_from_config(config)
    label_set = set(labels)
    root = project_path(config["data"]["root"])
    boxes = bbox_records(project_path(config["data"]["bbox_annotations"]))
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            samples = read_private_test_samples(
                root / protocol / "test_inputs.csv", root / protocol / "test_labels_private.csv",
                root, include_labels=label_set
            )
            model, processor = load_qwen(project_path(config["model"]["path"]))
            model = PeftModel.from_pretrained(
                model, project_path(config["output"]["adapter"], **values), local_files_only=True
            ).to(device).eval()
            collator = QwenCollator(
                processor, labels, root, boxes, config["representation"]["view"],
                config["representation"].get("context_margin", 0.0),
                config["prompt"], config["test"]["max_pixels"], False,
            )
            batches = vlm_loader(samples, collator, config["test"]["batch_size"], 0)
            predictions, valid, raw_outputs = [], [], []
            with torch.inference_mode():
                for batch in tqdm(batches, desc=f"{config['experiment']} {protocol}"):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    generated = model.generate(**batch, do_sample=False, max_new_tokens=config["test"]["max_new_tokens"])
                    input_length = batch["input_ids"].shape[1]
                    texts = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)
                    for text in texts:
                        raw_outputs.append(text)
                        prediction = parse_label(text, labels)
                        predictions.append(prediction)
                        valid.append(prediction is not None)
            targets = [sample.label for sample in samples]
            metrics = table3_metrics(targets, predictions, valid, labels)
            save_predictions(
                project_path(config["output"]["results"], **values), config["experiment"],
                protocol, seed, samples, predictions, valid, metrics, raw_outputs
            )
            print(protocol, seed, metrics)
            del model
            torch.cuda.empty_cache()


class FlorenceCollator:
    def __init__(self, processor, labels, data_root, boxes, prompt, training):
        self.processor = processor
        self.labels = labels
        self.data_root = data_root
        self.boxes = boxes
        self.training = training
        self.prompt = prompt.format(categories=", ".join(labels))

    def __call__(self, samples):
        images = [render_view(sample, self.data_root, self.boxes, "gt_crop", 0.0) for sample in samples]
        batch = self.processor(text=[self.prompt] * len(samples), images=images, padding=True, return_tensors="pt")
        if self.training:
            targets = self.processor.tokenizer(
                [sample.label for sample in samples], padding=True, return_tensors="pt"
            )["input_ids"]
            targets[targets == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = targets
        return dict(batch)


@torch.inference_mode()
def florence_val_loss(model, batches, device):
    model.eval(); losses = []
    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            losses.append(model(**batch).loss.float().item())
    return sum(losses) / len(losses)


@contextmanager
def florence_legacy_transformers_compat():
    """Restore APIs used while the Microsoft Florence-2 remote code is constructed."""
    added_attributes = []
    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None
        added_attributes.append((PretrainedConfig, "forced_bos_token_id"))
    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
        PreTrainedTokenizerBase.additional_special_tokens = property(
            lambda tokenizer: list(tokenizer._extra_special_tokens)
        )
        added_attributes.append((PreTrainedTokenizerBase, "additional_special_tokens"))
    try:
        yield
    finally:
        for owner, name in reversed(added_attributes):
            delattr(owner, name)


def load_florence(path: Path, device, dtype=torch.float16):
    with florence_legacy_transformers_compat():
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="eager",
        )
        # Generation creates a fresh instance of this dynamically loaded config,
        # after the temporary base-class compatibility context has exited.
        language_config_class = type(model.language_model.config)
        if "forced_bos_token_id" not in language_config_class.__dict__:
            language_config_class.forced_bos_token_id = None
        # Transformers 5 no longer resolves this legacy checkpoint's nested
        # tied-weight paths, so restore the shared BART embeddings explicitly.
        shared_embeddings = model.language_model.model.shared
        model.language_model.model.encoder.embed_tokens = shared_embeddings
        model.language_model.model.decoder.embed_tokens = shared_embeddings
        model.language_model.lm_head.weight = shared_embeddings.weight
        model._tied_weights_keys = {
            "language_model.model.encoder.embed_tokens.weight": "language_model.model.shared.weight",
            "language_model.model.decoder.embed_tokens.weight": "language_model.model.shared.weight",
            "language_model.lm_head.weight": "language_model.model.shared.weight",
        }
        model.language_model._tied_weights_keys = {}
        model.language_model.model._tied_weights_keys = {}
        model = model.to(device)
    return model, processor


def train_florence(config: dict) -> None:
    labels = labels_from_config(config); label_set = set(labels)
    root = project_path(config["data"]["root"])
    boxes = bbox_records(project_path(config["data"]["bbox_annotations"]))
    train_config = config["train"]; device = torch.device(config["runtime"]["device"])
    for protocol, seed in experiment_runs(config):
        values = {"protocol": protocol, "seed": seed}
        output_dir = project_path(config["output"]["root"], **values); best_dir = output_dir / "best"
        if config["output"].get("skip_existing") and (best_dir / "config.json").exists():
            print(f"[skip] {best_dir}"); continue
        seed_everything(seed)
        train_samples = read_samples(root / protocol / "train.csv", root, include_labels=label_set)
        val_samples = read_samples(root / protocol / "val.csv", root, include_labels=label_set)
        model, processor = load_florence(project_path(config["model"]["path"]), device)
        train_collator = FlorenceCollator(processor, labels, root, boxes, config["prompt"], True)
        val_collator = FlorenceCollator(processor, labels, root, boxes, config["prompt"], True)
        sampler = class_sampler([s.label for s in train_samples], train_config["class_balance_power"], seed)
        train_batches = vlm_loader(train_samples, train_collator, train_config["batch_size"], train_config["num_workers"], sampler)
        val_batches = vlm_loader(val_samples, val_collator, train_config["batch_size"], train_config["num_workers"])
        optimizer = AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])
        updates = math.ceil(len(train_batches) / train_config["gradient_accumulation"])
        scheduler = cosine_scheduler(optimizer, updates * train_config["epochs"], train_config["warmup_ratio"])
        output_dir.mkdir(parents=True, exist_ok=True); writer = SummaryWriter(output_dir / "tensorboard")
        best_loss = float("inf"); global_step = 0
        for epoch in range(1, train_config["epochs"] + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(tqdm(train_batches, desc=f"{config['experiment']} epoch {epoch}"), 1):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16): loss = model(**batch).loss
                (loss / train_config["gradient_accumulation"]).backward()
                if batch_index % train_config["gradient_accumulation"] == 0 or batch_index == len(train_batches):
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    global_step += 1; writer.add_scalar("train/loss", loss.item(), global_step)
            val_loss = florence_val_loss(model, val_batches, device)
            writer.add_scalar("epoch/val_loss", val_loss, epoch); writer.flush()
            if val_loss < best_loss:
                best_loss = val_loss; model.save_pretrained(best_dir); processor.save_pretrained(best_dir)
        writer.close(); del model, optimizer; torch.cuda.empty_cache()


def test_florence(config: dict) -> None:
    labels = labels_from_config(config); label_set = set(labels)
    root = project_path(config["data"]["root"])
    boxes = bbox_records(project_path(config["data"]["bbox_annotations"]))
    device = torch.device(config["runtime"]["device"])
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            samples = read_private_test_samples(
                root / protocol / "test_inputs.csv", root / protocol / "test_labels_private.csv",
                root, include_labels=label_set
            )
            model, processor = load_florence(project_path(config["output"]["model"], **values), device)
            collator = FlorenceCollator(processor, labels, root, boxes, config["prompt"], False)
            batches = vlm_loader(samples, collator, config["test"]["batch_size"], 0)
            predictions, valid, raw_outputs = [], [], []
            with torch.inference_mode():
                for batch in tqdm(batches, desc=f"{config['experiment']} {protocol}"):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast("cuda", dtype=torch.float16):
                        generated = model.generate(
                            **batch,
                            max_new_tokens=config["test"]["max_new_tokens"],
                            do_sample=False,
                            use_cache=False,
                        )
                    for text in processor.batch_decode(generated, skip_special_tokens=True):
                        raw_outputs.append(text)
                        prediction = parse_label(text, labels); predictions.append(prediction); valid.append(prediction is not None)
            targets = [sample.label for sample in samples]
            metrics = table3_metrics(targets, predictions, valid, labels)
            save_predictions(
                project_path(config["output"]["results"], **values), config["experiment"],
                protocol, seed, samples, predictions, valid, metrics, raw_outputs
            )
            print(protocol, seed, metrics); del model; torch.cuda.empty_cache()
