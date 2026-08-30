from __future__ import annotations

import json
import math
from collections import Counter

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from clear_uav.data import read_private_test_samples, read_samples
from clear_uav.experiment_config import experiment_runs, project_path
from clear_uav.modeling import LORA_PATTERNS, assistant_only_labels, load_qwen
from clear_uav.table3 import (
    QwenCollator,
    bbox_records,
    coordinate_text,
    cosine_scheduler,
    ensure_hf_model,
    labels_from_config,
    parse_label,
    render_view,
    save_predictions,
    seed_everything,
    table3_metrics,
    vlm_loader,
)


class MixedViewDataset(Dataset):
    def __init__(self, samples, representations: list[dict]) -> None:
        self.samples = samples
        self.representations = representations
        self.labels = [sample.label for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: tuple[int, int]) -> dict:
        sample_index, view_index = index
        return {
            "sample": self.samples[sample_index],
            "representation": self.representations[view_index],
        }


class BalancedViewSampler(Sampler):
    def __init__(
        self,
        labels: list[str],
        num_views: int,
        class_balance_power: float,
        num_samples: int,
        seed: int,
    ) -> None:
        counts = Counter(labels)
        self.weights = torch.tensor(
            [counts[label] ** (-class_balance_power) for label in labels],
            dtype=torch.double,
        )
        self.num_views = num_views
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sample_indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        view_indices = torch.arange(self.num_samples) % self.num_views
        view_indices = view_indices[
            torch.randperm(self.num_samples, generator=generator)
        ]
        return iter(zip(sample_indices.tolist(), view_indices.tolist()))

    def __len__(self) -> int:
        return self.num_samples


class MixedViewQwenCollator:
    def __init__(self, processor, labels, data_root, boxes, max_pixels) -> None:
        self.processor = processor
        self.labels = labels
        self.data_root = data_root
        self.boxes = boxes
        self.max_pixels = max_pixels

    def messages(self, row: dict):
        sample = row["sample"]
        representation = row["representation"]
        image = render_view(
            sample,
            self.data_root,
            self.boxes,
            representation["view"],
            representation.get("context_margin", 0.0),
        )
        coordinates = ""
        if "{coordinates}" in representation["prompt"]["user"]:
            coordinates = coordinate_text(sample, self.data_root, self.boxes)
        instruction = representation["prompt"]["user"].format(
            categories=", ".join(self.labels),
            coordinates=coordinates,
        )
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": representation["prompt"]["system"]}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.label}],
            },
        ]

    def __call__(self, rows: list[dict]) -> dict:
        conversations = [self.messages(row) for row in rows]
        encoded = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "size": {
                    "longest_edge": self.max_pixels,
                    "shortest_edge": min(65536, self.max_pixels),
                },
            },
        )
        encoded["labels"] = assistant_only_labels(
            encoded["input_ids"],
            encoded["attention_mask"],
            self.processor.tokenizer,
        )
        return dict(encoded)


@torch.inference_mode()
def predict_view(
    model,
    processor,
    samples,
    representation,
    labels,
    data_root,
    boxes,
    evaluation_config,
    device,
    description,
):
    model.eval()
    model.config.use_cache = True
    collator = QwenCollator(
        processor,
        labels,
        data_root,
        boxes,
        representation["view"],
        representation.get("context_margin", 0.0),
        representation["prompt"],
        evaluation_config["max_pixels"],
        False,
    )
    batches = vlm_loader(
        samples,
        collator,
        evaluation_config["batch_size"],
        evaluation_config["num_workers"],
    )
    predictions, valid, raw_outputs = [], [], []
    for batch in tqdm(batches, desc=description):
        batch = {key: value.to(device) for key, value in batch.items()}
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=evaluation_config["max_new_tokens"],
        )
        input_length = batch["input_ids"].shape[1]
        texts = processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True
        )
        for text in texts:
            raw_outputs.append(text)
            prediction = parse_label(text, labels)
            predictions.append(prediction)
            valid.append(prediction is not None)
    targets = [sample.label for sample in samples]
    metrics = table3_metrics(targets, predictions, valid, labels)
    model.config.use_cache = False
    return predictions, valid, raw_outputs, metrics


def train_qwen_shared(config: dict) -> None:
    ensure_hf_model(config["model"])
    labels = labels_from_config(config)
    label_set = set(labels)
    data_root = project_path(config["data"]["root"])
    boxes = bbox_records(project_path(config["data"]["bbox_annotations"]))
    train_config = config["train"]
    device = torch.device(config["runtime"]["device"])

    for protocol, seed in experiment_runs(config):
        values = {"protocol": protocol, "seed": seed}
        output_dir = project_path(config["output"]["root"], **values)
        adapter_dir = project_path(config["output"]["adapter"], **values)
        validation_path = project_path(config["output"]["validation"], **values)
        if (
            config["output"].get("skip_existing")
            and (adapter_dir / "adapter_config.json").exists()
            and validation_path.exists()
        ):
            print(f"[skip] {adapter_dir}")
            continue

        seed_everything(seed)
        train_samples = read_samples(
            data_root / protocol / "train.csv", data_root, include_labels=label_set
        )
        val_samples = read_samples(
            data_root / protocol / "val.csv", data_root, include_labels=label_set
        )
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
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ).to(device)
        model.gradient_checkpointing_enable()

        dataset = MixedViewDataset(train_samples, config["representations"])
        sampler = BalancedViewSampler(
            dataset.labels,
            len(config["representations"]),
            train_config["class_balance_power"],
            train_config.get("samples_per_epoch") or len(dataset),
            seed,
        )
        train_batches = DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            sampler=sampler,
            num_workers=train_config["num_workers"],
            collate_fn=MixedViewQwenCollator(
                processor,
                labels,
                data_root,
                boxes,
                train_config["max_pixels"],
            ),
            pin_memory=True,
            persistent_workers=train_config["num_workers"] > 0,
        )
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = AdamW(
            parameters,
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )
        updates = math.ceil(
            len(train_batches) / train_config["gradient_accumulation"]
        )
        scheduler = cosine_scheduler(
            optimizer,
            updates * train_config["epochs"],
            train_config["warmup_ratio"],
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir / "tensorboard")
        best_score = -1.0
        best_summary = None
        global_step = 0

        for epoch in range(1, train_config["epochs"] + 1):
            sampler.set_epoch(epoch)
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for batch_index, batch in enumerate(
                tqdm(
                    train_batches,
                    desc=f"{config['experiment']} {protocol} epoch {epoch}",
                ),
                1,
            ):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                (loss / train_config["gradient_accumulation"]).backward()
                running_loss += loss.item()
                if (
                    batch_index % train_config["gradient_accumulation"] == 0
                    or batch_index == len(train_batches)
                ):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    writer.add_scalar("train/loss", loss.item(), global_step)

            per_view = {}
            for representation in config["representations"]:
                _, _, _, metrics = predict_view(
                    model,
                    processor,
                    val_samples,
                    representation,
                    labels,
                    data_root,
                    boxes,
                    config["validation"],
                    device,
                    f"validation {representation['name']} epoch {epoch}",
                )
                per_view[representation["name"]] = metrics
                writer.add_scalar(
                    f"validation/{representation['name']}_macro_f1",
                    metrics["macro_f1"],
                    epoch,
                )
            composite = sum(
                metrics["macro_f1"] for metrics in per_view.values()
            ) / len(per_view)
            epoch_loss = running_loss / len(train_batches)
            writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
            writer.add_scalar("validation/mean_macro_f1", composite, epoch)
            writer.flush()
            print(
                f"{protocol} seed{seed} epoch {epoch}: "
                f"train_loss={epoch_loss:.6f}, mean_macro_f1={composite:.6f}"
            )

            if composite > best_score:
                best_score = composite
                model.save_pretrained(adapter_dir)
                processor.save_pretrained(adapter_dir)
                best_summary = {
                    "selection_metric": "mean_macro_f1",
                    "best_epoch": epoch,
                    "mean_macro_f1": composite,
                    "per_view": per_view,
                }
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                validation_path.write_text(
                    json.dumps(best_summary, indent=2), encoding="utf-8"
                )

        writer.close()
        print(f"selected checkpoint: {best_summary}")
        del model, optimizer
        torch.cuda.empty_cache()


def test_qwen_shared(config: dict, view_names: list[str] | None = None) -> None:
    labels = labels_from_config(config)
    label_set = set(labels)
    data_root = project_path(config["data"]["root"])
    boxes = bbox_records(project_path(config["data"]["bbox_annotations"]))
    device = torch.device(config["runtime"]["device"])
    representations = config["representations"]
    if view_names is not None:
        selected = set(view_names)
        representations = [row for row in representations if row["name"] in selected]

    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            values = {"protocol": protocol, "seed": seed}
            samples = read_private_test_samples(
                data_root / protocol / "test_inputs.csv",
                data_root / protocol / "test_labels_private.csv",
                data_root,
                include_labels=label_set,
            )
            model, processor = load_qwen(project_path(config["model"]["path"]))
            model = PeftModel.from_pretrained(
                model,
                project_path(config["output"]["adapter"], **values),
                local_files_only=True,
            ).to(device).eval()

            for representation in representations:
                predictions, valid, raw_outputs, metrics = predict_view(
                    model,
                    processor,
                    samples,
                    representation,
                    labels,
                    data_root,
                    boxes,
                    config["test"],
                    device,
                    f"test {representation['name']} {protocol}",
                )
                experiment = f"table3_qwen3vl_{representation['name']}"
                result_path = project_path(
                    config["output"]["results"],
                    view=representation["name"],
                    **values,
                )
                save_predictions(
                    result_path,
                    experiment,
                    protocol,
                    seed,
                    samples,
                    predictions,
                    valid,
                    metrics,
                    raw_outputs,
                )
                print(protocol, seed, representation["name"], metrics)

            del model
            torch.cuda.empty_cache()
