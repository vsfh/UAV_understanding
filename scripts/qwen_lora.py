#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, set_seed

from clear_uav.data import cap_per_class, read_samples
from clear_uav.experiment_config import experiment_runs, load_yaml, project_path
from clear_uav.modeling import LORA_PATTERNS, load_qwen
from clear_uav.ontology import load_label_subset, load_ontology
from clear_uav.supervision import read_targets
from clear_uav.training import ClearCollator, ClearTrainer


def train_run(config: dict, protocol: str, seed: int) -> None:
    values = {"protocol": protocol, "seed": seed}
    output_dir = project_path(config["output"]["root"], **values)
    final_adapter = output_dir / "final/adapter_config.json"
    if config["output"].get("skip_existing") and final_adapter.exists():
        print(f"[skip] {output_dir}")
        return

    data_root = project_path(config["data"]["root"])
    ontology = load_ontology(project_path(config["data"]["ontology"]))
    labels = set(
        load_label_subset(project_path(config["data"]["labels"]), ontology)
    )
    samples = cap_per_class(
        read_samples(
            data_root / protocol / "train.csv", data_root, include_labels=labels
        ),
        config["data"]["max_per_class"],
        seed,
    )
    train = config["train"]
    targets = counterfactuals = None
    target_pattern = config["data"].get("targets")
    if target_pattern:
        targets, counterfactuals, _ = read_targets(
            project_path(target_pattern, **values)
        )

    set_seed(seed)
    model, processor = load_qwen(project_path(config["model"]["path"]))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=train["lora_r"],
            lora_alpha=train["lora_alpha"],
            lora_dropout=train["lora_dropout"],
            target_modules=LORA_PATTERNS[train["lora_scope"]],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    collator = ClearCollator(
        processor=processor,
        ontology=ontology,
        view=train["view"],
        max_length=train["max_length"],
        max_pixels=train["max_pixels"],
        label_weight=train["label_weight"],
        targets=targets,
        counterfactual_targets=counterfactuals,
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir / "tensorboard"),
        num_train_epochs=train["epochs"],
        learning_rate=train["learning_rate"],
        per_device_train_batch_size=train["batch_size"],
        gradient_accumulation_steps=train["gradient_accumulation"],
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        logging_steps=train["logging_steps"],
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=seed,
        data_seed=seed,
    )
    trainer = ClearTrainer(
        model=model,
        args=arguments,
        train_dataset=samples,
        data_collator=collator,
        processing_class=processor,
        margin=0.2,
        lambda_neighbor=0.0,
        lambda_cf=0.0,
    )
    result = trainer.train()
    trainer.save_model(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    (output_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    del trainer, model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with LoRA")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    for protocol, seed in experiment_runs(config):
        train_run(config, protocol, seed)


if __name__ == "__main__":
    main()
