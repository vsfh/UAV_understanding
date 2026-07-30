#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, set_seed

from clear_uav.data import cap_per_class, read_samples
from clear_uav.modeling import LORA_PATTERNS, load_qwen
from clear_uav.ontology import load_label_subset, load_ontology
from clear_uav.training import ClearCollator, ClearTrainer


def jsonable_arguments(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA training for Qwen3-VL CLEAR-UAV")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--targets-jsonl", type=Path)
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--require-audited-targets", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=["context", "evidence", "pair"], default="pair")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-scope",
        choices=LORA_PATTERNS,
        default="projector_llm",
        help="Keep the vision tower frozen; optionally adapt the visual mergers",
    )
    parser.add_argument("--label-weight", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda-neighbor", type=float, default=0.0)
    parser.add_argument("--lambda-cf", type=float, default=0.0)
    parser.add_argument("--random-negative", action="store_true")
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--evidence-dropout", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def read_targets(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    targets = {}
    counterfactuals = {}
    tiers = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            uid = row["record_uid"]
            target = row["target"]
            counterfactual = row["counterfactual_target"]
            for name, value in (("target", target), ("counterfactual_target", counterfactual)):
                parsed = json.loads(value) if isinstance(value, str) else value
                if set(parsed) != {"events", "factors", "evidence", "uncertain"}:
                    raise ValueError(f"Invalid {name} schema at {path}:{line_number}")
            if uid in targets:
                raise ValueError(f"Duplicate record_uid in {path}: {uid}")
            tiers[uid] = row.get("supervision_tier", "unspecified")
            targets[uid] = json.dumps(
                json.loads(target) if isinstance(target, str) else target,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            counterfactuals[uid] = json.dumps(
                json.loads(counterfactual) if isinstance(counterfactual, str) else counterfactual,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
    return targets, counterfactuals, tiers


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ontology = load_ontology(args.ontology)
    included_labels = (
        set(load_label_subset(args.labels_file, ontology)) if args.labels_file else None
    )
    samples = read_samples(
        args.train_csv,
        args.data_root,
        limit=args.max_samples,
        include_labels=included_labels,
    )
    if args.max_per_class:
        samples = cap_per_class(samples, args.max_per_class, args.seed)
    unknown = sorted({sample.label for sample in samples} - set(ontology.labels))
    if unknown:
        raise ValueError(f"Training CSV contains labels absent from ontology: {unknown}")
    targets = counterfactuals = tiers = None
    if args.targets_jsonl:
        targets, counterfactuals, tiers = read_targets(args.targets_jsonl)
        missing_targets = {sample.record_uid for sample in samples} - set(targets)
        if missing_targets:
            raise ValueError(
                f"Audited target file is missing {len(missing_targets)} training records; "
                f"first={sorted(missing_targets)[0]}"
            )
        wrong_targets = [
            sample.record_uid
            for sample in samples
            if sample.label not in json.loads(targets[sample.record_uid])["events"]
        ]
        if wrong_targets:
            raise ValueError(f"Audited target omits its verified label: {wrong_targets[0]}")
    if args.require_audited_targets:
        if targets is None:
            raise ValueError("--require-audited-targets requires --targets-jsonl")
        non_audited = [uid for uid, tier in tiers.items() if tier != "human_audited"]
        if non_audited:
            raise ValueError(
                "--require-audited-targets found non-audited supervision: "
                f"{non_audited[0]} ({tiers[non_audited[0]]})"
            )

    model, processor = load_qwen(args.model_path)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_PATTERNS[args.lora_scope],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    collator = ClearCollator(
        processor=processor,
        ontology=ontology,
        view=args.view,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        label_weight=args.label_weight,
        neighbor_loss=bool(args.lambda_neighbor),
        counterfactual_loss=bool(args.lambda_cf),
        random_negative=args.random_negative,
        random_negative_pool=tuple(
            label
            for label, count in Counter(sample.label for sample in samples).items()
            for _ in range(count)
        ),
        context_dropout=args.context_dropout,
        evidence_dropout=args.evidence_dropout,
        targets=targets,
        counterfactual_targets=counterfactuals,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = ClearTrainer(
        model=model,
        args=training_args,
        train_dataset=samples,
        data_collator=collator,
        processing_class=processor,
        margin=args.margin,
        lambda_neighbor=args.lambda_neighbor,
        lambda_cf=args.lambda_cf,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_model(args.output_dir / "final")
    processor.save_pretrained(args.output_dir / "final")
    metadata = {
        "arguments": jsonable_arguments(args),
        "num_training_samples": len(samples),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "train_metrics": train_result.metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
