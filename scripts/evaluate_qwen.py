#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.auto import tqdm

from clear_uav.data import cap_per_class, read_private_test_samples, read_samples
from clear_uav.metrics import classification_metrics, group_sets
from clear_uav.modeling import load_qwen
from clear_uav.ontology import load_label_subset, load_ontology
from clear_uav.prompts import conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Qwen3-VL generation evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-labels", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--view", choices=["context", "evidence", "pair"], default="pair")
    parser.add_argument(
        "--prompt", choices=["adapted", "direct", "definition"], default="adapted"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    return parser.parse_args()


def parse_events(text: str, valid_labels: set[str]) -> set[str]:
    payload = json.loads(text)
    events = payload["events"]
    if not isinstance(events, list) or not all(isinstance(item, str) for item in events):
        raise ValueError("events must be a list of strings")
    unknown = set(events) - valid_labels
    if unknown:
        raise ValueError(f"Unknown predicted labels: {sorted(unknown)}")
    return set(events)


def main() -> None:
    args = parse_args()
    ontology = load_ontology(args.ontology)
    valid_labels = set(
        load_label_subset(args.labels_file, ontology)
        if args.labels_file
        else ontology.labels
    )
    if args.private_labels:
        samples = read_private_test_samples(
            args.csv,
            args.private_labels,
            args.data_root,
            limit=args.max_samples,
            include_labels=valid_labels,
        )
    else:
        samples = read_samples(
            args.csv,
            args.data_root,
            limit=args.max_samples,
            include_labels=valid_labels,
        )
    if args.max_per_class:
        samples = cap_per_class(samples, args.max_per_class, seed=0)

    model, processor = load_qwen(args.model_path, device_map="auto")
    if args.adapter_path:
        adapter_path = args.adapter_path.resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing local adapter_config.json: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()

    output_format = (
        'Return exactly {"events":["canonical_event"]}. Use only names from the allowed '
        'list, use {"events":[]} when none is supported, and return no other keys or text.'
    )
    if args.prompt == "direct":
        instruction = (
            output_format
            + "\nAllowed canonical events: "
            + ", ".join(sorted(valid_labels))
        )
    elif args.prompt == "definition":
        instruction = (
            output_format
            + "\nAllowed canonical events and definitions:\n"
            + "\n".join(
                f"- {label}: {ontology.definitions[label]}"
                for label in sorted(valid_labels)
            )
        )
    else:
        instruction = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[set[str]] = []
    invalid = 0
    with args.output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for sample in tqdm(
            samples,
            desc=f"free-generation {args.view}/{args.prompt}",
            unit="pair",
            dynamic_ncols=True,
            mininterval=0.5,
        ):
            inputs = processor.apply_chat_template(
                conversation(
                    sample.context_path,
                    sample.evidence_path,
                    args.view,
                    instruction=instruction,
                ),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "size": {
                        "longest_edge": args.max_pixels,
                        "shortest_edge": min(65_536, args.max_pixels),
                    }
                },
            ).to(model.device)
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
            )
            text = processor.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            try:
                prediction = parse_events(text, valid_labels)
                error = None
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                prediction = set()
                error = str(exc)
                invalid += 1
            predictions.append(prediction)
            handle.write(
                json.dumps(
                    {
                        "record_uid": sample.record_uid,
                        "target": sample.label,
                        "prediction": sorted(prediction),
                        "raw": text,
                        "parse_error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    evaluated_labels = sorted({sample.label for sample in samples})
    pair_targets = [{sample.label} for sample in samples]
    pair_metrics = classification_metrics(pair_targets, predictions, evaluated_labels)
    set_targets, set_predictions = group_sets(
        [sample.content_group_id for sample in samples],
        [sample.label for sample in samples],
        predictions,
    )
    set_metrics = classification_metrics(set_targets, set_predictions, evaluated_labels)
    result = {
        "config": {
            "model_path": str(args.model_path.resolve()),
            "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
            "prompt": args.prompt,
            "view": args.view,
            "labels": sorted(valid_labels),
            "max_pixels": args.max_pixels,
        },
        "num_pairs": len(samples),
        "invalid_outputs": invalid,
        "pair": pair_metrics,
        "set_union": set_metrics,
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
