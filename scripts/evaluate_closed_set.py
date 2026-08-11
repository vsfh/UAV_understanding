#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.auto import tqdm

from clear_uav.data import cap_per_class, read_private_test_samples, read_samples
from clear_uav.metrics import classification_metrics, pairwise_metrics, ranking_metrics
from clear_uav.modeling import load_qwen
from clear_uav.ontology import load_label_subset, load_ontology
from clear_uav.prompts import closed_set_conversation
from clear_uav.training import (
    aligned_normalized_log_likelihood,
    encode_assistant_batch,
    forward_answer_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-set normalized-likelihood evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-labels", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--fit-thresholds", action="store_true")
    parser.add_argument("--view", choices=["context", "evidence", "pair"], default="pair")
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--set-temperature", type=float, default=1.0)
    parser.add_argument(
        "--set-aggregator", choices=["logsumexp", "max"], default="logsumexp"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def best_threshold(scores: list[float], targets: list[bool]) -> float:
    candidates = [max(scores) + 1e-6, *sorted(set(scores), reverse=True)]
    best = (-1.0, candidates[0])
    for threshold in candidates:
        tp = sum(score >= threshold and target for score, target in zip(scores, targets))
        fp = sum(score >= threshold and not target for score, target in zip(scores, targets))
        fn = sum(score < threshold and target for score, target in zip(scores, targets))
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        if f1 > best[0]:
            best = (f1, threshold)
    return best[1]


def fit_thresholds(
    score_rows: list[dict[str, float]], targets: list[set[str]], labels: list[str]
) -> dict[str, float]:
    return {
        label: best_threshold(
            [row[label] for row in score_rows], [label in target for target in targets]
        )
        for label in labels
    }


def predict(score_rows: list[dict[str, float]], thresholds: dict[str, float]) -> list[set[str]]:
    return [
        {label for label, score in row.items() if score >= thresholds[label]}
        for row in score_rows
    ]


def aggregate_sets(samples, score_rows, labels, temperature, aggregator):
    grouped_scores = defaultdict(lambda: defaultdict(list))
    grouped_targets = defaultdict(set)
    for sample, row in zip(samples, score_rows):
        grouped_targets[sample.content_group_id].add(sample.label)
        for label in labels:
            grouped_scores[sample.content_group_id][label].append(row[label])
    group_ids = sorted(grouped_targets)
    set_rows = []
    for group_id in group_ids:
        if aggregator == "logsumexp":
            set_rows.append(
                {
                    label: float(
                        temperature
                        * torch.logsumexp(
                            torch.tensor(grouped_scores[group_id][label]) / temperature,
                            dim=0,
                        )
                    )
                    for label in labels
                }
            )
        elif aggregator == "max":
            set_rows.append(
                {
                    label: max(grouped_scores[group_id][label])
                    for label in labels
                }
            )
        else:
            raise ValueError(f"Unknown set aggregator: {aggregator}")
    return group_ids, set_rows, [grouped_targets[group_id] for group_id in group_ids]


def main() -> None:
    args = parse_args()
    if args.candidate_batch_size < 1:
        raise ValueError("--candidate-batch-size must be positive")
    if args.fit_thresholds == bool(args.thresholds):
        raise ValueError("Choose exactly one of --fit-thresholds or --thresholds")
    ontology = load_ontology(args.ontology)
    labels = list(
        load_label_subset(args.labels_file, ontology)
        if args.labels_file
        else ontology.labels
    )
    included_labels = set(labels)
    samples = (
        read_private_test_samples(
            args.csv,
            args.private_labels,
            args.data_root,
            limit=args.max_samples,
            include_labels=included_labels,
        )
        if args.private_labels
        else read_samples(
            args.csv,
            args.data_root,
            limit=args.max_samples,
            include_labels=included_labels,
        )
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
    model.config.use_cache = False
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        if (
            args.view == "pair"
            and total_bytes <= 52 * 2**30
            and args.candidate_batch_size > 4
        ):
            raise ValueError(
                "Pair-view closed-set evaluation on a <=52 GiB GPU requires "
                "--candidate-batch-size <= 4; use 2 for the 49140 MiB card"
            )
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device)
        print(
            f"GPU memory after model load: {free_bytes / 2**30:.1f} GiB free / "
            f"{total_bytes / 2**30:.1f} GiB total; candidate batch="
            f"{args.candidate_batch_size}, view={args.view}",
            flush=True,
        )

    score_rows = []
    batches_per_pair = math.ceil(len(labels) / args.candidate_batch_size)
    with torch.inference_mode(), tqdm(
        total=len(samples) * batches_per_pair,
        desc=f"closed-set {args.view}",
        unit="cand-batch",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as progress:
        for sample_index, sample in enumerate(samples, 1):
            scores = {}
            for start in range(0, len(labels), args.candidate_batch_size):
                batch_labels = labels[start : start + args.candidate_batch_size]
                conversations = [
                    closed_set_conversation(
                        sample.context_path,
                        sample.evidence_path,
                        args.view,
                        label=label,
                        definition=ontology.definitions[label],
                    )
                    for label in batch_labels
                ]
                encoded = encode_assistant_batch(
                    processor,
                    conversations,
                    max_length=args.max_length,
                    max_pixels=args.max_pixels,
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                outputs, answer_labels, _ = forward_answer_logits(model, encoded)
                values = aligned_normalized_log_likelihood(
                    outputs.logits, answer_labels
                ).tolist()
                scores.update(zip(batch_labels, values))
                postfix = {
                    "pair": f"{sample_index}/{len(samples)}",
                    "labels": f"{min(start + len(batch_labels), len(labels))}/{len(labels)}",
                }
                if torch.cuda.is_available():
                    postfix["GPU"] = (
                        f"{torch.cuda.memory_allocated() / 2**30:.1f}G alloc/"
                        f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G peak"
                    )
                progress.set_postfix(postfix, refresh=False)
                progress.update(1)
                del encoded, outputs, answer_labels
            score_rows.append(scores)

    pair_targets = [{sample.label} for sample in samples]
    group_ids, set_rows, set_targets = aggregate_sets(
        samples, score_rows, labels, args.set_temperature, args.set_aggregator
    )
    if args.fit_thresholds:
        thresholds = {
            "pair": fit_thresholds(score_rows, pair_targets, labels),
            "set": fit_thresholds(set_rows, set_targets, labels),
        }
        threshold_path = args.output.with_suffix(".thresholds.json")
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(
            json.dumps(
                {
                    "set_aggregator": args.set_aggregator,
                    "set_temperature": args.set_temperature,
                    "thresholds": thresholds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if threshold_payload["set_aggregator"] != args.set_aggregator:
            raise ValueError("Set aggregator differs from the validation threshold fit")
        if threshold_payload["set_temperature"] != args.set_temperature:
            raise ValueError("Set temperature differs from the validation threshold fit")
        thresholds = threshold_payload["thresholds"]

    evaluated_labels = sorted({sample.label for sample in samples})
    pair_predictions = predict(score_rows, thresholds["pair"])
    set_predictions = predict(set_rows, thresholds["set"])
    pair_metrics = classification_metrics(pair_targets, pair_predictions, evaluated_labels)
    pair_metrics.update(ranking_metrics(pair_targets, score_rows, evaluated_labels))
    pair_metrics.update(
        pairwise_metrics(
            pair_targets,
            score_rows,
            {label: ontology.neighbors(label) for label in evaluated_labels},
        )
    )
    set_metrics = classification_metrics(set_targets, set_predictions, evaluated_labels)
    set_metrics.update(ranking_metrics(set_targets, set_rows, evaluated_labels))
    union_by_group = defaultdict(set)
    for sample, prediction in zip(samples, pair_predictions):
        union_by_group[sample.content_group_id].update(prediction)
    union_metrics = classification_metrics(
        set_targets,
        [union_by_group[group_id] for group_id in group_ids],
        evaluated_labels,
    )
    result = {
        "num_pairs": len(samples),
        "num_sets": len(set_targets),
        "pair": pair_metrics,
        "independent_union": union_metrics,
        "set": set_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    "model_path": str(args.model_path.resolve()),
                    "adapter_path": (
                        str(args.adapter_path.resolve()) if args.adapter_path else None
                    ),
                    "view": args.view,
                    "labels": labels,
                    "candidate_batch_size": args.candidate_batch_size,
                    "max_length": args.max_length,
                    "set_temperature": args.set_temperature,
                    "set_aggregator": args.set_aggregator,
                    "max_pixels": args.max_pixels,
                },
                "metrics": result,
                "labels": labels,
                "thresholds": thresholds,
                "predictions": [
                    {
                        "record_uid": sample.record_uid,
                        "group_id": sample.content_group_id,
                        "target": sample.label,
                        "prediction": sorted(prediction),
                        "scores": scores,
                    }
                    for sample, prediction, scores in zip(
                        samples, pair_predictions, score_rows
                    )
                ],
                "sets": [
                    {
                        "group_id": group_id,
                        "targets": sorted(target),
                        "predictions": sorted(prediction),
                        "scores": scores,
                    }
                    for group_id, target, prediction, scores in zip(
                        group_ids, set_targets, set_predictions, set_rows
                    )
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
