#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from clear_uav.data import Sample, cap_per_class, read_samples
from clear_uav.ontology import load_label_subset, load_ontology
from clear_uav.teacher_targets import (
    PROMPT_VERSION,
    atomic_write_json,
    automatic_audit,
    canonical_json,
    choose_neighbor,
    counterfactual_statement,
    parse_json_object,
    perception_messages,
    rewrite_messages,
    validate_perception,
    validate_rewrite,
    validate_verification,
    verification_messages,
)


DEFAULT_TEACHER = Path(
    "/media/4tb/feihong/hf_cache/"
    "models--Qwen--Qwen3.6-35B-A3B-FP8/"
    "snapshots/95a723d08a9490559dae23d0cff1d9466213d989"
)
DEFAULT_FP8_KERNEL = Path(
    "/home/feihong/.cache/huggingface/hub/"
    "kernels--kernels-community--finegrained-fp8/"
    "snapshots/13d2d7021a8854a5b767daf6513875ab9eb6c09d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate label-agnostic and event-grounded UAV training descriptions with a "
            "local teacher VLM. Outputs remain pending human review."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument(
        "--fp8-kernel-path",
        type=Path,
        default=DEFAULT_FP8_KERNEL,
        help="Pinned local kernels-community/finegrained-fp8 version-1 snapshot",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument(
        "--labels-file", type=Path, default=Path("configs/core18_complete.txt")
    )
    parser.add_argument("--evidence-cards-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Generate the union of deterministic capped samples selected by these seeds",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--max-new-tokens-perception", type=int, default=512)
    parser.add_argument("--max-new-tokens-rewrite", type=int, default=640)
    parser.add_argument("--max-new-tokens-verification", type=int, default=384)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--positive-threshold", type=float, default=0.65)
    parser.add_argument("--counterfactual-threshold", type=float, default=0.50)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence_cards(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            uid = row["record_uid"]
            if uid in result:
                raise ValueError(f"Duplicate evidence card at {path}:{line_number}: {uid}")
            factors = row.get("verified_factors", row.get("factors"))
            if not isinstance(factors, dict):
                raise ValueError(
                    f"Evidence card factors must be an object at {path}:{line_number}"
                )
            result[uid] = factors
    return result


def configuration(args: argparse.Namespace, sample_count: int) -> dict[str, Any]:
    config_path = args.model_path / "config.json"
    return {
        "prompt_version": PROMPT_VERSION,
        "teacher_model_path": str(args.model_path.resolve()),
        "teacher_config_sha256": sha256_file(config_path),
        "fp8_kernel_path": str(args.fp8_kernel_path.resolve()),
        "fp8_kernel_metadata_sha256": sha256_file(
            args.fp8_kernel_path / "build/torch-cuda/metadata.json"
        ),
        "data_root": str(args.data_root.resolve()),
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": sha256_file(args.train_csv),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "labels_file": str(args.labels_file.resolve()),
        "labels_file_sha256": sha256_file(args.labels_file),
        "evidence_cards_jsonl": (
            str(args.evidence_cards_jsonl.resolve())
            if args.evidence_cards_jsonl is not None
            else None
        ),
        "evidence_cards_sha256": (
            sha256_file(args.evidence_cards_jsonl)
            if args.evidence_cards_jsonl is not None
            else None
        ),
        "max_per_class": args.max_per_class,
        "seeds": args.seeds,
        "max_samples": args.max_samples,
        "sample_count": sample_count,
        "max_pixels": args.max_pixels,
        "max_new_tokens": {
            "perception": args.max_new_tokens_perception,
            "rewrite": args.max_new_tokens_rewrite,
            "verification": args.max_new_tokens_verification,
        },
        "positive_threshold": args.positive_threshold,
        "counterfactual_threshold": args.counterfactual_threshold,
        "generation": {
            "do_sample": False,
            "enable_thinking": False,
            "max_retries": args.max_retries,
        },
    }


def fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def check_or_write_manifest(args: argparse.Namespace, config: dict[str, Any]) -> None:
    manifest_path = args.output_dir / "generation_manifest.json"
    manifest = {
        "configuration_fingerprint": fingerprint(config),
        "configuration": config,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_fingerprint") != manifest["configuration_fingerprint"]:
            raise ValueError(
                f"{manifest_path} belongs to a different generation configuration; "
                "use a new --output-dir"
            )
        return
    atomic_write_json(manifest_path, manifest)


def model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine a materialized model input device")


class LocalTeacher:
    def __init__(self, args: argparse.Namespace):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["LOCAL_KERNELS"] = (
            f"kernels-community/finegrained-fp8={args.fp8_kernel_path.resolve()}"
        )
        if not torch.cuda.is_available() and not args.allow_cpu:
            raise RuntimeError(
                "CUDA is unavailable. Qwen3.6-35B generation is fail-closed by default; "
                "use --allow-cpu only if intentional."
            )
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
        )
        load_kwargs: dict[str, Any] = {
            "dtype": "auto",
            "device_map": args.device_map,
            "local_files_only": True,
        }
        if args.attn_implementation is not None:
            load_kwargs["attn_implementation"] = args.attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            **load_kwargs,
        )
        self.model.eval()
        self.input_device = model_input_device(self.model)
        self.max_pixels = args.max_pixels
        self.max_retries = args.max_retries

    @torch.inference_mode()
    def _generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        image_size = {
            "longest_edge": self.max_pixels,
            "shortest_edge": min(65_536, self.max_pixels),
        }
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={"size": image_size},
        )
        inputs = {
            key: value.to(self.input_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        validator,
        validator_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, int]:
        current_messages = messages
        attempts = []
        validator_kwargs = validator_kwargs or {}
        for attempt in range(self.max_retries + 1):
            raw = self._generate(current_messages, max_new_tokens)
            try:
                parsed = parse_json_object(raw)
                validator(parsed, **validator_kwargs)
                return parsed, raw, attempt
            except (KeyError, TypeError, ValueError) as error:
                attempts.append({"attempt": attempt, "raw": raw, "error": str(error)})
                if attempt == self.max_retries:
                    break
                current_messages = messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": raw}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "The previous response failed strict validation: "
                                    f"{error}. Regenerate the complete JSON object only."
                                ),
                            }
                        ],
                    }
                ]
        raise TeacherGenerationError(
            f"Teacher failed after {self.max_retries + 1} attempts: "
            + " | ".join(item["error"] for item in attempts),
            attempts,
        )


class TeacherGenerationError(ValueError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def cache_identity(sample: Sample, negative_label: str) -> dict[str, str]:
    return {
        "record_uid": sample.record_uid,
        "source_class": sample.label,
        "negative_label": negative_label,
        "context_path": str(sample.context_path),
        "evidence_path": str(sample.evidence_path),
    }


def generate_record(
    teacher: LocalTeacher,
    *,
    args: argparse.Namespace,
    sample: Sample,
    ontology,
    verified_factors: dict[str, Any],
) -> dict[str, Any]:
    negative_label = choose_neighbor(sample.record_uid, sample.label, ontology)
    cache_path = args.output_dir / "cache" / f"{sample.record_uid}.json"
    identity = cache_identity(sample, negative_label)
    if cache_path.is_file():
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError(f"Cached identity mismatch: {cache_path}")
    else:
        record = {
            "prompt_version": PROMPT_VERSION,
            "identity": identity,
            "verified_factors": verified_factors,
        }

    if "perception" not in record:
        try:
            perception, raw, retries = teacher.generate_json(
                perception_messages(sample),
                max_new_tokens=args.max_new_tokens_perception,
                validator=validate_perception,
            )
        except TeacherGenerationError as error:
            record["perception_failed_attempts"] = error.attempts
            atomic_write_json(cache_path, record)
            raise
        record["perception"] = perception
        record["raw_perception"] = raw
        record["perception_retries"] = retries
        atomic_write_json(cache_path, record)

    if "rewrite" not in record:
        try:
            rewrite, raw, retries = teacher.generate_json(
                rewrite_messages(
                    sample,
                    perception=record["perception"],
                    ontology=ontology,
                    negative_label=negative_label,
                    verified_factors=verified_factors,
                ),
                max_new_tokens=args.max_new_tokens_rewrite,
                validator=validate_rewrite,
                validator_kwargs={
                    "label": sample.label,
                    "negative_label": negative_label,
                    "verified_factors": verified_factors,
                    "expected_counterfactual_statement": counterfactual_statement(
                        negative_label, ontology.definitions[negative_label]
                    ),
                },
            )
        except TeacherGenerationError as error:
            record["rewrite_failed_attempts"] = error.attempts
            atomic_write_json(cache_path, record)
            raise
        record["rewrite"] = rewrite
        record["raw_rewrite"] = raw
        record["rewrite_retries"] = retries
        atomic_write_json(cache_path, record)

    if "verification" not in record:
        try:
            verification, raw, retries = teacher.generate_json(
                verification_messages(
                    sample,
                    positive_statement=record["rewrite"]["target"]["evidence"],
                    counterfactual_statement=record["rewrite"]["counterfactual_target"][
                        "evidence"
                    ],
                ),
                max_new_tokens=args.max_new_tokens_verification,
                validator=validate_verification,
            )
        except TeacherGenerationError as error:
            record["verification_failed_attempts"] = error.attempts
            atomic_write_json(cache_path, record)
            raise
        record["verification"] = verification
        record["raw_verification"] = raw
        record["verification_retries"] = retries
        atomic_write_json(cache_path, record)

    record["automatic_audit"] = automatic_audit(
        perception=record["perception"],
        rewrite=record["rewrite"],
        verification=record["verification"],
        label=sample.label,
        negative_label=negative_label,
        positive_threshold=args.positive_threshold,
        counterfactual_threshold=args.counterfactual_threshold,
    )
    atomic_write_json(cache_path, record)
    return record


def target_rows(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = record["identity"]
    rewrite = record["rewrite"]
    audit_passed = record["automatic_audit"]["passed"]
    common = {
        "record_uid": identity["record_uid"],
        "source_class": identity["source_class"],
        "context_path": identity["context_path"],
        "evidence_path": identity["evidence_path"],
        "counterfactual_target": rewrite["counterfactual_target"],
        "automatic_audit_passed": audit_passed,
        "prompt_version": record["prompt_version"],
    }
    generic = {
        **common,
        "target": {
            "events": [identity["source_class"]],
            "factors": record["verified_factors"],
            "evidence": record["perception"]["description"],
            "uncertain": rewrite["target"]["uncertain"],
        },
        "supervision_tier": "teacher_generic_pending_human_review",
    }
    grounded = {
        **common,
        "target": rewrite["target"],
        "supervision_tier": (
            "teacher_grounded_auto_pass_pending_human_review"
            if audit_passed
            else "teacher_grounded_auto_gate_failed_pending_human_review"
        ),
    }
    return generic, grounded


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def finalize(
    args: argparse.Namespace,
    samples: list[Sample],
    records: dict[str, dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> None:
    ordered = [records[sample.record_uid] for sample in samples]
    generic_rows = []
    grounded_rows = []
    audit_rows = []
    raw_rows = []
    for record in ordered:
        generic, grounded = target_rows(record)
        generic_rows.append(generic)
        grounded_rows.append(grounded)
        audit_rows.append(
            {
                **record["identity"],
                **record["automatic_audit"],
                "distinction": record["rewrite"]["distinction"],
                "missing_required_factors": record["rewrite"][
                    "missing_required_factors"
                ],
                "positive_verdict": record["verification"]["positive"],
                "counterfactual_verdict": record["verification"]["counterfactual"],
            }
        )
        raw_rows.append(record)

    write_jsonl(args.output_dir / "generic_targets.pending_review.jsonl", generic_rows)
    write_jsonl(args.output_dir / "grounded_targets.pending_review.jsonl", grounded_rows)
    write_jsonl(args.output_dir / "automatic_audit.jsonl", audit_rows)
    write_jsonl(args.output_dir / "teacher_records.jsonl", raw_rows)
    passed = sum(row["automatic_audit_passed"] for row in grounded_rows)
    summary = {
        "status": "complete_pending_human_review",
        "num_records": len(ordered),
        "automatic_audit_passed": passed,
        "automatic_audit_failed": len(ordered) - passed,
        "automatic_acceptance_rate": passed / len(ordered) if ordered else None,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "supervision_warning": (
            "These files are synthetic teacher outputs, not human-audited ground truth. "
            "Do not rename their supervision tier or pass them to --require-audited-targets."
        ),
        "outputs": {
            "generic": str(args.output_dir / "generic_targets.pending_review.jsonl"),
            "grounded": str(args.output_dir / "grounded_targets.pending_review.jsonl"),
            "audit": str(args.output_dir / "automatic_audit.jsonl"),
            "raw": str(args.output_dir / "teacher_records.jsonl"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    atomic_write_json(args.output_dir / "generation_summary.json", summary)


def main() -> None:
    args = parse_args()
    for name in ("positive_threshold", "counterfactual_threshold"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Teacher model is incomplete or missing: {args.model_path}")
    kernel_metadata = args.fp8_kernel_path / "build/torch-cuda/metadata.json"
    if not kernel_metadata.is_file():
        raise FileNotFoundError(
            "The pinned local finegrained-fp8 kernel is missing: "
            f"{args.fp8_kernel_path}"
        )
    kernel_description = json.loads(kernel_metadata.read_text(encoding="utf-8"))
    if kernel_description.get("name") != "finegrained-fp8":
        raise ValueError(f"Unexpected FP8 kernel metadata: {kernel_metadata}")

    ontology = load_ontology(args.ontology)
    labels = set(load_label_subset(args.labels_file, ontology))
    all_samples = read_samples(args.train_csv, args.data_root, include_labels=labels)
    selected_by_uid = {}
    for seed in args.seeds:
        for sample in cap_per_class(all_samples, args.max_per_class, seed):
            selected_by_uid[sample.record_uid] = sample
    samples = sorted(
        selected_by_uid.values(),
        key=lambda sample: (sample.label, sample.record_uid),
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    evidence_cards = load_evidence_cards(args.evidence_cards_jsonl)
    missing_cards = (
        [sample.record_uid for sample in samples if sample.record_uid not in evidence_cards]
        if args.evidence_cards_jsonl is not None
        else []
    )
    if missing_cards:
        raise ValueError(
            f"Evidence cards are missing {len(missing_cards)} selected records; "
            f"first={missing_cards[0]}"
        )

    config = configuration(args, len(samples))
    print(
        f"selected {len(samples)} training pairs from {len(labels)} labels; "
        f"teacher={args.model_path}"
    )
    print(f"configuration_fingerprint={fingerprint(config)}")
    if args.dry_run:
        if samples:
            negative = choose_neighbor(samples[0].record_uid, samples[0].label, ontology)
            print(
                canonical_json(
                    {
                        "first_record_uid": samples[0].record_uid,
                        "source_class": samples[0].label,
                        "counterfactual_neighbor": negative,
                        "context_path": str(samples[0].context_path),
                        "evidence_path": str(samples[0].evidence_path),
                        "verified_factors": evidence_cards.get(
                            samples[0].record_uid, {}
                        ),
                    }
                )
            )
        print("dry-run complete; model was not loaded and no outputs were written")
        return

    check_or_write_manifest(args, config)
    teacher = LocalTeacher(args)
    started = time.monotonic()
    records: dict[str, dict[str, Any]] = {}
    failures = []
    for index, sample in enumerate(samples, 1):
        try:
            record = generate_record(
                teacher,
                args=args,
                sample=sample,
                ontology=ontology,
                verified_factors=evidence_cards.get(sample.record_uid, {}),
            )
            records[sample.record_uid] = record
            gate = "PASS" if record["automatic_audit"]["passed"] else "REVIEW"
            print(
                f"[{index}/{len(samples)}] {sample.record_uid} "
                f"{sample.label} auto_gate={gate}",
                flush=True,
            )
        except Exception as error:
            failure = {
                "record_uid": sample.record_uid,
                "source_class": sample.label,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            atomic_write_json(
                args.output_dir / "failures" / f"{sample.record_uid}.json",
                failure,
            )
            print(
                f"[{index}/{len(samples)}] FAILED {sample.record_uid}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise
    elapsed = time.monotonic() - started
    if failures:
        write_jsonl(args.output_dir / "failures.jsonl", failures)
        partial_records = [
            records[sample.record_uid]
            for sample in samples
            if sample.record_uid in records
        ]
        write_jsonl(args.output_dir / "teacher_records.partial.jsonl", partial_records)
        raise RuntimeError(
            f"{len(failures)} teacher records failed; rerun the identical command to resume. "
            "Final target files were not written."
        )
    finalize(args, samples, records, elapsed_seconds=elapsed)
    print(f"complete: {args.output_dir / 'generation_summary.json'}")


if __name__ == "__main__":
    main()
