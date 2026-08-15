#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("forward_temporal", "session_disjoint", "unseen_site")
GROUP_FIELDS = {
    "forward_temporal": "session_id",
    "session_disjoint": "session_id",
    "unseen_site": "site_id",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    view: str
    targets: str | None = None
    require_audited: bool = False
    extra_train_args: tuple[str, ...] = ()


@dataclass
class Step:
    step_id: str
    kind: str
    command: list[str]
    outputs: list[str]
    metadata: dict[str, str | int | float | bool | None]
    status: str = "pending"
    log: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and run the CLEAR-UAV paper experiment matrix with resumable outputs"
        )
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "development", "official"],
        default="development",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("um7"),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("hf_cache"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_suite"))
    parser.add_argument("--results-root", type=Path, default=Path("results/paper_suite"))
    parser.add_argument("--labels-file", type=Path, default=Path("configs/core18_complete.txt"))
    parser.add_argument("--protocols", choices=PROTOCOLS, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional subset of experiment names; omit to run the complete profile",
    )
    parser.add_argument(
        "--development-targets",
        type=Path,
        help=(
            "Optional non-audited structured targets covering every selected run. "
            "If omitted, per-protocol/per-seed definition proxies are generated."
        ),
    )
    parser.add_argument("--audited-targets", type=Path)
    parser.add_argument("--generic-targets", type=Path)
    parser.add_argument(
        "--cropped-captions-root",
        type=Path,
        default=Path("description"),
        help="Per-image teacher JSON root, relative to --data-root unless absolute",
    )
    parser.add_argument(
        "--provenance-ready-file",
        type=Path,
        help="Official mode requires a JSON file with a top-level READY gate status",
    )
    parser.add_argument(
        "--acknowledge-test",
        action="store_true",
        help="Required in official mode before private test labels are opened",
    )
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument(
        "--min-free-gpu-mib",
        type=int,
        default=40_000,
        help="Fail before the suite starts if logical CUDA device 0 has less free memory",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--openclip-batch-size", type=int, default=32)
    parser.add_argument(
        "--openclip-finetuning",
        action="store_true",
        help="Also schedule OpenCLIP linear-probe and full visual fine-tuning runs",
    )
    parser.add_argument("--openclip-linear-epochs", type=int, default=20)
    parser.add_argument("--openclip-linear-batch-size", type=int, default=32)
    parser.add_argument("--openclip-linear-learning-rate", type=float, default=1e-3)
    parser.add_argument("--openclip-full-epochs", type=int, default=3)
    parser.add_argument("--openclip-full-batch-size", type=int, default=4)
    parser.add_argument("--openclip-full-gradient-accumulation", type=int, default=4)
    parser.add_argument("--openclip-full-learning-rate", type=float, default=5e-4)
    parser.add_argument("--openclip-backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--openclip-num-workers", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--multi-loss-batch-size", type=int, default=1)
    parser.add_argument("--multi-loss-gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run split validation and target generation without loading any model",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print every planned command (the JSON plan always contains them)",
    )
    parser.add_argument(
        "--skip-zero-shot",
        action="store_true",
        help="Skip deterministic OpenCLIP and base-Qwen baselines",
    )
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="Run validation, selected zero-shot baselines, analysis, and summaries only",
    )
    parser.add_argument(
        "--zero-shot-models",
        choices=["qwen", "openclip"],
        nargs="+",
        default=["qwen", "openclip"],
        help="Zero-shot model families to schedule",
    )
    parser.add_argument(
        "--skip-free-generation",
        action="store_true",
        help="Run only the normalized-likelihood scorer",
    )
    parser.add_argument(
        "--paper-tables-dir",
        type=Path,
        help=(
            "Export the five registered paper-table schemas after suite aggregation; "
            "defaults to <results-root>/paper_tables"
        ),
    )
    parser.add_argument(
        "--caption-quality-ledger",
        type=Path,
        help="Optional three-rater blinded ledger consumed by the paper-table exporter",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def labels_from_file(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def target_inventory(path: Path) -> tuple[set[str], set[str]]:
    record_uids = set()
    non_audited = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        uid = row["record_uid"]
        if uid in record_uids:
            raise ValueError(f"Duplicate record_uid in {path}:{line_number}: {uid}")
        record_uids.add(uid)
        if row.get("supervision_tier") != "human_audited":
            non_audited.add(uid)
    return record_uids, non_audited


def selected_train_uids(
    data_root: Path,
    protocol: str,
    labels: set[str],
    maximum: int,
    seed: int,
) -> set[str]:
    path = data_root / protocol / "train.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in csv.DictReader(handle):
            if row["source_class"] in labels:
                grouped[row["source_class"]].append(row["record_uid"])
    selected = set()
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest(),
        )
        selected.update(ordered[:maximum])
    return selected


def ready_status(path: Path) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("status", "scientific_gate", "gate_status", "release_gate"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.upper()
        if isinstance(value, dict) and isinstance(value.get("status"), str):
            return value["status"].upper()
    return None


def experiments_for(args: argparse.Namespace) -> list[Experiment]:
    if args.zero_shot_only:
        return []
    if args.profile == "smoke":
        return [Experiment("label_pair", "pair")]

    experiments = [
        Experiment(
            "label_pair_llm_lora",
            "pair",
            extra_train_args=("--lora-scope", "llm"),
        ),
        Experiment(
            "label_pair_unweighted",
            "pair",
            extra_train_args=("--label-weight", "1.0"),
        ),
        Experiment("label_context", "context"),
        Experiment("label_evidence", "evidence"),
        Experiment("label_pair", "pair"),
    ]
    if args.profile == "development":
        experiments.extend(
            [
                Experiment(
                    "grounded_caption",
                    "context",
                    targets="__cropped_captions__",
                ),
                Experiment(
                    "random_negative",
                    "pair",
                    targets="__cropped_captions__",
                    extra_train_args=("--lambda-neighbor", "0.1", "--random-negative"),
                ),
                Experiment(
                    "graph_neighbor",
                    "pair",
                    targets="__cropped_captions__",
                    extra_train_args=("--lambda-neighbor", "0.1"),
                ),
                Experiment(
                    "proxy_clear_no_dropout",
                    "pair",
                    targets="__cropped_plus_proxy_cf__",
                    extra_train_args=(
                        "--lambda-neighbor",
                        "0.1",
                        "--lambda-cf",
                        "0.1",
                    ),
                ),
                Experiment(
                    "proxy_clear_full",
                    "pair",
                    targets="__cropped_plus_proxy_cf__",
                    extra_train_args=(
                        "--lambda-neighbor",
                        "0.1",
                        "--lambda-cf",
                        "0.1",
                        "--context-dropout",
                        "0.1",
                        "--evidence-dropout",
                        "0.1",
                    ),
                ),
            ]
        )
        return experiments

    if args.audited_targets is None or args.generic_targets is None:
        raise ValueError("Official mode requires --audited-targets and --generic-targets")
    audited = str(resolve(args.audited_targets))
    generic = str(resolve(args.generic_targets))
    experiments.extend(
        [
            Experiment("generic_caption", "context", targets=generic),
            Experiment(
                "grounded_caption",
                "context",
                targets=audited,
                require_audited=True,
            ),
            Experiment(
                "random_negative",
                "pair",
                targets=audited,
                require_audited=True,
                extra_train_args=("--lambda-neighbor", "0.1", "--random-negative"),
            ),
            Experiment(
                "graph_neighbor",
                "pair",
                targets=audited,
                require_audited=True,
                extra_train_args=("--lambda-neighbor", "0.1"),
            ),
            Experiment(
                "clear_no_dropout",
                "pair",
                targets=audited,
                require_audited=True,
                extra_train_args=(
                    "--lambda-neighbor",
                    "0.1",
                    "--lambda-cf",
                    "0.1",
                ),
            ),
            Experiment(
                "clear_full",
                "pair",
                targets=audited,
                require_audited=True,
                extra_train_args=(
                    "--lambda-neighbor",
                    "0.1",
                    "--lambda-cf",
                    "0.1",
                    "--context-dropout",
                    "0.1",
                    "--evidence-dropout",
                    "0.1",
                ),
            ),
        ]
    )
    return experiments


def preflight(
    args: argparse.Namespace,
    protocols: list[str],
    experiments: list[Experiment],
) -> list[str]:
    blockers = [
        "GeoChat is not automated: its upstream custom inference stack is absent.",
        "GeoChat-UAV/UAVIT-1M weights are present, but their upstream inference adapter "
        "is not integrated.",
        "Qwen linear-probe, projector-only, QLoRA, full-model fine-tuning, and "
        "parameter-matched attention implementations are absent.",
        "Human-audited caption ratings, factor/hierarchy annotations, and single-factor "
        "swap cases are absent.",
        "Proposal, jittered, irrelevant, and crop-budget inputs plus small-evidence, "
        "adverse-capture, and complex-background slice annotations are absent.",
        "Learning-curve group budgets, tail exemplars, six frozen prompt paraphrases, "
        "and shuffled/missing-crop evaluations are not yet wired into the suite.",
        "Per-domain macro-F1, evidence-assignment accuracy, ECE, paired permutation/Holm "
        "tests, and the final heat-map/error-gallery workflow are not yet implemented.",
    ]
    data_root = resolve(args.data_root)
    labels_file = resolve(args.labels_file)
    models_root = resolve(args.models_root)
    required = [data_root, labels_file]
    needs_qwen = bool(experiments) or (
        not args.skip_zero_shot and "qwen" in args.zero_shot_models
    )
    needs_openclip = args.openclip_finetuning or (
        not args.skip_zero_shot and "openclip" in args.zero_shot_models
    )
    if needs_qwen:
        required.append(models_root / "qwen3-vl" / "config.json")
    if needs_openclip:
        required.append(models_root / "openclip" / "config.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))

    labels = labels_from_file(labels_file)
    for protocol in protocols:
        protocol_dir = data_root / protocol
        filenames = ["train.csv", "val.csv"]
        if args.profile == "official":
            filenames.extend(["test_inputs.csv", "test_labels_private.csv"])
        for filename in filenames:
            path = protocol_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    target_paths = {
        experiment.targets
        for experiment in experiments
        if experiment.targets not in {
            None,
            "__development__",
            "__cropped_captions__",
            "__cropped_plus_proxy_cf__",
        }
    }
    if any(
        experiment.targets in {"__cropped_captions__", "__cropped_plus_proxy_cf__"}
        for experiment in experiments
    ):
        captions_root = (
            args.cropped_captions_root
            if args.cropped_captions_root.is_absolute()
            else data_root / args.cropped_captions_root
        )
        if not captions_root.is_dir():
            raise FileNotFoundError(captions_root)
    if args.development_targets is not None:
        target_paths.add(str(resolve(args.development_targets)))
    for target_text in sorted(target_paths):
        target_path = Path(target_text)
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        target_uids, non_audited = target_inventory(target_path)
        for protocol in protocols:
            for seed in set(args.seeds):
                selected = selected_train_uids(
                    data_root, protocol, labels, args.max_per_class, seed
                )
                missing_uids = selected - target_uids
                if missing_uids:
                    raise ValueError(
                        f"{target_path} misses {len(missing_uids)} selected {protocol} "
                        f"seed-{seed} training records; first={sorted(missing_uids)[0]}"
                    )
        if any(
            experiment.targets == target_text and experiment.require_audited
            for experiment in experiments
        ) and non_audited:
            raise ValueError(
                f"{target_path} contains {len(non_audited)} non-human-audited rows; "
                f"first={sorted(non_audited)[0]}"
            )

    if args.profile == "official":
        if len(set(args.seeds)) < 3:
            raise ValueError("Official mode requires at least three distinct seeds")
        if not args.acknowledge_test:
            raise ValueError("Official mode requires --acknowledge-test")
        if args.provenance_ready_file is None:
            raise ValueError("Official mode requires --provenance-ready-file")
        provenance_path = resolve(args.provenance_ready_file)
        if not provenance_path.is_file():
            raise FileNotFoundError(provenance_path)
        if ready_status(provenance_path) != "READY":
            raise ValueError(f"Provenance gate is not READY: {provenance_path}")
    return blockers


class Planner:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.ids: set[str] = set()

    def add(
        self,
        step_id: str,
        kind: str,
        command: list[str | Path | int | float],
        outputs: list[Path],
        **metadata,
    ) -> None:
        if step_id in self.ids:
            raise ValueError(f"Duplicate step id: {step_id}")
        self.ids.add(step_id)
        self.steps.append(
            Step(
                step_id=step_id,
                kind=kind,
                command=[str(item) for item in command],
                outputs=[str(path) for path in outputs],
                metadata=metadata,
            )
        )

    def keep_zero_shot_models(self, models: set[str]) -> None:
        """Drop zero-shot steps belonging to unselected model families."""
        kept: list[Step] = []
        for step in self.steps:
            experiment = str(step.metadata.get("experiment", ""))
            family = next(
                (
                    candidate
                    for candidate in ("qwen", "openclip")
                    if experiment.startswith(f"{candidate}_")
                ),
                None,
            )
            if family is None or family in models:
                kept.append(step)
        self.steps = kept
        self.ids = {step.step_id for step in kept}


def python_command(script: str, *args: str | Path | int | float) -> list:
    return [sys.executable, ROOT / "scripts" / script, *args]


def add_analysis(
    planner: Planner,
    *,
    step_prefix: str,
    predictions: Path,
    output: Path,
    manifest: Path,
    labels_file: Path,
    group_field: str,
    metadata: dict,
) -> None:
    planner.add(
        f"{step_prefix}.analyze",
        "analysis",
        python_command(
            "analyze_predictions.py",
            "--predictions",
            predictions,
            "--manifest",
            manifest,
            "--labels-file",
            labels_file,
            "--group-field",
            group_field,
            "--output",
            output,
        ),
        [output],
        **metadata,
    )


def add_zero_shot(
    planner: Planner,
    *,
    args: argparse.Namespace,
    protocol: str,
    split: str,
    data_root: Path,
    models_root: Path,
    output_root: Path,
    results_root: Path,
    labels_file: Path,
    max_samples: int | None,
) -> None:
    protocol_dir = data_root / protocol
    csv_path = (
        protocol_dir / "val.csv"
        if split == "val"
        else protocol_dir / "test_inputs.csv"
    )
    private_args: list[str | Path] = []
    if split == "test":
        private_args = [
            "--private-labels",
            protocol_dir / "test_labels_private.csv",
        ]
    sample_args: list[str | int] = []
    if max_samples is not None:
        sample_args = ["--max-samples", max_samples]

    for prompt in ("direct", "definition"):
        qwen_output = (
            output_root / protocol / "zero_shot" / f"qwen_{prompt}_{split}.jsonl"
        )
        prefix = f"{protocol}.zero.qwen_{prompt}.{split}"
        planner.add(
            prefix,
            "zero_shot",
            python_command(
                "evaluate_qwen.py",
                "--model-path",
                models_root / "qwen3-vl",
                "--data-root",
                data_root,
                "--csv",
                csv_path,
                *private_args,
                "--labels-file",
                labels_file,
                "--view",
                "context",
                "--prompt",
                prompt,
                "--max-new-tokens",
                args.max_new_tokens,
                "--max-pixels",
                args.max_pixels,
                "--output",
                qwen_output,
                *sample_args,
            ),
            [qwen_output, qwen_output.with_suffix(".metrics.json")],
            protocol=protocol,
            experiment=f"qwen_{prompt}",
            seed=0,
            split=split,
            decoder="free",
        )
        add_analysis(
            planner,
            step_prefix=prefix,
            predictions=qwen_output,
            output=(
                results_root
                / "metrics"
                / protocol
                / f"qwen_{prompt}_{split}_analysis.json"
            ),
            manifest=csv_path,
            labels_file=labels_file,
            group_field=GROUP_FIELDS[protocol],
            metadata={
                "protocol": protocol,
                "experiment": f"qwen_{prompt}",
                "seed": 0,
                "split": split,
                "decoder": "free",
            },
        )

        if prompt == "definition":
            qwen_closed_output = (
                output_root
                / protocol
                / "zero_shot"
                / f"qwen_definition_{split}_closed_set.json"
            )
            threshold_args: list[str | Path] = (
                ["--fit-thresholds"]
                if split == "val"
                else [
                    "--thresholds",
                    output_root
                    / protocol
                    / "zero_shot"
                    / "qwen_definition_val_closed_set.thresholds.json",
                ]
            )
            closed_prefix = f"{protocol}.zero.qwen_definition.closed.{split}"
            planner.add(
                closed_prefix,
                "evaluation",
                python_command(
                    "evaluate_closed_set.py",
                    "--model-path",
                    models_root / "qwen3-vl",
                    "--data-root",
                    data_root,
                    "--csv",
                    csv_path,
                    *private_args,
                    "--labels-file",
                    labels_file,
                    "--view",
                    "context",
                    "--candidate-batch-size",
                    args.candidate_batch_size,
                    "--max-length",
                    args.max_length,
                    "--max-pixels",
                    args.max_pixels,
                    "--set-aggregator",
                    "logsumexp",
                    *threshold_args,
                    "--output",
                    qwen_closed_output,
                    *sample_args,
                ),
                [
                    qwen_closed_output,
                    *(
                        [qwen_closed_output.with_suffix(".thresholds.json")]
                        if split == "val"
                        else []
                    ),
                ],
                protocol=protocol,
                experiment="qwen_definition",
                seed=0,
                split=split,
                decoder="closed",
            )
            add_analysis(
                planner,
                step_prefix=closed_prefix,
                predictions=qwen_closed_output,
                output=(
                    results_root
                    / "metrics"
                    / protocol
                    / f"qwen_definition_{split}_closed_analysis.json"
                ),
                manifest=csv_path,
                labels_file=labels_file,
                group_field=GROUP_FIELDS[protocol],
                metadata={
                    "protocol": protocol,
                    "experiment": "qwen_definition",
                    "seed": 0,
                    "split": split,
                    "decoder": "closed",
                },
            )

        clip_output = (
            output_root / protocol / "zero_shot" / f"openclip_{prompt}_{split}.json"
        )
        prefix = f"{protocol}.zero.openclip_{prompt}.{split}"
        planner.add(
            prefix,
            "zero_shot",
            python_command(
                "evaluate_clip.py",
                "--model-path",
                models_root / "openclip",
                "--data-root",
                data_root,
                "--csv",
                csv_path,
                *private_args,
                "--labels-file",
                labels_file,
                "--view",
                "context",
                "--prompt",
                prompt,
                "--batch-size",
                args.openclip_batch_size,
                "--output",
                clip_output,
                *sample_args,
            ),
            [clip_output],
            protocol=protocol,
            experiment=f"openclip_{prompt}",
            seed=0,
            split=split,
            decoder="score",
        )
        add_analysis(
            planner,
            step_prefix=prefix,
            predictions=clip_output,
            output=(
                results_root
                / "metrics"
                / protocol
                / f"openclip_{prompt}_{split}_analysis.json"
            ),
            manifest=csv_path,
            labels_file=labels_file,
            group_field=GROUP_FIELDS[protocol],
            metadata={
                "protocol": protocol,
                "experiment": f"openclip_{prompt}",
                "seed": 0,
                "split": split,
                "decoder": "score",
            },
        )


def add_openclip_trained_run(
    planner: Planner,
    *,
    args: argparse.Namespace,
    mode: str,
    protocol: str,
    seed: int,
    data_root: Path,
    models_root: Path,
    output_root: Path,
    results_root: Path,
    labels_file: Path,
    max_samples: int | None,
    run_test: bool,
) -> None:
    if mode not in {"linear_probe", "full_finetune"}:
        raise ValueError(f"Unknown OpenCLIP mode: {mode}")
    protocol_dir = data_root / protocol
    experiment = f"openclip_{mode}"
    run_dir = output_root / protocol / experiment / f"seed{seed}"
    checkpoint = run_dir / "final" / "openclip_classifier.pt"
    linear = mode == "linear_probe"
    epochs = args.openclip_linear_epochs if linear else args.openclip_full_epochs
    batch_size = (
        args.openclip_linear_batch_size if linear else args.openclip_full_batch_size
    )
    learning_rate = (
        args.openclip_linear_learning_rate
        if linear
        else args.openclip_full_learning_rate
    )
    gradient_accumulation = (
        1 if linear else args.openclip_full_gradient_accumulation
    )
    sample_args: list[str | int] = []
    if max_samples is not None:
        sample_args = ["--max-samples", max_samples]

    prefix = f"{protocol}.{experiment}.seed{seed}"
    train_args: list[str | Path | int | float] = [
        "--model-path",
        models_root / "openclip",
        "--data-root",
        data_root,
        "--train-csv",
        protocol_dir / "train.csv",
        "--val-csv",
        protocol_dir / "val.csv",
        "--output-dir",
        run_dir,
        "--labels-file",
        labels_file,
        "--mode",
        mode,
        "--view",
        "context",
        "--prompt",
        "definition",
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--gradient-accumulation",
        gradient_accumulation,
        "--learning-rate",
        learning_rate,
        "--backbone-learning-rate",
        args.openclip_backbone_learning_rate,
        "--num-workers",
        args.openclip_num_workers,
        "--max-per-class",
        args.max_per_class,
        "--seed",
        seed,
        *sample_args,
    ]
    if not linear:
        train_args.append("--gradient-checkpointing")
    planner.add(
        f"{prefix}.train",
        "train",
        python_command("train_openclip.py", *train_args),
        [checkpoint, run_dir / "run_metadata.json"],
        protocol=protocol,
        experiment=experiment,
        seed=seed,
        split="train",
        decoder="train",
    )

    for split in (["val", "test"] if run_test else ["val"]):
        csv_path = (
            protocol_dir / "val.csv"
            if split == "val"
            else protocol_dir / "test_inputs.csv"
        )
        private_args: list[str | Path] = []
        if split == "test":
            private_args = [
                "--private-labels",
                protocol_dir / "test_labels_private.csv",
            ]
        output = run_dir / f"{split}_predictions.json"
        eval_prefix = f"{prefix}.{split}"
        planner.add(
            eval_prefix,
            "evaluation",
            python_command(
                "evaluate_openclip_finetuned.py",
                "--model-path",
                models_root / "openclip",
                "--checkpoint",
                checkpoint,
                "--data-root",
                data_root,
                "--csv",
                csv_path,
                *private_args,
                "--labels-file",
                labels_file,
                "--view",
                "context",
                "--batch-size",
                args.openclip_batch_size,
                "--output",
                output,
                *sample_args,
            ),
            [output],
            protocol=protocol,
            experiment=experiment,
            seed=seed,
            split=split,
            decoder="score",
        )
        add_analysis(
            planner,
            step_prefix=eval_prefix,
            predictions=output,
            output=(
                results_root
                / "metrics"
                / protocol
                / experiment
                / f"seed{seed}_{split}.json"
            ),
            manifest=csv_path,
            labels_file=labels_file,
            group_field=GROUP_FIELDS[protocol],
            metadata={
                "protocol": protocol,
                "experiment": experiment,
                "seed": seed,
                "split": split,
                "decoder": "score",
            },
        )


def add_trained_run(
    planner: Planner,
    *,
    args: argparse.Namespace,
    experiment: Experiment,
    protocol: str,
    seed: int,
    data_root: Path,
    models_root: Path,
    output_root: Path,
    results_root: Path,
    labels_file: Path,
    max_samples: int | None,
    run_test: bool,
    targets_override: Path | None = None,
) -> None:
    run_dir = output_root / protocol / experiment.name / f"seed{seed}"
    protocol_dir = data_root / protocol
    uses_multiple_losses = any(
        flag in experiment.extra_train_args for flag in ("--lambda-neighbor", "--lambda-cf")
    )
    train_batch_size = (
        args.multi_loss_batch_size if uses_multiple_losses else args.batch_size
    )
    gradient_accumulation = (
        args.multi_loss_gradient_accumulation
        if uses_multiple_losses
        else args.gradient_accumulation
    )
    train_args: list[str | Path | int | float] = [
        "--model-path",
        models_root / "qwen3-vl",
        "--data-root",
        data_root,
        "--train-csv",
        protocol_dir / "train.csv",
        "--output-dir",
        run_dir,
        "--labels-file",
        labels_file,
        "--max-per-class",
        args.max_per_class,
        "--view",
        experiment.view,
        "--epochs",
        1 if args.profile == "smoke" else args.epochs,
        "--batch-size",
        train_batch_size,
        "--gradient-accumulation",
        1 if args.profile == "smoke" else gradient_accumulation,
        "--max-length",
        args.max_length,
        "--max-pixels",
        args.max_pixels,
        "--seed",
        seed,
    ]
    if max_samples is not None:
        train_args.extend(["--max-samples", max_samples])
    target_path = (
        targets_override
        if targets_override is not None
        else Path(experiment.targets)
        if experiment.targets
        else None
    )
    if target_path is not None:
        train_args.extend(["--targets-jsonl", target_path])
    if experiment.require_audited:
        train_args.append("--require-audited-targets")
    if args.resume and run_dir.is_dir():
        checkpoints = sorted(
            run_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
        if checkpoints:
            train_args.extend(["--resume-from-checkpoint", checkpoints[-1]])
    train_args.extend(experiment.extra_train_args)
    prefix = f"{protocol}.{experiment.name}.seed{seed}"
    planner.add(
        f"{prefix}.train",
        "train",
        python_command("train_qwen.py", *train_args),
        [run_dir / "final" / "adapter_config.json", run_dir / "run_metadata.json"],
        protocol=protocol,
        experiment=experiment.name,
        seed=seed,
        split="train",
        decoder="train",
    )

    splits = ["val", "test"] if run_test else ["val"]
    run_view_diagnostics = experiment.name in {
        "label_pair",
        "proxy_clear_full",
        "clear_full",
    }
    for split in splits:
        csv_path = (
            protocol_dir / "val.csv"
            if split == "val"
            else protocol_dir / "test_inputs.csv"
        )
        private_args: list[str | Path] = []
        if split == "test":
            private_args = [
                "--private-labels",
                protocol_dir / "test_labels_private.csv",
            ]
        sample_args: list[str | int] = []
        if max_samples is not None:
            sample_args = ["--max-samples", max_samples]

        if not args.skip_free_generation:
            predictions = run_dir / f"{split}_predictions.jsonl"
            free_prefix = f"{prefix}.free.{split}"
            planner.add(
                free_prefix,
                "evaluation",
                python_command(
                    "evaluate_qwen.py",
                    "--model-path",
                    models_root / "qwen3-vl",
                    "--adapter-path",
                    run_dir / "final",
                    "--data-root",
                    data_root,
                    "--csv",
                    csv_path,
                    *private_args,
                    "--labels-file",
                    labels_file,
                    "--view",
                    experiment.view,
                    "--max-new-tokens",
                    args.max_new_tokens,
                    "--max-pixels",
                    args.max_pixels,
                    "--output",
                    predictions,
                    *sample_args,
                ),
                [predictions, predictions.with_suffix(".metrics.json")],
                protocol=protocol,
                experiment=experiment.name,
                seed=seed,
                split=split,
                decoder="free",
            )
            add_analysis(
                planner,
                step_prefix=free_prefix,
                predictions=predictions,
                output=(
                    results_root
                    / "metrics"
                    / protocol
                    / experiment.name
                    / f"seed{seed}_{split}_free.json"
                ),
                manifest=csv_path,
                labels_file=labels_file,
                group_field=GROUP_FIELDS[protocol],
                metadata={
                    "protocol": protocol,
                    "experiment": experiment.name,
                    "seed": seed,
                    "split": split,
                    "decoder": "free",
                },
            )

        closed_output = run_dir / f"{split}_closed_set.json"
        threshold_args: list[str | Path] = (
            ["--fit-thresholds"]
            if split == "val"
            else ["--thresholds", run_dir / "val_closed_set.thresholds.json"]
        )
        closed_prefix = f"{prefix}.closed.{split}"
        planner.add(
            closed_prefix,
            "evaluation",
            python_command(
                "evaluate_closed_set.py",
                "--model-path",
                models_root / "qwen3-vl",
                "--adapter-path",
                run_dir / "final",
                "--data-root",
                data_root,
                "--csv",
                csv_path,
                *private_args,
                "--labels-file",
                labels_file,
                "--view",
                experiment.view,
                "--candidate-batch-size",
                args.candidate_batch_size,
                "--max-length",
                args.max_length,
                "--max-pixels",
                args.max_pixels,
                "--set-aggregator",
                "logsumexp",
                *threshold_args,
                "--output",
                closed_output,
                *sample_args,
            ),
            [
                closed_output,
                *(
                    [run_dir / "val_closed_set.thresholds.json"]
                    if split == "val"
                    else []
                ),
            ],
            protocol=protocol,
            experiment=experiment.name,
            seed=seed,
            split=split,
            decoder="closed",
        )
        add_analysis(
            planner,
            step_prefix=closed_prefix,
            predictions=closed_output,
            output=(
                results_root
                / "metrics"
                / protocol
                / experiment.name
                / f"seed{seed}_{split}_closed.json"
            ),
            manifest=csv_path,
            labels_file=labels_file,
            group_field=GROUP_FIELDS[protocol],
            metadata={
                "protocol": protocol,
                "experiment": experiment.name,
                "seed": seed,
                "split": split,
                "decoder": "closed",
            },
        )

        max_output = run_dir / f"{split}_set_max.json"
        max_threshold_args: list[str | Path] = (
            ["--fit-thresholds"]
            if split == "val"
            else ["--thresholds", run_dir / "val_set_max.thresholds.json"]
        )
        planner.add(
            f"{prefix}.set_max.{split}",
            "set_rescore",
            python_command(
                "rescore_sets.py",
                "--scores",
                closed_output,
                "--aggregator",
                "max",
                *max_threshold_args,
                "--output",
                max_output,
            ),
            [
                max_output,
                *(
                    [run_dir / "val_set_max.thresholds.json"]
                    if split == "val"
                    else []
                ),
            ],
            protocol=protocol,
            experiment=experiment.name,
            seed=seed,
            split=split,
            decoder="set_max",
        )
        if run_view_diagnostics:
            for aggregator in ("logsumexp", "max"):
                deletion_output = (
                    results_root
                    / "diagnostics"
                    / protocol
                    / experiment.name
                    / f"seed{seed}_{split}_deletion_{aggregator}.json"
                )
                planner.add(
                    f"{prefix}.deletion_{aggregator}.{split}",
                    "evidence_deletion",
                    python_command(
                        "evaluate_evidence_deletion.py",
                        "--scores",
                        closed_output,
                        "--aggregator",
                        aggregator,
                        "--seed",
                        seed,
                        "--output",
                        deletion_output,
                    ),
                    [deletion_output],
                    protocol=protocol,
                    experiment=experiment.name,
                    seed=seed,
                    split=split,
                    decoder=f"deletion_{aggregator}",
                )

    if not run_view_diagnostics:
        return

    for intervention_view in ("context", "evidence"):
        intervention_name = f"{experiment.name}_as_{intervention_view}"
        for split in splits:
            csv_path = (
                protocol_dir / "val.csv"
                if split == "val"
                else protocol_dir / "test_inputs.csv"
            )
            private_args: list[str | Path] = []
            if split == "test":
                private_args = [
                    "--private-labels",
                    protocol_dir / "test_labels_private.csv",
                ]
            sample_args: list[str | int] = []
            if max_samples is not None:
                sample_args = ["--max-samples", max_samples]

            if not args.skip_free_generation:
                predictions = (
                    run_dir / f"{split}_as_{intervention_view}_predictions.jsonl"
                )
                free_prefix = (
                    f"{prefix}.as_{intervention_view}.free.{split}"
                )
                planner.add(
                    free_prefix,
                    "evaluation",
                    python_command(
                        "evaluate_qwen.py",
                        "--model-path",
                        models_root / "qwen3-vl",
                        "--adapter-path",
                        run_dir / "final",
                        "--data-root",
                        data_root,
                        "--csv",
                        csv_path,
                        *private_args,
                        "--labels-file",
                        labels_file,
                        "--view",
                        intervention_view,
                        "--max-new-tokens",
                        args.max_new_tokens,
                        "--max-pixels",
                        args.max_pixels,
                        "--output",
                        predictions,
                        *sample_args,
                    ),
                    [predictions, predictions.with_suffix(".metrics.json")],
                    protocol=protocol,
                    experiment=intervention_name,
                    seed=seed,
                    split=split,
                    decoder="free",
                )
                add_analysis(
                    planner,
                    step_prefix=free_prefix,
                    predictions=predictions,
                    output=(
                        results_root
                        / "metrics"
                        / protocol
                        / intervention_name
                        / f"seed{seed}_{split}_free.json"
                    ),
                    manifest=csv_path,
                    labels_file=labels_file,
                    group_field=GROUP_FIELDS[protocol],
                    metadata={
                        "protocol": protocol,
                        "experiment": intervention_name,
                        "seed": seed,
                        "split": split,
                        "decoder": "free",
                    },
                )

            closed_output = (
                run_dir / f"{split}_as_{intervention_view}_closed_set.json"
            )
            threshold_args: list[str | Path] = (
                ["--fit-thresholds"]
                if split == "val"
                else [
                    "--thresholds",
                    (
                        run_dir
                        / f"val_as_{intervention_view}_closed_set.thresholds.json"
                    ),
                ]
            )
            closed_prefix = (
                f"{prefix}.as_{intervention_view}.closed.{split}"
            )
            planner.add(
                closed_prefix,
                "evaluation",
                python_command(
                    "evaluate_closed_set.py",
                    "--model-path",
                    models_root / "qwen3-vl",
                    "--adapter-path",
                    run_dir / "final",
                    "--data-root",
                    data_root,
                    "--csv",
                    csv_path,
                    *private_args,
                    "--labels-file",
                    labels_file,
                    "--view",
                    intervention_view,
                    "--candidate-batch-size",
                    args.candidate_batch_size,
                    "--max-length",
                    args.max_length,
                    "--max-pixels",
                    args.max_pixels,
                    "--set-aggregator",
                    "logsumexp",
                    *threshold_args,
                    "--output",
                    closed_output,
                    *sample_args,
                ),
                [
                    closed_output,
                    *(
                        [
                            run_dir
                            / (
                                f"val_as_{intervention_view}_closed_set."
                                "thresholds.json"
                            )
                        ]
                        if split == "val"
                        else []
                    ),
                ],
                protocol=protocol,
                experiment=intervention_name,
                seed=seed,
                split=split,
                decoder="closed",
            )
            add_analysis(
                planner,
                step_prefix=closed_prefix,
                predictions=closed_output,
                output=(
                    results_root
                    / "metrics"
                    / protocol
                    / intervention_name
                    / f"seed{seed}_{split}_closed.json"
                ),
                manifest=csv_path,
                labels_file=labels_file,
                group_field=GROUP_FIELDS[protocol],
                metadata={
                    "protocol": protocol,
                    "experiment": intervention_name,
                    "seed": seed,
                    "split": split,
                    "decoder": "closed",
                },
            )

    for split in splits:
        for decoder, joint_name, intervention_pattern in (
            (
                "free",
                f"{split}_predictions.jsonl",
                f"{split}_as_{{view}}_predictions.jsonl",
            ),
            (
                "closed",
                f"{split}_closed_set.json",
                f"{split}_as_{{view}}_closed_set.json",
            ),
        ):
            if decoder == "free" and args.skip_free_generation:
                continue
            output = (
                results_root
                / "diagnostics"
                / protocol
                / experiment.name
                / f"seed{seed}_{split}_view_reliance_{decoder}.json"
            )
            planner.add(
                f"{prefix}.view_reliance.{decoder}.{split}",
                "view_reliance",
                python_command(
                    "evaluate_view_reliance.py",
                    "--joint",
                    run_dir / joint_name,
                    "--context",
                    run_dir / intervention_pattern.format(view="context"),
                    "--evidence",
                    run_dir / intervention_pattern.format(view="evidence"),
                    "--output",
                    output,
                ),
                [output],
                protocol=protocol,
                experiment=experiment.name,
                seed=seed,
                split=split,
                decoder=f"view_reliance_{decoder}",
            )


def add_comparisons(
    planner: Planner,
    *,
    args: argparse.Namespace,
    experiments: list[Experiment],
    protocols: list[str],
    seeds: list[int],
    data_root: Path,
    output_root: Path,
    results_root: Path,
    labels_file: Path,
    split: str,
) -> None:
    names = {experiment.name for experiment in experiments}
    pairs = [
        ("label_context", "label_pair", "pair_vs_context"),
        ("label_evidence", "label_pair", "pair_vs_evidence"),
        ("label_pair_llm_lora", "label_pair", "projector_lora_effect"),
        ("label_pair_unweighted", "label_pair", "label_weight_effect"),
        ("label_context", "grounded_caption", "caption_effect"),
        ("grounded_caption", "clear_full", "clear_vs_grounded"),
        (
            "grounded_caption",
            "proxy_clear_full",
            "proxy_clear_vs_grounded",
        ),
        ("random_negative", "graph_neighbor", "graph_vs_random"),
        ("graph_neighbor", "clear_no_dropout", "counterfactual_effect"),
        (
            "graph_neighbor",
            "proxy_clear_no_dropout",
            "proxy_counterfactual_effect",
        ),
        ("clear_no_dropout", "clear_full", "dropout_effect"),
        ("proxy_clear_no_dropout", "proxy_clear_full", "proxy_dropout_effect"),
    ]
    for left, right, comparison in pairs:
        if left not in names or right not in names:
            continue
        for protocol in protocols:
            manifest = (
                data_root / protocol / "val.csv"
                if split == "val"
                else data_root / protocol / "test_inputs.csv"
            )
            for seed in seeds:
                for decoder, filename in (
                    ("free", f"{split}_predictions.jsonl"),
                    ("closed", f"{split}_closed_set.json"),
                ):
                    if decoder == "free" and args.skip_free_generation:
                        continue
                    left_path = output_root / protocol / left / f"seed{seed}" / filename
                    right_path = output_root / protocol / right / f"seed{seed}" / filename
                    result = (
                        results_root
                        / "comparisons"
                        / protocol
                        / f"seed{seed}_{split}_{comparison}_{decoder}.json"
                    )
                    planner.add(
                        (
                            f"{protocol}.compare.{comparison}.seed{seed}."
                            f"{decoder}.{split}"
                        ),
                        "comparison",
                        python_command(
                            "compare_predictions.py",
                            "--left",
                            left_path,
                            "--right",
                            right_path,
                            "--manifest",
                            manifest,
                            "--labels-file",
                            labels_file,
                            "--group-field",
                            GROUP_FIELDS[protocol],
                            "--output",
                            result,
                        ),
                        [result],
                        protocol=protocol,
                        experiment=comparison,
                        seed=seed,
                        split=split,
                        decoder=decoder,
                    )


def make_plan(args: argparse.Namespace) -> tuple[list[Step], list[str], Path, Path]:
    if args.zero_shot_only and args.skip_zero_shot and not args.openclip_finetuning:
        raise ValueError("--zero-shot-only cannot be combined with --skip-zero-shot")
    data_root = resolve(args.data_root)
    models_root = resolve(args.models_root)
    output_root = resolve(args.output_root)
    results_root = resolve(args.results_root)
    labels_file = resolve(args.labels_file)
    protocols = args.protocols or (
        list(PROTOCOLS) if args.profile == "official" else ["session_disjoint"]
    )
    seeds = list(dict.fromkeys(args.seeds))
    if args.profile == "smoke":
        seeds = seeds[:1]
    experiments = experiments_for(args)
    if args.experiments:
        available = {experiment.name for experiment in experiments}
        unknown = set(args.experiments) - available
        if unknown:
            raise ValueError(
                f"Unknown experiments for {args.profile}: {sorted(unknown)}; "
                f"available={sorted(available)}"
            )
        requested = set(args.experiments)
        experiments = [
            experiment for experiment in experiments if experiment.name in requested
        ]
    blockers = preflight(args, protocols, experiments)
    planner = Planner()
    max_samples = 8 if args.profile == "smoke" else None

    for protocol in protocols:
        planner.add(
            f"{protocol}.validate",
            "validation",
            python_command(
                "validate_split.py",
                "--data-root",
                data_root,
                "--protocol",
                protocol,
                "--labels-file",
                labels_file,
                *(["--train-val-only"] if args.profile != "official" else []),
            ),
            [],
            protocol=protocol,
            experiment="data_validation",
            seed=0,
            split="all",
            decoder="none",
        )
        if not args.skip_zero_shot:
            add_zero_shot(
                planner,
                args=args,
                protocol=protocol,
                split="val",
                data_root=data_root,
                models_root=models_root,
                output_root=output_root,
                results_root=results_root,
                labels_file=labels_file,
                max_samples=max_samples,
            )
            if args.profile == "official":
                add_zero_shot(
                    planner,
                    args=args,
                    protocol=protocol,
                    split="test",
                    data_root=data_root,
                    models_root=models_root,
                    output_root=output_root,
                    results_root=results_root,
                    labels_file=labels_file,
                    max_samples=None,
                )
        if args.openclip_finetuning:
            for mode in ("linear_probe", "full_finetune"):
                for seed in seeds:
                    add_openclip_trained_run(
                        planner,
                        args=args,
                        mode=mode,
                        protocol=protocol,
                        seed=seed,
                        data_root=data_root,
                        models_root=models_root,
                        output_root=output_root,
                        results_root=results_root,
                        labels_file=labels_file,
                        max_samples=max_samples,
                        run_test=args.profile == "official",
                    )
        for experiment in experiments:
            for seed in seeds:
                targets_override = None
                if experiment.targets == "__development__":
                    if args.development_targets is not None:
                        targets_override = resolve(args.development_targets)
                    else:
                        targets_override = (
                            output_root
                            / protocol
                            / "proxy_targets"
                            / f"seed{seed}.jsonl"
                        )
                        target_prefix = f"{protocol}.proxy_targets.seed{seed}"
                        if target_prefix not in planner.ids:
                            planner.add(
                                target_prefix,
                                "target_generation",
                                python_command(
                                    "build_proxy_targets.py",
                                    "--data-root",
                                    data_root,
                                    "--train-csv",
                                    data_root / protocol / "train.csv",
                                    "--labels-file",
                                    labels_file,
                                    "--max-per-class",
                                    args.max_per_class,
                                    "--seed",
                                    seed,
                                    "--output",
                                    targets_override,
                                ),
                                [targets_override],
                                protocol=protocol,
                                experiment="proxy_targets",
                                seed=seed,
                                split="train",
                                decoder="none",
                            )
                elif experiment.targets == "__cropped_captions__":
                    targets_override = (
                        output_root
                        / protocol
                        / "cropped_caption_targets"
                        / f"seed{seed}.jsonl"
                    )
                    target_prefix = f"{protocol}.cropped_caption_targets.seed{seed}"
                    if target_prefix not in planner.ids:
                        captions_root = (
                            args.cropped_captions_root
                            if args.cropped_captions_root.is_absolute()
                            else data_root / args.cropped_captions_root
                        )
                        planner.add(
                            target_prefix,
                            "target_generation",
                            python_command(
                                "build_cropped_caption_targets.py",
                                "--data-root",
                                data_root,
                                "--train-csv",
                                data_root / protocol / "train.csv",
                                "--captions-root",
                                captions_root,
                                "--labels-file",
                                labels_file,
                                "--max-per-class",
                                args.max_per_class,
                                "--seed",
                                seed,
                                "--output",
                                targets_override,
                            ),
                            [targets_override],
                            protocol=protocol,
                            experiment="grounded_caption_targets",
                            seed=seed,
                            split="train",
                            decoder="none",
                        )
                elif experiment.targets == "__cropped_plus_proxy_cf__":
                    cropped_targets = (
                        output_root
                        / protocol
                        / "cropped_caption_targets"
                        / f"seed{seed}.jsonl"
                    )
                    cropped_prefix = f"{protocol}.cropped_caption_targets.seed{seed}"
                    if cropped_prefix not in planner.ids:
                        captions_root = (
                            args.cropped_captions_root
                            if args.cropped_captions_root.is_absolute()
                            else data_root / args.cropped_captions_root
                        )
                        planner.add(
                            cropped_prefix,
                            "target_generation",
                            python_command(
                                "build_cropped_caption_targets.py",
                                "--data-root",
                                data_root,
                                "--train-csv",
                                data_root / protocol / "train.csv",
                                "--captions-root",
                                captions_root,
                                "--labels-file",
                                labels_file,
                                "--max-per-class",
                                args.max_per_class,
                                "--seed",
                                seed,
                                "--output",
                                cropped_targets,
                            ),
                            [cropped_targets],
                            protocol=protocol,
                            experiment="grounded_caption_targets",
                            seed=seed,
                            split="train",
                            decoder="none",
                        )
                    proxy_targets = (
                        output_root / protocol / "proxy_targets" / f"seed{seed}.jsonl"
                    )
                    proxy_prefix = f"{protocol}.proxy_targets.seed{seed}"
                    if proxy_prefix not in planner.ids:
                        planner.add(
                            proxy_prefix,
                            "target_generation",
                            python_command(
                                "build_proxy_targets.py",
                                "--data-root",
                                data_root,
                                "--train-csv",
                                data_root / protocol / "train.csv",
                                "--labels-file",
                                labels_file,
                                "--max-per-class",
                                args.max_per_class,
                                "--seed",
                                seed,
                                "--output",
                                proxy_targets,
                            ),
                            [proxy_targets],
                            protocol=protocol,
                            experiment="proxy_targets",
                            seed=seed,
                            split="train",
                            decoder="none",
                        )
                    targets_override = (
                        output_root
                        / protocol
                        / "cropped_caption_proxy_cf_targets"
                        / f"seed{seed}.jsonl"
                    )
                    merged_prefix = (
                        f"{protocol}.cropped_caption_proxy_cf_targets.seed{seed}"
                    )
                    if merged_prefix not in planner.ids:
                        planner.add(
                            merged_prefix,
                            "target_generation",
                            python_command(
                                "merge_grounded_proxy_counterfactuals.py",
                                "--grounded-targets",
                                cropped_targets,
                                "--proxy-targets",
                                proxy_targets,
                                "--output",
                                targets_override,
                            ),
                            [targets_override],
                            protocol=protocol,
                            experiment="grounded_proxy_counterfactual_targets",
                            seed=seed,
                            split="train",
                            decoder="none",
                        )
                add_trained_run(
                    planner,
                    args=args,
                    experiment=experiment,
                    protocol=protocol,
                    seed=seed,
                    data_root=data_root,
                    models_root=models_root,
                    output_root=output_root,
                    results_root=results_root,
                    labels_file=labels_file,
                    max_samples=max_samples,
                    run_test=args.profile == "official",
                    targets_override=targets_override,
                )

    planner.keep_zero_shot_models(set(args.zero_shot_models))
    add_comparisons(
        planner,
        args=args,
        experiments=experiments,
        protocols=protocols,
        seeds=seeds,
        data_root=data_root,
        output_root=output_root,
        results_root=results_root,
        labels_file=labels_file,
        split="test" if args.profile == "official" else "val",
    )
    plan_path = results_root / "suite_plan.json"
    summary_path = results_root / "suite_summary.json"
    planner.add(
        "suite.summarize",
        "summary",
        python_command(
            "summarize_suite.py",
            "--plan",
            plan_path,
            "--output",
            summary_path,
            "--csv-output",
            results_root / "suite_summary.csv",
            "--tex-output",
            results_root / "suite_results.tex",
        ),
        [
            summary_path,
            results_root / "suite_summary.csv",
            results_root / "suite_results.tex",
        ],
        protocol="all",
        experiment="summary",
        seed=0,
        split="all",
        decoder="all",
    )
    paper_tables_dir = resolve(args.paper_tables_dir) if args.paper_tables_dir else (
        results_root / "paper_tables"
    )
    table_args: list[str | Path | int] = [
        "--summary",
        summary_path,
        "--output-dir",
        paper_tables_dir,
        "--primary-protocol",
        "session_disjoint" if "session_disjoint" in protocols else protocols[0],
        "--expected-seeds",
        *seeds,
        "--grounded-source",
        "human-audited" if args.profile == "official" else "crop-caption",
    ]
    if args.caption_quality_ledger is not None:
        table_args.extend(
            ["--caption-quality-ledger", resolve(args.caption_quality_ledger)]
        )
    table_outputs = [
        paper_tables_dir / filename
        for filename in (
            "main_results.tex",
            "ablation.tex",
            "peft_efficiency.tex",
            "robustness.tex",
            "caption_quality.tex",
            "table_manifest.json",
        )
    ]
    planner.add(
        "suite.export_paper_tables",
        "paper_table_export",
        python_command("export_paper_tables.py", *table_args),
        table_outputs,
        protocol="all",
        experiment="paper_tables",
        seed=0,
        split="all",
        decoder="all",
    )
    return planner.steps, blockers, plan_path, results_root


def save_plan(
    path: Path,
    *,
    args: argparse.Namespace,
    blockers: list[str],
    steps: list[Step],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": args.profile,
        "root": str(ROOT),
        "supervision_assumptions": {
            "crop_caption_is_grounded_caption": args.profile == "development",
            "crop_caption_supervision_tier": (
                "teacher_cropped_caption_not_human_audited"
                if args.profile == "development"
                else None
            ),
            "development_clear_counterfactual": (
                "ontology_definition_proxy" if args.profile == "development" else None
            ),
        },
        "blockers_not_automated": blockers,
        "steps": [asdict(step) for step in steps],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def outputs_exist(step: Step) -> bool:
    return bool(step.outputs) and all(Path(path).exists() for path in step.outputs)


def acquire_suite_lock(results_root: Path):
    results_root.mkdir(parents=True, exist_ok=True)
    lock_path = results_root / ".suite.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown process"
        handle.close()
        raise RuntimeError(
            f"Another experiment suite owns {lock_path}: {owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    handle.flush()
    return handle


def run_step(step: Step, logs_root: Path, env: dict[str, str]) -> None:
    log_path = logs_root / f"{step.step_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step.log = str(log_path)
    print(f"\n[{step.step_id}] {shlex.join(step.command)}", flush=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            step.command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, step.command)


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    _suite_lock = acquire_suite_lock(resolve(args.results_root))
    if not args.dry_run and not args.prepare_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use --prepare-only for data preparation, --dry-run "
                "to inspect the suite, or run on a GPU host."
            )
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_mib = free_bytes // 2**20
        total_mib = total_bytes // 2**20
        print(
            f"CUDA logical device 0: {free_mib} MiB free / {total_mib} MiB total "
            f"(physical selection CUDA_VISIBLE_DEVICES={args.cuda_devices})",
            flush=True,
        )
        if free_mib < args.min_free_gpu_mib:
            raise RuntimeError(
                f"Only {free_mib} MiB GPU memory is free; this paper suite requires at "
                f"least {args.min_free_gpu_mib} MiB before model loading. Stop other GPU "
                "processes or explicitly lower --min-free-gpu-mib after reviewing the risk."
            )
    try:
        steps, blockers, plan_path, results_root = make_plan(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"PRECHECK FAILED: {exc}") from exc
    save_plan(plan_path, args=args, blockers=blockers, steps=steps)

    print(f"Profile: {args.profile}")
    print(f"Plan: {plan_path}")
    print(f"Steps: {len(steps)}")
    if blockers:
        print("Inputs/experiments still outside automation:")
        for blocker in blockers:
            print(f"- {blocker}")
    if args.dry_run:
        if args.print_commands:
            for step in steps:
                print(f"[DRY-RUN] {step.step_id}: {shlex.join(step.command)}")
        else:
            print("Dry run complete; pass --print-commands or inspect the JSON plan.")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    logs_root = results_root / "logs"
    with tqdm(
        total=len(steps),
        desc="paper experiment suite",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as suite_progress:
        for step_index, step in enumerate(steps, 1):
            suite_progress.set_postfix_str(
                f"{step_index}/{len(steps)} {step.step_id}", refresh=True
            )
            if args.prepare_only and step.kind not in {
                "validation",
                "target_generation",
            }:
                step.status = "deferred_prepare_only"
                suite_progress.update(1)
                continue
            if args.resume and step.kind != "summary" and outputs_exist(step):
                step.status = "skipped_existing"
                tqdm.write(f"[SKIP] {step.step_id}")
                save_plan(plan_path, args=args, blockers=blockers, steps=steps)
                suite_progress.update(1)
                continue
            step.status = "running"
            save_plan(plan_path, args=args, blockers=blockers, steps=steps)
            try:
                run_step(step, logs_root, env)
            except Exception:
                step.status = "failed"
                save_plan(plan_path, args=args, blockers=blockers, steps=steps)
                raise
            step.status = "completed"
            save_plan(plan_path, args=args, blockers=blockers, steps=steps)
            suite_progress.update(1)
    if args.prepare_only:
        save_plan(plan_path, args=args, blockers=blockers, steps=steps)
        print(f"Preparation completed. Full plan: {plan_path}")
        return
    print(f"Completed. Summary: {results_root / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
