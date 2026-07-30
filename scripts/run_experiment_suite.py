#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


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
        default=Path("/media/data1/feihong/uav_understanding_data"),
    )
    parser.add_argument("--models-root", type=Path, default=Path("models"))
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
    parser.add_argument("--candidate-batch-size", type=int, default=18)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        "--skip-free-generation",
        action="store_true",
        help="Run only the normalized-likelihood scorer",
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
        target = "__development__"
        experiments.extend(
            [
                Experiment("proxy_grounded_caption", "context", targets=target),
                Experiment(
                    "proxy_random_negative",
                    "pair",
                    targets=target,
                    extra_train_args=("--lambda-neighbor", "0.1", "--random-negative"),
                ),
                Experiment(
                    "proxy_graph_neighbor",
                    "pair",
                    targets=target,
                    extra_train_args=("--lambda-neighbor", "0.1"),
                ),
                Experiment(
                    "proxy_clear_no_dropout",
                    "pair",
                    targets=target,
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
                    targets=target,
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
        "PEFT rows other than the implemented LLM-only and projector+LLM LoRA require "
        "separate architecture-specific training code and hardware.",
        "Caption ratings, proposal/jitter crops, and challenge-slice annotations are not "
        "present; the suite cannot fabricate them.",
    ]
    data_root = resolve(args.data_root)
    labels_file = resolve(args.labels_file)
    qwen = resolve(args.models_root) / "qwen3-vl" / "config.json"
    openclip = resolve(args.models_root) / "openclip" / "config.json"
    required = [data_root, labels_file, qwen]
    if not args.skip_zero_shot:
        required.append(openclip)
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
        if experiment.targets not in {None, "__development__"}
    }
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
        args.batch_size,
        "--gradient-accumulation",
        1 if args.profile == "smoke" else args.gradient_accumulation,
        "--seed",
        seed,
    ]
    if max_samples is not None:
        train_args.extend(["--max-samples", max_samples])
    target_path = (
        targets_override
        if experiment.targets == "__development__"
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
        ("grounded_caption", "clear_full", "clear_vs_grounded"),
        ("random_negative", "graph_neighbor", "graph_vs_random"),
        ("graph_neighbor", "clear_no_dropout", "counterfactual_effect"),
        ("clear_no_dropout", "clear_full", "dropout_effect"),
        ("proxy_grounded_caption", "proxy_clear_full", "proxy_clear_vs_grounded"),
        ("proxy_random_negative", "proxy_graph_neighbor", "proxy_graph_vs_random"),
        (
            "proxy_graph_neighbor",
            "proxy_clear_no_dropout",
            "proxy_counterfactual_effect",
        ),
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
        "blockers_not_automated": blockers,
        "steps": [asdict(step) for step in steps],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def outputs_exist(step: Step) -> bool:
    return bool(step.outputs) and all(Path(path).exists() for path in step.outputs)


def run_step(step: Step, logs_root: Path, env: dict[str, str]) -> None:
    log_path = logs_root / f"{step.step_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step.log = str(log_path)
    print(f"\n[{step.step_id}] {shlex.join(step.command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            step.command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, step.command)


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
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

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use --dry-run to inspect the suite or run on a GPU host."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    logs_root = results_root / "logs"
    for step in steps:
        if args.resume and step.kind != "summary" and outputs_exist(step):
            step.status = "skipped_existing"
            print(f"[SKIP] {step.step_id}")
            save_plan(plan_path, args=args, blockers=blockers, steps=steps)
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
    print(f"Completed. Summary: {results_root / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
