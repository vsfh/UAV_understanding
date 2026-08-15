#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


TBD = r"\tbd"
EXPECTED_CAPTION_METHODS = (
    "qwen_definition",
    "generic_caption",
    "grounded_caption",
    "clear_full",
)
CAPTION_ALIASES = {
    "zero-shot qwen3-vl": "qwen_definition",
    "qwen_definition": "qwen_definition",
    "generic-caption lora": "generic_caption",
    "generic_caption": "generic_caption",
    "grounded-caption lora": "grounded_caption",
    "grounded_caption": "grounded_caption",
    "clear": "clear_full",
    "clear_full": "clear_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map a completed experiment-suite summary into the five registered paper tables"
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-protocol", default="session_disjoint")
    parser.add_argument("--split", choices=["auto", "val", "test"], default="auto")
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--grounded-source",
        choices=["crop-caption", "human-audited"],
        default="crop-caption",
    )
    parser.add_argument(
        "--caption-quality-ledger",
        type=Path,
        help=(
            "Optional blinded TSV with item_id, method, rater_id, event_correct, "
            "evidence_supported, contradiction, hallucination"
        ),
    )
    return parser.parse_args()


class TableExporter:
    def __init__(self, args: argparse.Namespace, summary: dict) -> None:
        self.args = args
        self.summary = summary
        self.profile = str(summary.get("profile", "unknown"))
        self.expected_seeds = set(args.expected_seeds)
        self.split = self._select_split(args.split)
        self.index = {
            (
                row["protocol"],
                row["experiment"],
                row["split"],
                row["decoder"],
            ): row
            for row in summary.get("summaries", [])
        }
        self.missing_cells: list[dict[str, str]] = []
        self.warnings: list[str] = []

    def _select_split(self, requested: str) -> str:
        if requested != "auto":
            return requested
        return (
            "test"
            if any(row.get("split") == "test" for row in self.summary.get("summaries", []))
            else "val"
        )

    def find(
        self,
        experiment: str | None,
        decoder: str,
        *,
        protocol: str | None = None,
    ) -> dict | None:
        if experiment is None:
            return None
        return self.index.get(
            (
                protocol or self.args.primary_protocol,
                experiment,
                self.split,
                decoder,
            )
        )

    def missing(self, table: str, row: str, column: str, reason: str) -> str:
        self.missing_cells.append(
            {"table": table, "row": row, "column": column, "reason": reason}
        )
        return TBD

    def metric(
        self,
        summary: dict | None,
        metric: str,
        *,
        table: str,
        row: str,
        column: str,
        percentage: bool = True,
        require_three_seeds: bool = False,
    ) -> str:
        if summary is None:
            return self.missing(table, row, column, "experiment result is absent")
        values = summary.get("metrics", {}).get(metric)
        if values is None:
            return self.missing(table, row, column, f"metric {metric} is absent")
        if require_three_seeds:
            seeds = {int(seed) for seed in summary.get("seeds", [])}
            if seeds != self.expected_seeds:
                return self.missing(
                    table,
                    row,
                    column,
                    f"expected seeds {sorted(self.expected_seeds)}, found {sorted(seeds)}",
                )
        scale = 100.0 if percentage else 1.0
        mean = scale * float(values["mean"])
        std = scale * float(values["std"])
        if int(summary.get("num_seeds", 0)) > 1:
            precision = 1 if percentage else 3
            return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"
        return f"{mean:.1f}" if percentage else f"{mean:.3f}"

    def adapted_experiment(self, official: str, development: str | None = None) -> str:
        if self.profile == "official" or development is None:
            return official
        return development

    def main_table(self) -> str:
        table = "main_results.tex"
        clear_experiment = self.adapted_experiment("clear_full", "proxy_clear_full")
        rows = [
            (
                "OpenCLIP ViT-L/14",
                "zero-shot",
                "context",
                "openclip_definition",
                "score",
                False,
                False,
            ),
            (
                "OpenCLIP ViT-L/14",
                "linear probe",
                "context",
                "openclip_linear_probe",
                "score",
                True,
                False,
            ),
            (
                "OpenCLIP ViT-L/14",
                "full visual FT",
                "context",
                "openclip_full_finetune",
                "score",
                True,
                False,
            ),
            ("GeoChat-7B", "zero-shot", "context", None, "score", False, False),
            (
                "Qwen3-VL-8B + definition",
                "prompt",
                "context",
                "qwen_definition",
                "closed",
                False,
                False,
            ),
            (
                "Grounded-caption Qwen3-VL",
                "LoRA",
                "context",
                "grounded_caption",
                "closed",
                True,
                False,
            ),
            (
                "Two-view concatenation",
                "LoRA",
                "pair",
                "label_pair",
                "closed",
                True,
                True,
            ),
            (
                "CLEAR / CLEAR-Set (proxy CF in development)",
                "LoRA",
                "pair / set",
                clear_experiment,
                "closed",
                True,
                True,
            ),
        ]
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            (
                r"\caption{Automatically generated "
                + ("official test" if self.profile == "official" else "development validation")
                + r" results. Crop-teacher descriptions are treated as grounded captions. "
                + r"Unavailable inputs remain \texttt{TBD}; lower is better for AURC.}"
            ),
            r"\label{tab:auto_main}",
            r"\footnotesize",
            r"\begin{tabularx}{\textwidth}{@{}Y l l C{1.0cm} C{0.9cm} C{1.0cm} C{1.0cm} C{1.0cm} C{1.0cm} C{0.9cm}@{}}",
            r"\toprule",
            r"Method & Adapt. & Views & Macro F1 & mAP & Hard-neg. & Set F1 & Assign. & Unseen F1 & AURC $\downarrow$ \\",
            r"\midrule",
        ]
        for name, adapt, views, experiment, decoder, has_set, has_assignment in rows:
            pair = self.find(experiment, decoder)
            require_seeds = adapt == "LoRA"
            macro = self.metric(
                pair,
                "macro_f1",
                table=table,
                row=name,
                column="Macro F1",
                require_three_seeds=require_seeds,
            )
            map_value = self.metric(
                pair,
                "mean_average_precision",
                table=table,
                row=name,
                column="mAP",
                require_three_seeds=require_seeds,
            )
            hard_negative = self.metric(
                pair,
                "hard_negative_accuracy",
                table=table,
                row=name,
                column="Hard-neg.",
                require_three_seeds=require_seeds,
            )
            aurc = self.metric(
                pair,
                "aurc",
                table=table,
                row=name,
                column="AURC",
                percentage=False,
                require_three_seeds=require_seeds,
            )
            if has_set:
                set_summary = self.find(experiment, "set_max")
                set_f1 = self.metric(
                    set_summary,
                    "macro_f1",
                    table=table,
                    row=name,
                    column="Set F1",
                    require_three_seeds=require_seeds,
                )
            else:
                set_f1 = "--"
            assignment = (
                self.missing(
                    table,
                    name,
                    "Assign.",
                    "evidence-assignment annotations and metric are absent",
                )
                if has_assignment
                else "--"
            )
            unseen = self.metric(
                self.find(experiment, decoder, protocol="unseen_site"),
                "macro_f1",
                table=table,
                row=name,
                column="Unseen F1",
                require_three_seeds=require_seeds,
            )
            lines.append(
                " & ".join(
                    [
                        name,
                        adapt,
                        views,
                        macro,
                        map_value,
                        hard_negative,
                        set_f1,
                        assignment,
                        unseen,
                        aurc,
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
        return "\n".join(lines)

    def ablation_table(self) -> str:
        table = "ablation.tex"
        mappings = [
            ("Label-only LoRA", "label_context", "Yes", "No", "No", "No"),
            ("Grounded-caption LoRA", "grounded_caption", "Yes", "No", "No", "No"),
            ("Two-view concatenation", "label_pair", "Yes", "Yes", "No", "No"),
            (
                "Random-negative margin",
                "random_negative",
                "Yes",
                "Yes",
                "Random",
                "No",
            ),
            (
                "Graph-neighbor margin",
                "graph_neighbor",
                "Yes",
                "Yes",
                "Yes",
                "No",
            ),
            (
                "CLEAR without view dropout",
                self.adapted_experiment("clear_no_dropout", "proxy_clear_no_dropout"),
                "Yes",
                "Yes",
                "Yes",
                "Yes",
            ),
            (
                "CLEAR full",
                self.adapted_experiment("clear_full", "proxy_clear_full"),
                "Yes",
                "Yes",
                "Yes",
                "Yes",
            ),
        ]
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Automatically generated ablations. Development CLEAR rows use proxy counterfactual targets; crop captions supply the grounded-caption row.}",
            r"\label{tab:auto_ablation}",
            r"\footnotesize",
            r"\begin{tabularx}{\textwidth}{@{}Y c c c c C{1.25cm} C{1.25cm} C{1.25cm}@{}}",
            r"\toprule",
            r"Variant & Context & Crop & Graph neighbor & Counterfactual & Macro-F1 & Pair acc. & AURC $\downarrow$ \\",
            r"\midrule",
        ]
        for name, experiment, context, crop, neighbor, counterfactual in mappings:
            row = self.find(experiment, "closed")
            values = [
                self.metric(
                    row,
                    "macro_f1",
                    table=table,
                    row=name,
                    column="Macro-F1",
                    require_three_seeds=True,
                ),
                self.metric(
                    row,
                    "hard_negative_accuracy",
                    table=table,
                    row=name,
                    column="Pair acc.",
                    require_three_seeds=True,
                ),
                self.metric(
                    row,
                    "aurc",
                    table=table,
                    row=name,
                    column="AURC",
                    percentage=False,
                    require_three_seeds=True,
                ),
            ]
            lines.append(
                " & ".join([name, context, crop, neighbor, counterfactual, *values])
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
        return "\n".join(lines)

    def training_group(self, experiment: str) -> list[dict]:
        return [
            row
            for row in self.summary.get("training", [])
            if row.get("protocol") == self.args.primary_protocol
            and row.get("experiment") == experiment
            and int(row.get("seed", -1)) in self.expected_seeds
        ]

    def training_value(
        self,
        rows: list[dict],
        key: str,
        *,
        table: str,
        row_name: str,
        column: str,
        scale: float = 1.0,
        precision: int = 1,
    ) -> str:
        seeds = {int(row["seed"]) for row in rows}
        if seeds != self.expected_seeds:
            return self.missing(
                table,
                row_name,
                column,
                f"expected training seeds {sorted(self.expected_seeds)}, found {sorted(seeds)}",
            )
        values = []
        for row in rows:
            value = row
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                values.append(float(value) / scale)
        if len(values) != len(rows):
            return self.missing(table, row_name, column, f"training field {key} is absent")
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"

    def peft_table(self) -> str:
        table = "peft_efficiency.tex"
        implemented = {
            "LLM LoRA": "label_pair_llm_lora",
            "Projector + LLM LoRA": "label_pair",
        }
        names = [
            "Linear probe",
            "Projector only",
            "LLM LoRA",
            "Projector + LLM LoRA",
            "QLoRA",
            "Full fine-tuning",
        ]
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Automatically generated accuracy--efficiency results on the same Qwen3-VL backbone.}",
            r"\label{tab:auto_peft}",
            r"\footnotesize",
            r"\begin{tabularx}{\columnwidth}{@{}Y C{1.15cm} C{1.05cm} C{1.05cm} C{1.05cm}@{}}",
            r"\toprule",
            r"Trainable modules & Params. (M) & Peak GB & pair/s & Macro-F1 \\",
            r"\midrule",
        ]
        for name in names:
            experiment = implemented.get(name)
            if experiment is None:
                values = [
                    self.missing(table, name, column, "training mode is not implemented")
                    for column in ("Params.", "Peak GB", "pair/s", "Macro-F1")
                ]
            else:
                training = self.training_group(experiment)
                values = [
                    self.training_value(
                        training,
                        "trainable_parameters",
                        table=table,
                        row_name=name,
                        column="Params.",
                        scale=1e6,
                    ),
                    self.training_value(
                        training,
                        "peak_gpu_memory_bytes",
                        table=table,
                        row_name=name,
                        column="Peak GB",
                        scale=2**30,
                    ),
                    self.training_value(
                        training,
                        "train_metrics.train_samples_per_second",
                        table=table,
                        row_name=name,
                        column="pair/s",
                        precision=2,
                    ),
                    self.metric(
                        self.find(experiment, "closed"),
                        "macro_f1",
                        table=table,
                        row=name,
                        column="Macro-F1",
                        require_three_seeds=True,
                    ),
                ]
            lines.append(" & ".join([name, *values]) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
        return "\n".join(lines)

    def robustness_table(self) -> str:
        table = "robustness.tex"
        mappings = [
            ("Qwen3-VL + definition", "qwen_definition", "closed", False),
            ("Grounded-caption LoRA", "grounded_caption", "closed", True),
            ("Two-view concatenation", "label_pair", "closed", True),
            (
                "CLEAR",
                self.adapted_experiment("clear_full", "proxy_clear_full"),
                "closed",
                True,
            ),
        ]
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Automatically generated stress-test macro-F1. Missing reviewed slices and proposal crops remain \texttt{TBD}.}",
            r"\label{tab:auto_robustness}",
            r"\footnotesize",
            r"\begin{tabularx}{\textwidth}{@{}Y C{1.15cm} C{1.15cm} C{1.15cm} C{1.15cm} C{1.15cm} C{1.15cm}@{}}",
            r"\toprule",
            r"Method & All & Small evidence & Adverse capture & Complex bg. & Unseen site & Proposal crop \\",
            r"\midrule",
        ]
        for name, experiment, decoder, require_seeds in mappings:
            all_value = self.metric(
                self.find(experiment, decoder),
                "macro_f1",
                table=table,
                row=name,
                column="All",
                require_three_seeds=require_seeds,
            )
            unavailable = [
                self.missing(table, name, column, reason)
                for column, reason in (
                    ("Small evidence", "reviewed slice annotation is absent"),
                    ("Adverse capture", "reviewed slice annotation is absent"),
                    ("Complex bg.", "reviewed slice annotation is absent"),
                )
            ]
            unseen = self.metric(
                self.find(experiment, decoder, protocol="unseen_site"),
                "macro_f1",
                table=table,
                row=name,
                column="Unseen site",
                require_three_seeds=require_seeds,
            )
            proposal = self.missing(
                table, name, "Proposal crop", "proposal-crop manifest is absent"
            )
            lines.append(
                " & ".join([name, all_value, *unavailable, unseen, proposal]) + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
        return "\n".join(lines)

    @staticmethod
    def binary(value: str, *, field: str, line: int) -> int:
        normalized = value.strip().lower()
        if normalized in {"1", "yes", "true", "y"}:
            return 1
        if normalized in {"0", "no", "false", "n"}:
            return 0
        raise ValueError(f"Invalid binary {field} at ledger line {line}: {value!r}")

    def caption_scores(self) -> dict[str, dict[str, float]]:
        path = self.args.caption_quality_ledger
        if path is None:
            return {}
        required = {
            "item_id",
            "method",
            "rater_id",
            "event_correct",
            "evidence_supported",
            "contradiction",
            "hallucination",
        }
        grouped: dict[tuple[str, str], list[dict[str, int | str]]] = defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Caption ledger is missing columns: {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                method_key = row["method"].strip().lower()
                method = CAPTION_ALIASES.get(method_key)
                if method is None:
                    raise ValueError(f"Unknown caption method at ledger line {line}: {row['method']}")
                parsed: dict[str, int | str] = {"rater_id": row["rater_id"].strip()}
                for field in (
                    "event_correct",
                    "evidence_supported",
                    "contradiction",
                    "hallucination",
                ):
                    parsed[field] = self.binary(row[field], field=field, line=line)
                grouped[(method, row["item_id"].strip())].append(parsed)

        item_scores: dict[str, list[dict[str, int]]] = defaultdict(list)
        for (method, item_id), ratings in grouped.items():
            raters = {str(row["rater_id"]) for row in ratings}
            if len(ratings) != 3 or len(raters) != 3:
                raise ValueError(
                    f"Caption ledger requires three distinct raters for {method}/{item_id}"
                )
            item_scores[method].append(
                {
                    field: int(sum(int(row[field]) for row in ratings) >= 2)
                    for field in (
                        "event_correct",
                        "evidence_supported",
                        "contradiction",
                        "hallucination",
                    )
                }
            )
        missing_methods = set(EXPECTED_CAPTION_METHODS) - set(item_scores)
        if missing_methods:
            raise ValueError(
                f"Caption ledger is missing methods: {sorted(missing_methods)}"
            )
        return {
            method: {
                field: statistics.fmean(row[field] for row in rows)
                for field in (
                    "event_correct",
                    "evidence_supported",
                    "contradiction",
                    "hallucination",
                )
            }
            for method, rows in item_scores.items()
        }

    def caption_table(self) -> str:
        table = "caption_quality.tex"
        scores = self.caption_scores()
        mappings = [
            ("Zero-shot Qwen3-VL", "qwen_definition"),
            ("Generic-caption LoRA", "generic_caption"),
            ("Grounded-caption LoRA", "grounded_caption"),
            ("CLEAR", "clear_full"),
        ]
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Blinded evidence-statement evaluation. Each item is majority-voted by three distinct raters.}",
            r"\label{tab:auto_caption}",
            r"\footnotesize",
            r"\begin{tabularx}{\columnwidth}{@{}Y C{1.05cm} C{1.05cm} C{1.05cm} C{1.05cm}@{}}",
            r"\toprule",
            r"Method & Event correct & Evidence supported & Contrad. $\downarrow$ & Halluc. $\downarrow$ \\",
            r"\midrule",
        ]
        for name, method in mappings:
            if method not in scores:
                values = [
                    self.missing(
                        table,
                        name,
                        column,
                        "three-rater blinded caption ledger was not supplied",
                    )
                    for column in (
                        "Event correct",
                        "Evidence supported",
                        "Contrad.",
                        "Halluc.",
                    )
                ]
            else:
                values = [
                    f"{100 * scores[method][field]:.1f}"
                    for field in (
                        "event_correct",
                        "evidence_supported",
                        "contradiction",
                        "hallucination",
                    )
                ]
            lines.append(" & ".join([name, *values]) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
        return "\n".join(lines)

    def write(self) -> None:
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        contents = {
            "main_results.tex": self.main_table(),
            "ablation.tex": self.ablation_table(),
            "peft_efficiency.tex": self.peft_table(),
            "robustness.tex": self.robustness_table(),
            "caption_quality.tex": self.caption_table(),
        }
        for name, content in contents.items():
            (self.args.output_dir / name).write_text(content, encoding="utf-8")
        manifest = {
            "source_summary": str(self.args.summary.resolve()),
            "profile": self.profile,
            "split": self.split,
            "primary_protocol": self.args.primary_protocol,
            "expected_seeds": sorted(self.expected_seeds),
            "grounded_caption_source": self.args.grounded_source,
            "supervision_assumptions": self.summary.get(
                "supervision_assumptions", {}
            ),
            "scientific_status": (
                "development_only_non_human_audited"
                if self.args.grounded_source == "crop-caption"
                else "human_audited"
            ),
            "generated_tables": sorted(contents),
            "missing_cells": self.missing_cells,
            "warnings": self.warnings,
        }
        (self.args.output_dir / "table_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "output_dir": str(self.args.output_dir),
                    "tables": len(contents),
                    "missing_cells": len(self.missing_cells),
                    "scientific_status": manifest["scientific_status"],
                },
                indent=2,
            )
        )


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    TableExporter(args, summary).write()


if __name__ == "__main__":
    main()
