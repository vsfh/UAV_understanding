#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRIC_KEYS = (
    "macro_f1",
    "micro_f1",
    "exact_set_accuracy",
    "worst_class_recall",
    "mean_average_precision",
    "micro_average_precision",
    "hard_negative_accuracy",
    "aurc",
)

DISPLAY_NAMES = {
    "qwen_direct": "Qwen3-VL direct",
    "qwen_definition": "Qwen3-VL definition",
    "openclip_direct": "OpenCLIP direct",
    "openclip_definition": "OpenCLIP definition",
    "label_pair_unweighted": "Label LoRA (pair, no token weight)",
    "label_pair_llm_lora": "LLM LoRA (pair)",
    "label_context": "Label LoRA (context)",
    "label_evidence": "Label LoRA (crop)",
    "label_pair": "Label LoRA (pair)",
    "generic_caption": "Generic-caption LoRA",
    "grounded_caption": "Grounded-caption LoRA",
    "random_negative": "Random-negative margin",
    "graph_neighbor": "Graph-neighbor margin",
    "clear_no_dropout": "CLEAR w/o view dropout",
    "clear_full": "CLEAR (full)",
    "proxy_grounded_caption": "Proxy grounded-caption LoRA",
    "proxy_random_negative": "Proxy random-negative margin",
    "proxy_graph_neighbor": "Proxy graph-neighbor margin",
    "proxy_clear_no_dropout": "Proxy CLEAR w/o dropout",
    "proxy_clear_full": "Proxy CLEAR (full)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a run_experiment_suite.py plan into JSON, CSV, and LaTeX"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    return parser.parse_args()


def numeric_metrics(payload: dict) -> dict[str, float]:
    return {
        key: float(payload[key])
        for key in METRIC_KEYS
        if key in payload and isinstance(payload[key], (int, float))
    }


def record_from_metadata(metadata: dict, decoder: str, metrics: dict, source: Path) -> dict:
    return {
        "protocol": metadata["protocol"],
        "experiment": metadata["experiment"],
        "seed": int(metadata["seed"]),
        "split": metadata["split"],
        "decoder": decoder,
        "source": str(source),
        "metrics": numeric_metrics(metrics),
    }


def collect_records(plan: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    metric_records = []
    comparisons = []
    training = []
    diagnostics = []
    for step in plan["steps"]:
        if step["status"] not in {"completed", "skipped_existing", "running"}:
            continue
        outputs = [Path(path) for path in step["outputs"]]
        metadata = step["metadata"]
        if step["kind"] == "analysis" and outputs and outputs[0].is_file():
            payload = json.loads(outputs[0].read_text(encoding="utf-8"))
            metric_records.append(
                record_from_metadata(
                    metadata, str(metadata["decoder"]), payload, outputs[0]
                )
            )
        elif (
            step["kind"] == "evaluation"
            and metadata["decoder"] == "closed"
            and outputs
            and outputs[0].is_file()
        ):
            payload = json.loads(outputs[0].read_text(encoding="utf-8"))
            metric_records.append(
                record_from_metadata(
                    metadata,
                    "set_logsumexp",
                    payload["metrics"]["set"],
                    outputs[0],
                )
            )
        elif step["kind"] == "set_rescore" and outputs and outputs[0].is_file():
            payload = json.loads(outputs[0].read_text(encoding="utf-8"))
            metric_records.append(
                record_from_metadata(metadata, "set_max", payload["metrics"], outputs[0])
            )
        elif step["kind"] == "comparison" and outputs and outputs[0].is_file():
            payload = json.loads(outputs[0].read_text(encoding="utf-8"))
            comparisons.append({**metadata, "source": str(outputs[0]), **payload})
        elif (
            step["kind"] in {"evidence_deletion", "view_reliance"}
            and outputs
            and outputs[0].is_file()
        ):
            payload = json.loads(outputs[0].read_text(encoding="utf-8"))
            diagnostics.append({**metadata, "source": str(outputs[0]), **payload})
        elif step["kind"] == "train" and len(outputs) > 1 and outputs[1].is_file():
            payload = json.loads(outputs[1].read_text(encoding="utf-8"))
            training.append({**metadata, "source": str(outputs[1]), **payload})
    return metric_records, comparisons, training, diagnostics


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["protocol"],
            record["experiment"],
            record["split"],
            record["decoder"],
        )
        grouped[key].append(record)

    summaries = []
    for key in sorted(grouped):
        rows = grouped[key]
        metrics = {}
        available_keys = sorted(
            set().union(*(row["metrics"].keys() for row in rows))
        )
        for metric in available_keys:
            values = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        summaries.append(
            {
                "protocol": key[0],
                "experiment": key[1],
                "split": key[2],
                "decoder": key[3],
                "num_seeds": len(rows),
                "seeds": [row["seed"] for row in rows],
                "metrics": metrics,
                "sources": [row["source"] for row in rows],
            }
        )
    return summaries


def write_csv(path: Path, summaries: list[dict]) -> None:
    fieldnames = [
        "protocol",
        "experiment",
        "split",
        "decoder",
        "num_seeds",
        "metric",
        "mean",
        "std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for metric, values in summary["metrics"].items():
                writer.writerow(
                    {
                        "protocol": summary["protocol"],
                        "experiment": summary["experiment"],
                        "split": summary["split"],
                        "decoder": summary["decoder"],
                        "num_seeds": summary["num_seeds"],
                        "metric": metric,
                        "mean": values["mean"],
                        "std": values["std"],
                    }
                )


def latex_value(summary: dict, metric: str, *, percentage: bool = True) -> str:
    if metric not in summary["metrics"]:
        return "--"
    values = summary["metrics"][metric]
    scale = 100 if percentage else 1
    mean = scale * values["mean"]
    std = scale * values["std"]
    if summary["num_seeds"] > 1:
        return f"{mean:.1f} $\\pm$ {std:.1f}"
    return f"{mean:.1f}" if percentage else f"{mean:.3f}"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_tex(path: Path, summaries: list[dict], training: list[dict]) -> None:
    split = "test" if any(row["split"] == "test" for row in summaries) else "val"
    protocol = (
        "session_disjoint"
        if any(row["protocol"] == "session_disjoint" for row in summaries)
        else summaries[0]["protocol"]
        if summaries
        else "unknown"
    )
    rows = [
        row
        for row in summaries
        if row["split"] == split
        and row["protocol"] == protocol
        and row["decoder"] in {"free", "score", "closed"}
    ]
    decoder_order = {"free": 0, "score": 1, "closed": 2}
    rows.sort(
        key=lambda row: (
            DISPLAY_NAMES.get(row["experiment"], row["experiment"]),
            decoder_order[row["decoder"]],
        )
    )
    lines = [
        "% Auto-generated by scripts/summarize_suite.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        (
            "\\caption{Automatically aggregated "
            f"{latex_escape(protocol)} {split} results. "
            "Values are mean $\\pm$ standard deviation over available seeds.}"
        ),
        "\\label{tab:auto_suite_results}",
        "\\footnotesize",
        "\\begin{tabularx}{\\textwidth}{@{}Y l c c c c c@{}}",
        "\\toprule",
        "Method & Decoder & Macro-F1 & Micro-F1 & mAP & Hard-neg. & AURC $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    DISPLAY_NAMES.get(
                        row["experiment"], latex_escape(row["experiment"])
                    ),
                    latex_escape(row["decoder"]),
                    latex_value(row, "macro_f1"),
                    latex_value(row, "micro_f1"),
                    latex_value(row, "mean_average_precision"),
                    latex_value(row, "hard_negative_accuracy"),
                    latex_value(row, "aurc", percentage=False),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\end{table*}",
            "",
        ]
    )
    peft_names = {"label_pair_llm_lora", "label_pair"}
    training_groups = defaultdict(list)
    for row in training:
        if row["protocol"] == protocol and row["experiment"] in peft_names:
            training_groups[row["experiment"]].append(row)
    if training_groups:
        lines.extend(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Automatically aggregated LoRA training efficiency.}",
                "\\label{tab:auto_suite_efficiency}",
                "\\footnotesize",
                "\\begin{tabularx}{\\columnwidth}{@{}Y r r r@{}}",
                "\\toprule",
                "Scope & Trainable M & Peak GB & Train pair/s \\\\",
                "\\midrule",
            ]
        )
        for experiment in ("label_pair_llm_lora", "label_pair"):
            rows_for_experiment = training_groups.get(experiment, [])
            if not rows_for_experiment:
                continue
            params = statistics.fmean(
                row["trainable_parameters"] for row in rows_for_experiment
            )
            memories = [
                row["peak_gpu_memory_bytes"]
                for row in rows_for_experiment
                if row["peak_gpu_memory_bytes"] is not None
            ]
            throughputs = [
                row["train_metrics"]["train_samples_per_second"]
                for row in rows_for_experiment
                if "train_samples_per_second" in row["train_metrics"]
            ]
            memory_text = (
                f"{statistics.fmean(memories) / 2**30:.1f}" if memories else "--"
            )
            throughput_text = (
                f"{statistics.fmean(throughputs):.2f}" if throughputs else "--"
            )
            lines.append(
                f"{DISPLAY_NAMES[experiment]} & {params / 1e6:.1f} & "
                f"{memory_text} & {throughput_text} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabularx}",
                "\\end{table}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    records, comparisons, training, diagnostics = collect_records(plan)
    summaries = aggregate(records)
    result = {
        "profile": plan["profile"],
        "blockers_not_automated": plan["blockers_not_automated"],
        "metric_records": records,
        "summaries": summaries,
        "comparisons": comparisons,
        "training": training,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.csv_output, summaries)
    write_tex(args.tex_output, summaries, training)
    print(
        json.dumps(
            {
                "metric_records": len(records),
                "summaries": len(summaries),
                "comparisons": len(comparisons),
                "training_runs": len(training),
                "diagnostics": len(diagnostics),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
