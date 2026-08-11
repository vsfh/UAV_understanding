from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/export_paper_tables.py"


def summary_row(protocol: str, experiment: str, decoder: str, seeds: list[int]) -> dict:
    return {
        "protocol": protocol,
        "experiment": experiment,
        "split": "val",
        "decoder": decoder,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "metrics": {
            "macro_f1": {"mean": 0.5, "std": 0.01, "values": [0.5] * len(seeds)},
            "mean_average_precision": {
                "mean": 0.6,
                "std": 0.02,
                "values": [0.6] * len(seeds),
            },
            "hard_negative_accuracy": {
                "mean": 0.7,
                "std": 0.03,
                "values": [0.7] * len(seeds),
            },
            "aurc": {"mean": 0.2, "std": 0.01, "values": [0.2] * len(seeds)},
        },
        "sources": [],
    }


def test_exporter_maps_crop_caption_grounded_results_without_faking_missing_data(
    tmp_path: Path,
) -> None:
    seeds = [42, 43, 44]
    trained = [
        "label_context",
        "grounded_caption",
        "label_pair",
        "label_pair_llm_lora",
        "random_negative",
        "graph_neighbor",
        "proxy_clear_no_dropout",
        "proxy_clear_full",
    ]
    summaries = [
        summary_row("session_disjoint", "openclip_definition", "score", [0]),
        summary_row("session_disjoint", "qwen_definition", "closed", [0]),
    ]
    for experiment in trained:
        summaries.append(summary_row("session_disjoint", experiment, "closed", seeds))
        summaries.append(summary_row("session_disjoint", experiment, "set_max", seeds))
    for experiment in (
        "openclip_definition",
        "qwen_definition",
        "grounded_caption",
        "label_pair",
        "proxy_clear_full",
    ):
        summaries.append(
            summary_row(
                "unseen_site",
                experiment,
                "score" if experiment == "openclip_definition" else "closed",
                [0] if experiment in {"openclip_definition", "qwen_definition"} else seeds,
            )
        )
    training = []
    for experiment in ("label_pair_llm_lora", "label_pair"):
        for seed in seeds:
            training.append(
                {
                    "protocol": "session_disjoint",
                    "experiment": experiment,
                    "seed": seed,
                    "trainable_parameters": 10_000_000,
                    "peak_gpu_memory_bytes": 12 * 2**30,
                    "train_metrics": {"train_samples_per_second": 1.5},
                }
            )
    summary = {
        "profile": "development",
        "summaries": summaries,
        "training": training,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output_dir = tmp_path / "tables"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary_path),
            "--output-dir",
            str(output_dir),
            "--grounded-source",
            "crop-caption",
        ],
        check=True,
    )

    main = (output_dir / "main_results.tex").read_text(encoding="utf-8")
    ablation = (output_dir / "ablation.tex").read_text(encoding="utf-8")
    caption = (output_dir / "caption_quality.tex").read_text(encoding="utf-8")
    manifest = json.loads((output_dir / "table_manifest.json").read_text())
    assert "Grounded-caption Qwen3-VL" in main
    assert r"50.0 $\pm$ 1.0" in main
    assert "proxy CF in development" in main
    assert "Grounded-caption LoRA" in ablation
    assert r"\tbd" in caption
    assert manifest["grounded_caption_source"] == "crop-caption"
    assert manifest["scientific_status"] == "development_only_non_human_audited"
    assert any(
        cell["table"] == "caption_quality.tex"
        for cell in manifest["missing_cells"]
    )
