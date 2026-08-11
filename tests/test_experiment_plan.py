from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_paper_plan_covers_every_tbd_table_row_and_keeps_roots_blank() -> None:
    plan = (ROOT / "EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
    paths = json.loads(
        (ROOT / "configs/paper_experiment_paths.json").read_text(encoding="utf-8")
    )

    assert set(paths["roots"].values()) == {""}
    assert all(
        not Path(relative).is_absolute()
        for relative in paths["relative_paths"].values()
    )

    required_rows = {
        "main_results.tex": [
            "OpenCLIP ViT-L/14",
            "GeoChat-7B",
            "Qwen3-VL-8B + definition",
            "Grounded-caption Qwen3-VL",
            "Two-view concatenation",
            "CLEAR / CLEAR-Set",
        ],
        "ablation.tex": [
            "Label-only LoRA",
            "Grounded-caption LoRA",
            "Random-negative margin",
            "Graph-neighbor margin",
            "CLEAR without view dropout",
            "CLEAR full",
        ],
        "peft_efficiency.tex": [
            "Linear probe",
            "Projector only",
            "LLM LoRA",
            "Projector + LLM LoRA",
            "QLoRA",
            "Full fine-tuning",
        ],
        "robustness.tex": [
            "small evidence",
            "adverse capture",
            "complex background",
            "unseen site",
            "proposal crop",
        ],
        "caption_quality.tex": [
            "Zero-shot Qwen",
            "Generic-caption LoRA",
            "Grounded-caption LoRA",
            "CLEAR",
        ],
    }
    for table, rows in required_rows.items():
        assert table in plan
        for row in rows:
            assert row.lower() in plan.lower(), f"{table} row missing from plan: {row}"

    headings = [line for line in plan.splitlines() if line.startswith("## ")]
    assert headings[-1] == "## 缺少的实验"

