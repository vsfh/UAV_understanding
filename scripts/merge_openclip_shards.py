#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TERMINAL_STATUSES = {"completed", "skipped_existing"}
EXCLUDED_KINDS = {"summary", "paper_table_export"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge completed OpenCLIP GPU-shard plans and export unified results"
    )
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_paths = sorted(args.shards_root.glob("**/suite_plan.json"))
    if not plan_paths:
        raise FileNotFoundError(f"No suite_plan.json files under {args.shards_root}")

    combined_steps: dict[str, dict] = {}
    blockers = set()
    for path in plan_paths:
        plan = json.loads(path.read_text(encoding="utf-8"))
        blockers.update(plan.get("blockers_not_automated", []))
        for step in plan["steps"]:
            if step["kind"] in EXCLUDED_KINDS:
                continue
            if step["status"] not in TERMINAL_STATUSES:
                raise RuntimeError(
                    f"Shard is incomplete: {path}: {step['step_id']}={step['status']}"
                )
            existing = combined_steps.get(step["step_id"])
            if existing is None:
                combined_steps[step["step_id"]] = step
            elif existing["command"] != step["command"] and step["kind"] != "validation":
                raise ValueError(f"Conflicting duplicate step id: {step['step_id']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_plan = args.output_dir / "combined_suite_plan.json"
    combined_plan.write_text(
        json.dumps(
            {
                "profile": "development",
                "root": str(ROOT),
                "supervision_assumptions": {
                    "openclip_same_backbone_comparison": True,
                    "full_finetune_scope": "vision_encoder_visual_projection_classifier",
                    "text_encoder_frozen": True,
                },
                "blockers_not_automated": sorted(blockers),
                "steps": list(combined_steps.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = args.output_dir / "suite_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_suite.py"),
            "--plan",
            str(combined_plan),
            "--output",
            str(summary),
            "--csv-output",
            str(args.output_dir / "suite_summary.csv"),
            "--tex-output",
            str(args.output_dir / "suite_results.tex"),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_paper_tables.py"),
            "--summary",
            str(summary),
            "--output-dir",
            str(args.output_dir / "paper_tables"),
            "--primary-protocol",
            "session_disjoint",
            "--expected-seeds",
            *[str(seed) for seed in args.expected_seeds],
            "--grounded-source",
            "crop-caption",
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        json.dumps(
            {
                "shard_plans": len(plan_paths),
                "combined_steps": len(combined_steps),
                "summary": str(summary),
                "paper_tables": str(args.output_dir / "paper_tables"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
