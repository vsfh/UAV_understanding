#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether joint predictions require both views"
    )
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    return {row["record_uid"]: row for row in rows}


def predicted_labels(row: dict) -> set[str]:
    prediction = row["prediction"]
    return {prediction} if isinstance(prediction, str) else set(prediction)


def main() -> None:
    args = parse_args()
    variants = {
        "joint": load_rows(args.joint),
        "context": load_rows(args.context),
        "evidence": load_rows(args.evidence),
    }
    record_ids = set(variants["joint"])
    if any(set(rows) != record_ids for rows in variants.values()):
        raise ValueError("View prediction files contain different record_uids")
    ordered = sorted(record_ids)
    if any(
        len({variants[name][uid]["target"] for name in variants}) != 1
        for uid in ordered
    ):
        raise ValueError("View prediction files contain different targets")

    outcomes = []
    for uid in ordered:
        target = variants["joint"][uid]["target"]
        correct = {
            name: target in predicted_labels(rows[uid])
            for name, rows in variants.items()
        }
        outcomes.append({"record_uid": uid, "target": target, **correct})

    total = len(outcomes)
    joint_correct = sum(row["joint"] for row in outcomes)
    joint_only = sum(
        row["joint"] and not row["context"] and not row["evidence"]
        for row in outcomes
    )
    joint_harmed = sum(
        not row["joint"] and (row["context"] or row["evidence"])
        for row in outcomes
    )
    result = {
        "joint": str(args.joint),
        "context": str(args.context),
        "evidence": str(args.evidence),
        "num_rows": total,
        "joint_accuracy": joint_correct / total,
        "context_accuracy": sum(row["context"] for row in outcomes) / total,
        "evidence_accuracy": sum(row["evidence"] for row in outcomes) / total,
        "single_view_insufficiency_count": joint_only,
        "single_view_insufficiency_rate": joint_only / total,
        "single_view_insufficiency_given_joint_correct": (
            joint_only / joint_correct if joint_correct else 0.0
        ),
        "joint_harmed_count": joint_harmed,
        "joint_harmed_rate": joint_harmed / total,
        "outcomes": outcomes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "outcomes"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
