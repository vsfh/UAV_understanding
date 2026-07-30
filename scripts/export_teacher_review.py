#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REVIEW_FIELDS = [
    "record_uid",
    "source_class",
    "context_path",
    "evidence_path",
    "automatic_audit_passed",
    "generic_description",
    "grounded_evidence",
    "grounded_uncertain",
    "counterfactual_event",
    "counterfactual_evidence",
    "generic_decision",
    "corrected_generic_description",
    "grounded_decision",
    "corrected_grounded_evidence",
    "corrected_grounded_uncertain",
    "counterfactual_decision",
    "corrected_counterfactual_evidence",
    "reviewer_id",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export teacher-generated targets to a strict human-review TSV"
    )
    parser.add_argument("--generic-targets", type=Path, required=True)
    parser.add_argument("--grounded-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def indexed(rows: list[dict], source: Path) -> dict[str, dict]:
    result = {}
    for row in rows:
        uid = row["record_uid"]
        if uid in result:
            raise ValueError(f"Duplicate record_uid in {source}: {uid}")
        result[uid] = row
    return result


def main() -> None:
    args = parse_args()
    generic_rows = read_jsonl(args.generic_targets)
    grounded_rows = read_jsonl(args.grounded_targets)
    generic = indexed(generic_rows, args.generic_targets)
    grounded = indexed(grounded_rows, args.grounded_targets)
    if set(generic) != set(grounded):
        raise ValueError("Generic and grounded target files have different record coverage")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=REVIEW_FIELDS,
            dialect="excel-tab",
            lineterminator="\n",
        )
        writer.writeheader()
        for uid in [row["record_uid"] for row in generic_rows]:
            generic_row = generic[uid]
            grounded_row = grounded[uid]
            if generic_row["source_class"] != grounded_row["source_class"]:
                raise ValueError(f"Source-class mismatch for {uid}")
            counterfactual = grounded_row["counterfactual_target"]
            writer.writerow(
                {
                    "record_uid": uid,
                    "source_class": grounded_row["source_class"],
                    "context_path": grounded_row["context_path"],
                    "evidence_path": grounded_row["evidence_path"],
                    "automatic_audit_passed": str(
                        grounded_row["automatic_audit_passed"]
                    ).lower(),
                    "generic_description": generic_row["target"]["evidence"],
                    "grounded_evidence": grounded_row["target"]["evidence"],
                    "grounded_uncertain": str(
                        grounded_row["target"]["uncertain"]
                    ).lower(),
                    "counterfactual_event": counterfactual["events"][0],
                    "counterfactual_evidence": counterfactual["evidence"],
                    "generic_decision": "",
                    "corrected_generic_description": "",
                    "grounded_decision": "",
                    "corrected_grounded_evidence": "",
                    "corrected_grounded_uncertain": "",
                    "counterfactual_decision": "",
                    "corrected_counterfactual_evidence": "",
                    "reviewer_id": "",
                    "notes": "",
                }
            )
    temporary.replace(args.output)
    print(f"wrote {len(generic_rows)} review rows to {args.output}")
    print("allowed decisions: accept, correct, reject")


if __name__ == "__main__":
    main()

