#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALLOWED_DECISIONS = {"accept", "correct", "reject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed teacher-review decisions and emit human_audited target JSONL"
        )
    )
    parser.add_argument("--generic-targets", type=Path, required=True)
    parser.add_argument("--grounded-targets", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--generic-output", type=Path, required=True)
    parser.add_argument("--grounded-output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index(rows: list[dict], source: Path) -> dict[str, dict]:
    result = {}
    for row in rows:
        uid = row["record_uid"]
        if uid in result:
            raise ValueError(f"Duplicate record_uid in {source}: {uid}")
        result[uid] = row
    return result


def read_reviews(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle, dialect="excel-tab"), 2):
            uid = row["record_uid"]
            if uid in result:
                raise ValueError(f"Duplicate review record at {path}:{line_number}: {uid}")
            result[uid] = row
    return result


def decision(row: dict[str, str], field: str, uid: str) -> str:
    value = row[field].strip().lower()
    if value not in ALLOWED_DECISIONS:
        raise ValueError(
            f"{uid}: {field} must be one of {sorted(ALLOWED_DECISIONS)}, got {value!r}"
        )
    if value == "reject":
        raise ValueError(
            f"{uid}: {field} is reject; correct or remove this sample from the frozen "
            "training manifest before finalization"
        )
    return value


def corrected_text(
    row: dict[str, str],
    *,
    decision_value: str,
    correction_field: str,
    original: str,
    uid: str,
) -> str:
    correction = row[correction_field].strip()
    if decision_value == "correct" and not correction:
        raise ValueError(f"{uid}: {correction_field} is required for a correction")
    value = correction if decision_value == "correct" else original
    if not value:
        raise ValueError(f"{uid}: reviewed text cannot be empty")
    return value


def parse_bool(value: str, field: str, uid: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{uid}: {field} must be true or false")
    return normalized == "true"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    generic_rows = read_jsonl(args.generic_targets)
    grounded_rows = read_jsonl(args.grounded_targets)
    generic = index(generic_rows, args.generic_targets)
    grounded = index(grounded_rows, args.grounded_targets)
    reviews = read_reviews(args.reviews)
    if set(generic) != set(grounded) or set(generic) != set(reviews):
        raise ValueError(
            "Generic targets, grounded targets, and reviews must have identical record coverage"
        )

    audited_generic = []
    audited_grounded = []
    for uid in [row["record_uid"] for row in generic_rows]:
        generic_row = generic[uid]
        grounded_row = grounded[uid]
        review = reviews[uid]
        immutable = {
            "source_class": generic_row["source_class"],
            "context_path": generic_row["context_path"],
            "evidence_path": generic_row["evidence_path"],
        }
        for field, expected in immutable.items():
            if review[field] != str(expected):
                raise ValueError(f"{uid}: immutable review field changed: {field}")
        reviewer_id = review["reviewer_id"].strip()
        if not reviewer_id:
            raise ValueError(f"{uid}: reviewer_id is required")

        generic_decision = decision(review, "generic_decision", uid)
        grounded_decision = decision(review, "grounded_decision", uid)
        counterfactual_decision = decision(review, "counterfactual_decision", uid)
        generic_evidence = corrected_text(
            review,
            decision_value=generic_decision,
            correction_field="corrected_generic_description",
            original=generic_row["target"]["evidence"],
            uid=uid,
        )
        grounded_evidence = corrected_text(
            review,
            decision_value=grounded_decision,
            correction_field="corrected_grounded_evidence",
            original=grounded_row["target"]["evidence"],
            uid=uid,
        )
        counterfactual_evidence = corrected_text(
            review,
            decision_value=counterfactual_decision,
            correction_field="corrected_counterfactual_evidence",
            original=grounded_row["counterfactual_target"]["evidence"],
            uid=uid,
        )
        grounded_uncertain = grounded_row["target"]["uncertain"]
        corrected_uncertain = review["corrected_grounded_uncertain"].strip()
        if grounded_decision == "correct":
            if not corrected_uncertain:
                raise ValueError(
                    f"{uid}: corrected_grounded_uncertain is required for correction"
                )
            grounded_uncertain = parse_bool(
                corrected_uncertain,
                "corrected_grounded_uncertain",
                uid,
            )
        elif corrected_uncertain:
            raise ValueError(
                f"{uid}: corrected_grounded_uncertain must be blank when accepting"
            )

        review_metadata = {
            "reviewer_id": reviewer_id,
            "generic_decision": generic_decision,
            "grounded_decision": grounded_decision,
            "counterfactual_decision": counterfactual_decision,
            "notes": review["notes"].strip(),
        }
        generic_output = {
            **generic_row,
            "target": {
                **generic_row["target"],
                "evidence": generic_evidence,
                "uncertain": grounded_uncertain,
            },
            "counterfactual_target": {
                **generic_row["counterfactual_target"],
                "evidence": counterfactual_evidence,
            },
            "supervision_tier": "human_audited",
            "human_review": review_metadata,
        }
        grounded_output = {
            **grounded_row,
            "target": {
                **grounded_row["target"],
                "evidence": grounded_evidence,
                "uncertain": grounded_uncertain,
            },
            "counterfactual_target": {
                **grounded_row["counterfactual_target"],
                "evidence": counterfactual_evidence,
            },
            "supervision_tier": "human_audited",
            "human_review": review_metadata,
        }
        audited_generic.append(generic_output)
        audited_grounded.append(grounded_output)

    write_jsonl(args.generic_output, audited_generic)
    write_jsonl(args.grounded_output, audited_grounded)
    print(f"wrote {len(audited_grounded)} human-audited targets")
    print(f"generic: {args.generic_output}")
    print(f"grounded: {args.grounded_output}")


if __name__ == "__main__":
    main()

