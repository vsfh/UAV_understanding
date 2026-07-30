from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REQUIRED_COLUMNS = {
    "record_uid",
    "source_class",
    "context_path",
    "evidence_path",
    "content_group_id",
    "session_id",
    "site_id",
    "split",
    "protocol_version",
}


@dataclass(frozen=True)
class Sample:
    record_uid: str
    label: str
    context_path: Path
    evidence_path: Path
    content_group_id: str
    session_id: str
    site_id: str


def resolve_image_path(data_root: Path, manifest_path: str) -> Path:
    relative = Path(manifest_path)
    if relative.parts[0] == "data":
        relative = Path(*relative.parts[1:])
    path = data_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_samples(
    csv_path: str | Path,
    data_root: str | Path,
    *,
    limit: int | None = None,
    include_labels: set[str] | None = None,
) -> list[Sample]:
    csv_path = Path(csv_path)
    data_root = Path(data_root)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            if include_labels is not None and row["source_class"] not in include_labels:
                continue
            rows.append(
                Sample(
                    record_uid=row["record_uid"],
                    label=row["source_class"],
                    context_path=resolve_image_path(data_root, row["context_path"]),
                    evidence_path=resolve_image_path(data_root, row["evidence_path"]),
                    content_group_id=row["content_group_id"],
                    session_id=row["session_id"],
                    site_id=row["site_id"],
                )
            )
            if limit is not None and len(rows) == limit:
                break
    return rows


def read_private_test_samples(
    inputs_csv: str | Path,
    labels_csv: str | Path,
    data_root: str | Path,
    *,
    limit: int | None = None,
    include_labels: set[str] | None = None,
) -> list[Sample]:
    private_labels = {
        row["record_uid"]: row["source_class"] for row in iter_csv_rows(labels_csv)
    }
    rows = []
    data_root = Path(data_root)
    for row in iter_csv_rows(inputs_csv):
        record_uid = row["record_uid"]
        if record_uid not in private_labels:
            raise ValueError(f"Missing private label for {record_uid}")
        if include_labels is not None and private_labels[record_uid] not in include_labels:
            continue
        rows.append(
            Sample(
                record_uid=record_uid,
                label=private_labels[record_uid],
                context_path=resolve_image_path(data_root, row["context_path"]),
                evidence_path=resolve_image_path(data_root, row["evidence_path"]),
                content_group_id=row["content_group_id"],
                session_id=row["session_id"],
                site_id=row["site_id"],
            )
        )
        if limit is not None and len(rows) == limit:
            break
    if len(private_labels) != sum(1 for _ in iter_csv_rows(inputs_csv)):
        raise ValueError("Test inputs and private labels have different row counts")
    return rows


def iter_csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def cap_per_class(samples: list[Sample], maximum: int, seed: int) -> list[Sample]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    selected = []
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda sample: hashlib.sha256(
                f"{seed}:{sample.record_uid}".encode()
            ).digest(),
        )
        selected.extend(ordered[:maximum])
    return selected
