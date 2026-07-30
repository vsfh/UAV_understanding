#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from clear_uav.data import iter_csv_rows, resolve_image_path
from clear_uav.ontology import load_label_subset, load_ontology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed validation for one curated split")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=["forward_temporal", "session_disjoint", "unseen_site"],
        required=True,
    )
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument(
        "--train-val-only",
        action="store_true",
        help="Validate only public train/validation manifests without opening private test labels",
    )
    return parser.parse_args()


def assert_disjoint(rows_by_split: dict[str, list[dict[str, str]]], field: str) -> None:
    sets = {
        split: {row[field] for row in rows} for split, rows in rows_by_split.items()
    }
    splits = list(sets)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(
                    f"{field} leaks between {left} and {right}: {sorted(overlap)[:5]}"
                )


def main() -> None:
    args = parse_args()
    protocol_dir = args.data_root / args.protocol
    paths = {
        "train": protocol_dir / "train.csv",
        "val": protocol_dir / "val.csv",
    }
    if not args.train_val_only:
        paths["test"] = protocol_dir / "test_inputs.csv"
    rows_by_split = {name: list(iter_csv_rows(path)) for name, path in paths.items()}
    private_labels = {}
    if not args.train_val_only:
        private_labels = {
            row["record_uid"]: row["source_class"]
            for row in iter_csv_rows(protocol_dir / "test_labels_private.csv")
        }
        if set(private_labels) != {
            row["record_uid"] for row in rows_by_split["test"]
        }:
            raise ValueError("Private test labels do not match test inputs one-to-one")

    ontology = load_ontology(args.ontology)
    ontology_labels = set(ontology.labels)
    if args.labels_file:
        included_labels = set(load_label_subset(args.labels_file, ontology))
        rows_by_split["train"] = [
            row for row in rows_by_split["train"] if row["source_class"] in included_labels
        ]
        rows_by_split["val"] = [
            row for row in rows_by_split["val"] if row["source_class"] in included_labels
        ]
        if not args.train_val_only:
            rows_by_split["test"] = [
                row
                for row in rows_by_split["test"]
                if private_labels[row["record_uid"]] in included_labels
            ]
    seen_uids: set[str] = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            uid = row["record_uid"]
            if uid in seen_uids:
                raise ValueError(f"record_uid occurs in multiple scored splits: {uid}")
            seen_uids.add(uid)
            if row["split"] != split:
                raise ValueError(f"{uid} has split={row['split']} inside {split}.csv")
            if row["protocol_version"] != "uts_uav_splits_v2_curated":
                raise ValueError(
                    f"Unexpected protocol version for {uid}: {row['protocol_version']}"
                )
            resolve_image_path(args.data_root, row["context_path"])
            resolve_image_path(args.data_root, row["evidence_path"])
            if split != "test" and row["source_class"] not in ontology_labels:
                raise ValueError(f"Unknown label for {uid}: {row['source_class']}")
        if not rows:
            raise ValueError(f"{paths[split]} is empty")

    if not args.train_val_only:
        unknown_test_labels = set(private_labels.values()) - ontology_labels
        if unknown_test_labels:
            raise ValueError(f"Unknown private test labels: {sorted(unknown_test_labels)}")
    assert_disjoint(rows_by_split, "content_group_id")
    if args.protocol == "forward_temporal":
        times = {
            split: [datetime.fromisoformat(row["detected_at"]) for row in rows]
            for split, rows in rows_by_split.items()
        }
        if max(times["train"]) >= min(times["val"]):
            raise ValueError("Forward-temporal train/validation dates overlap")
        if not args.train_val_only and max(times["val"]) >= min(times["test"]):
            raise ValueError("Forward-temporal validation/test dates overlap")
    if args.protocol == "session_disjoint":
        assert_disjoint(rows_by_split, "session_id")
    if args.protocol == "unseen_site":
        assert_disjoint(rows_by_split, "site_id")

    print(
        f"VALID {args.protocol}: "
        + ", ".join(f"{split}={len(rows):,}" for split, rows in rows_by_split.items())
    )


if __name__ == "__main__":
    main()
