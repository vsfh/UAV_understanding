#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from clear_uav.data import cap_per_class, read_samples
from clear_uav.ontology import load_label_subset, load_ontology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build definition-only targets for method smoke runs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_neighbor(uid: str, neighbors: tuple[str, ...]) -> str:
    value = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big")
    return neighbors[value % len(neighbors)]


def target(label: str, evidence: str) -> dict:
    return {
        "events": [label],
        "factors": {},
        "evidence": evidence,
        "uncertain": False,
    }


def main() -> None:
    args = parse_args()
    ontology = load_ontology(args.ontology)
    labels = set(load_label_subset(args.labels_file, ontology))
    samples = read_samples(args.train_csv, args.data_root, include_labels=labels)
    samples = cap_per_class(samples, args.max_per_class, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in tqdm(
            samples,
            desc="build proxy counterfactuals",
            unit="target",
            dynamic_ncols=True,
        ):
            neighbors = ontology.neighbors(sample.label)
            if not neighbors:
                raise ValueError(f"No graph neighbor for {sample.label}")
            negative = choose_neighbor(sample.record_uid, neighbors)
            row = {
                "record_uid": sample.record_uid,
                "target": target(
                    sample.label,
                    f"Visible criteria: {ontology.definitions[sample.label]}.",
                ),
                "counterfactual_target": target(
                    negative,
                    f"Visible criteria: {ontology.definitions[negative]}.",
                ),
                "supervision_tier": "definition_proxy_not_human_audited",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} proxy targets to {args.output}")


if __name__ == "__main__":
    main()
