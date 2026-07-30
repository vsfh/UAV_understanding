#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare assigned-evidence deletion with size-matched non-assigned deletion "
            "using cached closed-set edge scores"
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--aggregator", choices=["max", "logsumexp"], default="logsumexp")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(values: list[float], aggregator: str, temperature: float) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty evidence set")
    if aggregator == "max":
        return max(values)
    maximum = max(values)
    return maximum + temperature * math.log(
        sum(math.exp((value - maximum) / temperature) for value in values)
    )


def stable_order(group_id: str, label: str, indices: list[int], seed: int) -> list[int]:
    return sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"{seed}:{group_id}:{label}:{index}".encode()
        ).digest(),
    )


def main() -> None:
    args = parse_args()
    source = json.loads(args.scores.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in source["predictions"]:
        grouped[row["group_id"]].append(row)

    comparisons = []
    skipped_no_remaining = 0
    skipped_no_matched_random = 0
    for group_id in sorted(grouped):
        rows = grouped[group_id]
        targets = sorted({row["target"] for row in rows})
        all_indices = list(range(len(rows)))
        for label in targets:
            assigned = [index for index, row in enumerate(rows) if row["target"] == label]
            non_assigned = [index for index in all_indices if index not in assigned]
            if not non_assigned:
                skipped_no_remaining += 1
                continue
            if len(non_assigned) < len(assigned):
                skipped_no_matched_random += 1
                continue
            random_deleted = set(
                stable_order(group_id, label, non_assigned, args.seed)[: len(assigned)]
            )
            assigned_deleted = set(assigned)
            full_score = aggregate(
                [row["scores"][label] for row in rows],
                args.aggregator,
                args.temperature,
            )
            assigned_removed_score = aggregate(
                [
                    row["scores"][label]
                    for index, row in enumerate(rows)
                    if index not in assigned_deleted
                ],
                args.aggregator,
                args.temperature,
            )
            random_removed_score = aggregate(
                [
                    row["scores"][label]
                    for index, row in enumerate(rows)
                    if index not in random_deleted
                ],
                args.aggregator,
                args.temperature,
            )
            assigned_drop = full_score - assigned_removed_score
            random_drop = full_score - random_removed_score
            comparisons.append(
                {
                    "group_id": group_id,
                    "label": label,
                    "num_edges": len(rows),
                    "num_deleted": len(assigned),
                    "assigned_drop": assigned_drop,
                    "random_drop": random_drop,
                    "gap": assigned_drop - random_drop,
                }
            )

    if not comparisons:
        raise ValueError(
            "No group has both assigned and enough non-assigned evidence edges"
        )
    assigned_drops = [row["assigned_drop"] for row in comparisons]
    random_drops = [row["random_drop"] for row in comparisons]
    gaps = [row["gap"] for row in comparisons]
    result = {
        "source": str(args.scores),
        "aggregator": args.aggregator,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_comparisons": len(comparisons),
        "num_groups": len({row["group_id"] for row in comparisons}),
        "mean_assigned_drop": sum(assigned_drops) / len(assigned_drops),
        "mean_random_drop": sum(random_drops) / len(random_drops),
        "mean_gap": sum(gaps) / len(gaps),
        "positive_gap_rate": sum(gap > 0 for gap in gaps) / len(gaps),
        "skipped_no_remaining": skipped_no_remaining,
        "skipped_no_matched_random": skipped_no_matched_random,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        key: value for key, value in result.items() if key != "comparisons"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
