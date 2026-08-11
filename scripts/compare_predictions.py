#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired grouped-bootstrap model comparison")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--group-field", default="session_id")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines()]
    else:
        rows = json.loads(text)["predictions"]
    return {row["record_uid"]: row for row in rows}


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)


def grouped_counts(rows, record_ids, manifest, groups, labels, group_field):
    group_index = {group: index for index, group in enumerate(groups)}
    label_index = {label: index for index, label in enumerate(labels)}
    shape = (len(groups), len(labels))
    tp = np.zeros(shape, dtype=np.int64)
    fp = np.zeros(shape, dtype=np.int64)
    fn = np.zeros(shape, dtype=np.int64)
    for uid in record_ids:
        row = rows[uid]
        group = group_index[manifest[uid][group_field]]
        target = row["target"]
        prediction = row["prediction"]
        predicted = {prediction} if isinstance(prediction, str) else set(prediction)
        for label, index in label_index.items():
            tp[group, index] += label == target and label in predicted
            fp[group, index] += label != target and label in predicted
            fn[group, index] += label == target and label not in predicted
    return tp, fp, fn


def main() -> None:
    args = parse_args()
    labels = [
        line.strip()
        for line in args.labels_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["record_uid"]: row for row in csv.DictReader(handle)}
    left = load_rows(args.left)
    right = load_rows(args.right)
    if left.keys() != right.keys():
        raise ValueError("Prediction files contain different record_uids")
    record_ids = sorted(left)
    if any(left[uid]["target"] != right[uid]["target"] for uid in record_ids):
        raise ValueError("Prediction files contain different targets")
    groups = sorted({manifest[uid][args.group_field] for uid in record_ids})
    left_counts = grouped_counts(
        left, record_ids, manifest, groups, labels, args.group_field
    )
    right_counts = grouped_counts(
        right, record_ids, manifest, groups, labels, args.group_field
    )

    rng = np.random.default_rng(args.seed)
    macro_differences = []
    micro_differences = []
    for start in tqdm(
        range(0, args.resamples, 250),
        total=(args.resamples + 249) // 250,
        desc="paired bootstrap",
        unit="chunk",
        dynamic_ncols=True,
    ):
        count = min(250, args.resamples - start)
        weights = rng.multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=count
        )
        values = []
        for tp, fp, fn in (left_counts, right_counts):
            sample_tp = weights @ tp
            sample_fp = weights @ fp
            sample_fn = weights @ fn
            values.append(
                (
                    f1(sample_tp, sample_fp, sample_fn).mean(1),
                    f1(sample_tp.sum(1), sample_fp.sum(1), sample_fn.sum(1)),
                )
            )
        macro_differences.extend(values[1][0] - values[0][0])
        micro_differences.extend(values[1][1] - values[0][1])

    point_values = []
    for tp, fp, fn in (left_counts, right_counts):
        total_tp, total_fp, total_fn = tp.sum(0), fp.sum(0), fn.sum(0)
        point_values.append(
            (
                float(f1(total_tp, total_fp, total_fn).mean()),
                float(f1(total_tp.sum(), total_fp.sum(), total_fn.sum())),
            )
        )
    result = {
        "comparison": f"{args.right} - {args.left}",
        "num_rows": len(record_ids),
        "num_groups": len(groups),
        "macro_f1_difference": point_values[1][0] - point_values[0][0],
        "macro_f1_difference_ci95": np.percentile(
            macro_differences, [2.5, 97.5]
        ).tolist(),
        "micro_f1_difference": point_values[1][1] - point_values[0][1],
        "micro_f1_difference_ci95": np.percentile(
            micro_differences, [2.5, 97.5]
        ).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
