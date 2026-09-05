#!/usr/bin/env python3
"""Recompute Table IV metrics from saved per-record predictions.

This avoids rerunning model inference when only the validation-selected
presence threshold or a threshold-independent metric definition changes.
Raw prediction rows are never modified; --write keeps a one-time backup.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from clear_uav.experiment_config import load_yaml_with_base
from clear_uav.table4 import labels_from_config, table4_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result_path = Path(args.result)
    calibration_path = Path(args.calibration)
    config = load_yaml_with_base(args.config)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold = float(calibration["threshold"])

    samples = [
        SimpleNamespace(
            presence=row["target"]["presence"],
            bbox_1000=row["target"]["bbox_1000"],
            label=row["target"]["category"],
        )
        for row in result["rows"]
    ]
    predictions = [row["prediction"] for row in result["rows"]]
    metrics = table4_metrics(
        samples, predictions, labels_from_config(config), threshold, True
    )
    print(json.dumps(metrics, indent=2))

    if not args.write:
        return
    backup = result_path.with_suffix(result_path.suffix + ".pre_recompute")
    if not backup.exists():
        shutil.copy2(result_path, backup)
    result["metrics"] = metrics
    result["metric_provenance"] = {
        "source": "saved_per_record_predictions",
        "calibration": str(calibration_path),
        "classification_metric": "positive_candidate_macro_f1_before_presence_gating",
        "raw_rows_modified": False,
    }
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(result_path)


if __name__ == "__main__":
    main()
