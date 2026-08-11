from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).parents[1]
BUILD_SCRIPT = ROOT / "scripts/build_cropped_caption_targets.py"
MERGE_SCRIPT = ROOT / "scripts/merge_grounded_proxy_counterfactuals.py"
TRAIN_SCRIPT = ROOT / "scripts/train_qwen.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_qwen", TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crop_caption_becomes_grounded_target_without_fake_counterfactual(
    tmp_path: Path,
) -> None:
    event = "floating_garbage"
    image_relative = Path("photos_batch") / event / "cropped" / "one.jpg"
    context_relative = Path("photos_batch") / event / "one.jpg"
    for relative in (image_relative, context_relative):
        image_path = tmp_path / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), "white").save(image_path)

    train_csv = tmp_path / "session_disjoint" / "train.csv"
    train_csv.parent.mkdir()
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_uid",
                "source_class",
                "context_path",
                "evidence_path",
                "content_group_id",
                "session_id",
                "site_id",
                "split",
                "protocol_version",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_uid": "record-1",
                "source_class": event,
                "context_path": str(context_relative),
                "evidence_path": str(image_relative),
                "content_group_id": "content-1",
                "session_id": "session-1",
                "site_id": "site-1",
                "split": "train",
                "protocol_version": "test",
            }
        )

    description = "Visible floating debris is concentrated near the water surface."
    caption_path = tmp_path / "description" / image_relative.with_suffix(".json")
    caption_path.parent.mkdir(parents=True)
    caption_path.write_text(
        json.dumps(
            {
                "image_path": str(image_relative),
                "commercial_event": event,
                "description": description,
                "check": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text(event + "\n", encoding="utf-8")
    output_path = tmp_path / "targets.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--data-root",
            str(tmp_path),
            "--train-csv",
            str(train_csv),
            "--captions-root",
            "description",
            "--ontology",
            str(ROOT / "configs/ontology.yaml"),
            "--labels-file",
            str(labels_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["record_uid"] == "record-1"
    assert row["source_caption"] == str(
        Path("description") / image_relative.with_suffix(".json")
    )
    assert row["target"]["events"] == [event]
    assert row["target"]["evidence"] == description
    assert row["supervision_tier"] == "teacher_cropped_caption_not_human_audited"
    assert "counterfactual_target" not in row

    targets, counterfactuals, tiers = load_train_module().read_targets(output_path)
    assert json.loads(targets["record-1"])["evidence"] == description
    assert counterfactuals == {}
    assert tiers == {"record-1": "teacher_cropped_caption_not_human_audited"}

    counterfactual_run = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model-path",
            str(tmp_path / "unused-model"),
            "--data-root",
            str(tmp_path),
            "--train-csv",
            str(train_csv),
            "--ontology",
            str(ROOT / "configs/ontology.yaml"),
            "--labels-file",
            str(labels_path),
            "--targets-jsonl",
            str(output_path),
            "--output-dir",
            str(tmp_path / "unused-output"),
            "--lambda-cf",
            "0.1",
        ],
        capture_output=True,
        text=True,
    )
    assert counterfactual_run.returncode != 0
    assert "Counterfactual loss requires counterfactual_target" in (
        counterfactual_run.stdout + counterfactual_run.stderr
    )


def test_crop_grounded_positive_can_merge_with_explicit_proxy_counterfactual(
    tmp_path: Path,
) -> None:
    grounded_path = tmp_path / "grounded.jsonl"
    proxy_path = tmp_path / "proxy.jsonl"
    output_path = tmp_path / "merged.jsonl"
    grounded_path.write_text(
        json.dumps(
            {
                "record_uid": "record-1",
                "source_caption": "description/crop.json",
                "caption_check": {"passed": True},
                "target": {
                    "events": ["floating_garbage"],
                    "factors": {},
                    "evidence": "Visible debris is floating on water.",
                    "uncertain": False,
                },
                "supervision_tier": "teacher_cropped_caption_not_human_audited",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proxy_path.write_text(
        json.dumps(
            {
                "record_uid": "record-1",
                "target": {
                    "events": ["floating_garbage"],
                    "factors": {},
                    "evidence": "proxy positive is not used",
                    "uncertain": False,
                },
                "counterfactual_target": {
                    "events": ["waterside_garbage"],
                    "factors": {},
                    "evidence": "Debris lies beside water.",
                    "uncertain": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(MERGE_SCRIPT),
            "--grounded-targets",
            str(grounded_path),
            "--proxy-targets",
            str(proxy_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["target"]["evidence"] == "Visible debris is floating on water."
    assert row["counterfactual_target"]["events"] == ["waterside_garbage"]
    assert row["supervision_tier"] == "teacher_crop_grounded_with_proxy_counterfactual"
