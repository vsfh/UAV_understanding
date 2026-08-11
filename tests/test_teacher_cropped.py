from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch
from PIL import Image


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_teacher_cropped.py"
SPEC = importlib.util.spec_from_file_location("generate_teacher_cropped", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

check_description = MODULE.check_description
expected_summary = MODULE.expected_summary
generate_description = MODULE.generate_description


class FakeProcessor:
    def __init__(self, description: str):
        self.description = description
        self.received_image_size: tuple[int, int] | None = None

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.received_image_size = messages[1]["content"][0]["image"].size
        return {"input_ids": torch.tensor([[1, 2]])}

    def batch_decode(self, generated, **kwargs):
        del generated, kwargs
        return [self.description]


class FakeModel:
    device = torch.device("cpu")

    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[1, 2, 3]])


def valid_description(event: str = "floating_garbage") -> str:
    body = " ".join(["visible"] * 70)
    return f"{body} {expected_summary(event)}"


def test_generate_description_resizes_long_edge_to_512(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (1024, 400), "white").save(image_path)
    processor = FakeProcessor(valid_description())

    result = generate_description(
        image_path,
        "floating_garbage",
        FakeModel(),
        processor,
    )

    assert processor.received_image_size == (512, 200)
    assert result == valid_description()


def test_check_description_accepts_valid_length_and_summary() -> None:
    report = check_description(valid_description(), "floating_garbage")

    assert report["passed"]
    assert 70 <= report["word_count"] <= 100
    assert report["summary_ok"]
    assert report["forbidden_words"] == []


def test_check_description_reports_forbidden_terms_and_short_length() -> None:
    description = (
        "A drone view of the northeastern road. "
        "In summary, this image contains floating garbage."
    )
    report = check_description(description, "floating_garbage")

    assert not report["passed"]
    assert not report["length_ok"]
    assert "drone view of" in report["forbidden_phrases"]
    assert {"a", "the", "north"} <= set(report["forbidden_words"])
    assert report["summary_ok"]


def test_check_description_rejects_wrong_event_or_extra_text() -> None:
    description = valid_description("waterside_garbage")
    report = check_description(description, "floating_garbage")

    assert not report["passed"]
    assert not report["summary_ok"]


def test_existing_json_is_skipped_even_when_check_failed(tmp_path: Path) -> None:
    image_path = tmp_path / "photos_batch" / "floating_garbage" / "cropped" / "one.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image_path)

    output_path = (
        tmp_path
        / "description"
        / "photos_batch"
        / "floating_garbage"
        / "cropped"
        / "one.json"
    )
    output_path.parent.mkdir(parents=True)
    description = "short invalid description"
    output_path.write_text(
        json.dumps(
            {
                "image_path": "photos_batch/floating_garbage/cropped/one.jpg",
                "commercial_event": "floating_garbage",
                "description": description,
                "check": {"passed": False},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--data-root",
            str(tmp_path),
            "--model-path",
            str(tmp_path / "unused-model"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f'"output_json_path": "{output_path}"' in result.stdout
    assert '"description":' in result.stdout
    completed_progress = (
        "[██████████████████████████████] 1/1 (100.0%) skipped_existing_json"
    )
    assert completed_progress in result.stdout
