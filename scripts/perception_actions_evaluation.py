"""Standard-library-only checkpoint calibration and final-output validation."""
import hashlib
import json
import math
from pathlib import Path

CALIBRATION_VERSION = 2


def checkpoint_fingerprint(checkpoint):
    """Hash checkpoint file names and contents, including processor/config assets.

    File contents are streamed, so replacing weights at the same path invalidates
    calibration without loading a model or relying on modification timestamps.
    """
    root = Path(checkpoint)
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise FileNotFoundError(f"Checkpoint has no files: {root}")
    combined = hashlib.sha256(b"perception-actions-checkpoint-sha256-v1\0")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content.update(chunk)
        combined.update(len(relative).to_bytes(8, "big"))
        combined.update(relative)
        combined.update(content.digest())
    return combined.hexdigest()


def validate_calibration(calibration, checkpoint, fingerprint, max_test_samples=None):
    if (calibration.get("schema_version") != CALIBRATION_VERSION
            or not isinstance(calibration.get("checkpoint_fingerprint"), str)
            or type(calibration.get("limited_run")) is not bool):
        raise ValueError("Legacy/incomplete calibration; rerun validation to bind it to checkpoint contents")
    if calibration.get("checkpoint") != str(Path(checkpoint).resolve()):
        raise ValueError("Calibration checkpoint path differs; rerun validation for this checkpoint")
    if calibration["checkpoint_fingerprint"] != fingerprint:
        raise ValueError("Checkpoint contents changed after calibration; rerun validation")
    if calibration["limited_run"] and max_test_samples is None:
        raise ValueError("Refusing full test with tiny-run calibration; run full validation first")
    threshold = calibration.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("Calibration threshold must be a finite number")
    # A calibrated reject-all threshold can legally exceed 1 by one float ULP.
    return float(threshold)


def calibration_preflight(split, calibration_path, checkpoint, max_val_samples=None, max_test_samples=None):
    """Validate test prerequisites before allocating GPU memory or generating."""
    if split not in {"val", "test", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    for name, limit in [("max_val_samples", max_val_samples), ("max_test_samples", max_test_samples)]:
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError(f"{name} must be a positive integer when provided")
    if split == "all" and max_val_samples is not None and max_test_samples is None:
        raise ValueError("Refusing tiny validation followed by full test; use full validation or limit both splits")
    fingerprint = checkpoint_fingerprint(checkpoint)
    if split == "test":
        path = Path(calibration_path)
        if not path.is_file():
            raise FileNotFoundError(f"Run validation first to create {path}")
        calibration = json.loads(path.read_text(encoding="utf-8"))
        validate_calibration(calibration, checkpoint, fingerprint, max_test_samples)
    return fingerprint


def valid_final_event(prediction, threshold, labels=None):
    """Whether the deployed output is a valid nonempty supported event."""
    category = prediction.get("category")
    if (not isinstance(category, str) or not category or category == "no_event"
            or (labels is not None and category not in labels) or not prediction.get("valid", False)):
        return False
    box = prediction.get("bbox_1000")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return False
    try:
        coordinates = [float(value) for value in box]
        score = float(prediction["presence_score"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(score) or score < threshold or not all(math.isfinite(value) for value in coordinates):
        return False
    x1, y1, x2, y2 = coordinates
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
