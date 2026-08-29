#!/usr/bin/env python3
"""Locate resized cropped images in their original images and export COCO boxes."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_ROOT = Path("um7")
DEFAULT_OUTPUT = Path("um7/crop_bboxes_all_photos")
SEED = 43


def locate(pair: tuple[Path, Path, str]) -> dict:
    original_path, cropped_path, category = pair
    original = cv2.imread(str(original_path))
    cropped = cv2.imread(str(cropped_path))
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, score, _, (x, y) = cv2.minMaxLoc(
        cv2.matchTemplate(original_gray, cropped_gray, cv2.TM_CCOEFF_NORMED)
    )
    height, width = cropped.shape[:2]

    return {
        "original_path": original_path,
        "cropped_path": cropped_path,
        "category": category,
        "width": original.shape[1],
        "height": original.shape[0],
        "bbox": [x, y, width, height],
        "score": float(score),
    }


def fit_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = np.full((512, 512, 3), 245, dtype=np.uint8)
    scale = min(512 / image.shape[1], 476 / image.shape[0])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x = (512 - shown.shape[1]) // 2
    y = 36 + (476 - shown.shape[0]) // 2
    panel[y : y + shown.shape[0], x : x + shown.shape[1]] = shown
    cv2.putText(panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return panel


def save_preview(record: dict, output_path: Path) -> None:
    original = cv2.imread(str(record["original_path"]))
    cropped = cv2.imread(str(record["cropped_path"]))
    x, y, w, h = record["bbox"]
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 12)
    score = record["score"]
    left = fit_panel(original, f"original  bbox={record['bbox']}  score={score:.3f}")
    right = fit_panel(cropped, f"cropped  {record['cropped_path'].name}")
    cv2.imwrite(str(output_path), np.hstack([left, right]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cropped_files = sorted(args.root.glob("photo*/*/cropped/*"))
    pairs = [
        (path.parent.parent / "original" / path.name, path, path.parent.parent.name)
        for path in cropped_files
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(tqdm(pool.map(locate, pairs), total=len(pairs), desc="Matching"))

    categories = {
        name: index + 1
        for index, name in enumerate(sorted({r["category"] for r in records}))
    }
    images, annotations = [], []
    for index, record in enumerate(records, start=1):
        bbox = record["bbox"]
        images.append(
            {
                "id": index,
                "file_name": str(record["original_path"].relative_to(args.root)),
                "width": record["width"],
                "height": record["height"],
            }
        )
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": categories[record["category"]],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "cropped_file_name": str(record["cropped_path"].relative_to(args.root)),
                "match_score": round(record["score"], 4),
            }
        )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name} for name, category_id in categories.items()
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "crop_bboxes.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for number, record in enumerate(random.Random(SEED).sample(records, 5), start=1):
        save_preview(record, args.output / f"preview_{number:02d}.jpg")

    print(f"Saved {len(records)} boxes to {args.output / 'crop_bboxes.json'}")


if __name__ == "__main__":
    main()
