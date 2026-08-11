#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from clear_uav.data import cap_per_class, read_samples
from clear_uav.ontology import load_label_subset, load_ontology


def caption_path(captions_root: Path, data_root: Path, evidence_path: Path) -> Path:
    relative = evidence_path.resolve().relative_to(data_root.resolve())
    return captions_root / relative.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert per-crop teacher captions into grounded-caption training targets"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--captions-root", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.yaml"))
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    labels = set(load_label_subset(args.labels_file, ontology))
    samples = read_samples(args.train_csv, args.data_root, include_labels=labels)
    samples = cap_per_class(samples, args.max_per_class, args.seed)
    captions_root = (
        args.captions_root
        if args.captions_root.is_absolute()
        else args.data_root / args.captions_root
    )

    rows = []
    for sample in tqdm(
        samples,
        desc="build grounded crop targets",
        unit="caption",
        dynamic_ncols=True,
    ):
        source = caption_path(captions_root, args.data_root, sample.evidence_path)
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing crop caption for {sample.record_uid}: {source}"
            )
        caption = json.loads(source.read_text(encoding="utf-8"))
        if caption.get("commercial_event") != sample.label:
            raise ValueError(
                f"Caption event mismatch for {sample.record_uid}: "
                f"{caption.get('commercial_event')!r} != {sample.label!r}"
            )
        expected_image = sample.evidence_path.resolve().relative_to(
            args.data_root.resolve()
        )
        caption_image = Path(str(caption.get("image_path", "")))
        if caption_image.parts[:1] == ("data",):
            caption_image = Path(*caption_image.parts[1:])
        if caption_image != expected_image:
            raise ValueError(
                f"Caption image mismatch for {sample.record_uid}: "
                f"{caption_image} != {expected_image}"
            )
        description = caption.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Empty caption for {sample.record_uid}: {source}")
        rows.append(
            {
                "record_uid": sample.record_uid,
                "source_caption": str(source.relative_to(args.data_root)),
                "caption_check": caption.get("check"),
                "target": {
                    "events": [sample.label],
                    "factors": {},
                    "evidence": description.strip(),
                    "uncertain": False,
                },
                "supervision_tier": "teacher_cropped_caption_not_human_audited",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} grounded-caption targets to {args.output}")


if __name__ == "__main__":
    main()
