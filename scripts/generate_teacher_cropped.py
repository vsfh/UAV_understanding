#!/usr/bin/env python3
"""Generate descriptions for images stored in ``*/<event>/cropped``."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor


LONG_EDGE = 512
MIN_WORDS = 70
MAX_WORDS = 100
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_DATA_ROOT = Path("/media/data1/feihong/uav_understanding_data")
DEFAULT_MODEL = Path(
    "/media/4tb/feihong/hf_cache/"
    "models--Qwen--Qwen3.6-35B-A3B-FP8/"
    "snapshots/95a723d08a9490559dae23d0cff1d9466213d989"
)


def expected_summary(event: str) -> str:
    event = re.sub(r"[_-]+", " ", event.strip())
    return f"In summary, this image contains {event}."


@torch.inference_mode()
def generate_description(
    image_path: str | Path,
    commercial_event: str,
    teacher_model: Any,
    processor: Any,
    *,
    max_new_tokens: int = 180,
) -> str:
    """Resize one image to a 512-pixel long edge and generate its description."""

    readable_event = re.sub(r"[_-]+", " ", commercial_event.strip())
    summary = expected_summary(commercial_event)

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        scale = LONG_EDGE / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)

    prompt = f"""
Write one precise English paragraph describing visible content in this cropped UAV image.
Known commercial-event annotation: {readable_event}.

Describe visible entities or regions, count when reliable, physical appearance, materials,
colors, textures, shapes, condition or activity, surrounding scene elements, relative layout,
functional relations, and visual evidence supporting this event. Mention uncertainty whenever
pixels do not establish a detail.

Requirements:
- Write 70 to 100 English words in total.
- Describe only visible content. Do not invent identity, ownership, legality, intent, location,
  time, causes, or consequences.
- Do not use "drone view of", "aerial view of", standalone "the" or "a", or compass words such
  as north, south, east, and west.
- Use relative relations such as beside, within, along, near, around, or connected to.
- End with this exact sentence: {summary}
- Return one plain paragraph only.
""".strip()

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe cropped UAV imagery conservatively. Follow all constraints "
                        "and report only visible evidence."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {
        name: value.to(teacher_model.device) if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }
    output = teacher_model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    prompt_length = inputs["input_ids"].shape[1]
    description = processor.batch_decode(
        output[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    description = re.sub(
        r"<think>.*?</think>", "", description, flags=re.DOTALL | re.IGNORECASE
    )
    return re.sub(r"\s+", " ", description).strip().strip('"')


def check_description(
    description: str,
    commercial_event: str,
    *,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> dict[str, Any]:
    """Check length, forbidden wording, and the final commercial event."""

    words = re.findall(r"\b[^\W_]+(?:[-'][^\W_]+)*\b", description)
    lowered = description.casefold()
    forbidden_phrases = [
        phrase
        for phrase in ("drone view of", "aerial view of", "areial view of")
        if phrase in lowered
    ]
    forbidden_patterns = {
        "a": r"\ba\b",
        "the": r"\bthe\b",
        "north": r"\bnorth\w*\b",
        "south": r"\bsouth\w*\b",
        "east": r"\beast\w*\b",
        "west": r"\bwest\w*\b",
    }
    forbidden_words = [
        word
        for word, pattern in forbidden_patterns.items()
        if re.search(pattern, lowered, flags=re.IGNORECASE)
    ]
    summary = expected_summary(commercial_event)
    length_ok = min_words <= len(words) <= max_words
    summary_ok = description.rstrip().casefold().endswith(summary.casefold())

    errors = []
    if not length_ok:
        errors.append(f"word count {len(words)} is outside [{min_words}, {max_words}]")
    if forbidden_phrases:
        errors.append(f"forbidden phrases: {forbidden_phrases}")
    if forbidden_words:
        errors.append(f"forbidden words: {forbidden_words}")
    if not summary_ok:
        errors.append(f"description must end with: {summary}")

    return {
        "passed": not errors,
        "word_count": len(words),
        "length_ok": length_ok,
        "forbidden_phrases": forbidden_phrases,
        "forbidden_words": forbidden_words,
        "summary_ok": summary_ok,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    images = [
        (path, path.parent.parent.name)
        for path in sorted(args.data_root.rglob("*"))
        if path.is_file()
        and path.parent.name.casefold() == "cropped"
        and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise ValueError(f"No cropped images found in {args.data_root}")

    print(f"Found {len(images)} cropped images")
    output_dir = args.data_root / "description"
    if args.dry_run:
        for image_path, event in images[:5]:
            relative_path = image_path.relative_to(args.data_root)
            output_path = output_dir / relative_path.with_suffix(".json")
            print(f"{relative_path} -> {output_path}")
        return

    pending = []
    existing_rows = {}
    for image_path, event in images:
        relative_path = image_path.relative_to(args.data_root)
        output_path = output_dir / relative_path.with_suffix(".json")
        if output_path.is_file():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {
                    "image_path": relative_path.as_posix(),
                    "commercial_event": event,
                    "note": "Existing JSON could not be parsed but was skipped.",
                }
            existing["output_json_path"] = str(output_path.resolve())
            existing_rows[output_path] = existing
            continue
        pending.append((image_path, event, output_path))

    model = None
    processor = None
    if pending:
        if not torch.cuda.is_available() and not args.allow_cpu:
            raise RuntimeError("CUDA is unavailable; use --allow-cpu if intentional")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype="auto",
            device_map=args.device_map,
            local_files_only=True,
        )
        model.eval()

    for index, (image_path, event) in enumerate(images, 1):
        relative_path = image_path.relative_to(args.data_root)
        output_path = output_dir / relative_path.with_suffix(".json")
        if output_path in existing_rows:
            row = existing_rows[output_path]
            status = "skipped_existing_json"
        else:
            description = generate_description(
                image_path,
                event,
                model,
                processor,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                "image_path": relative_path.as_posix(),
                "output_json_path": str(output_path.resolve()),
                "commercial_event": event,
                "description": description,
                "check": check_description(description, event),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(output_path)
            status = "generated"

        progress = index / len(images)
        filled = round(30 * progress)
        progress_bar = "█" * filled + "-" * (30 - filled)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        print(
            f"[{progress_bar}] {index}/{len(images)} ({progress:.1%}) "
            f"{status} output_json_path={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
