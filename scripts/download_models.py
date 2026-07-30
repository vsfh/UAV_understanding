#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "qwen3-vl": "Qwen/Qwen3-VL-8B-Instruct",
    "openclip": "openai/clip-vit-large-patch14",
    "geochat": "MBZUAI/geochat-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one paper model from Hugging Face")
    parser.add_argument("preset", choices=MODELS)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id = MODELS[args.preset]
    destination = (args.models_root / args.preset).resolve()
    snapshot_download(
        repo_id=repo_id,
        revision=args.revision,
        local_dir=destination,
    )
    if not (destination / "config.json").is_file():
        raise FileNotFoundError(f"Downloaded snapshot has no config.json: {destination}")
    print(destination)


if __name__ == "__main__":
    main()

