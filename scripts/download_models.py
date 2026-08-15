#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download


DEFAULT_MODELS_ROOT = Path("hf_cache")


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    directory: str
    allow_patterns: tuple[str, ...]
    weight_patterns: tuple[str, ...]
    note: str


PAPER_MODELS = {
    "qwen3-vl": ModelSpec(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        directory="qwen3-vl",
        allow_patterns=(
            "*.json",
            "*.jinja",
            "*.model",
            "*.safetensors",
            "*.tiktoken",
            "*.txt",
            "*.py",
            "README.md",
        ),
        weight_patterns=("*.safetensors",),
        note="Primary Qwen3-VL-8B-Instruct student and zero-shot baseline",
    ),
    "openclip": ModelSpec(
        repo_id="openai/clip-vit-large-patch14",
        directory="openclip",
        allow_patterns=(
            "config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "README.md",
        ),
        weight_patterns=("model.safetensors",),
        note="Paper OpenCLIP ViT-L/14 zero-shot baseline",
    ),
    "geochat": ModelSpec(
        repo_id="MBZUAI/geochat-7B",
        directory="geochat",
        allow_patterns=("*.json", "*.model", "*.bin", "README.md"),
        weight_patterns=("pytorch_model*.bin",),
        note="Original GeoChat-7B remote-sensing baseline",
    ),
    "uavit": ModelSpec(
        repo_id="ZhanYang-nwpu/GeoChat-UAV",
        directory="uavit-1m",
        allow_patterns=("*.json", "*.model", "*.bin", "README.md"),
        weight_patterns=("pytorch_model*.bin",),
        note="Official GeoChat checkpoint adapted on UAVIT-1M",
    ),
    "geochat-vision-tower": ModelSpec(
        repo_id="openai/clip-vit-large-patch14-336",
        directory="geochat-vision-tower",
        allow_patterns=(
            "config.json",
            "merges.txt",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "README.md",
        ),
        weight_patterns=("pytorch_model.bin",),
        note="Vision tower required by both GeoChat checkpoints",
    ),
}

DEFAULT_PRESETS = (
    "qwen3-vl",
    "openclip",
    "geochat",
    "uavit",
    "geochat-vision-tower",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the paper model checkpoints from Hugging Face"
    )
    parser.add_argument(
        "presets",
        nargs="*",
        help=(
            "Models to download; omit or use 'all' for every paper model. "
            f"Choices: {', '.join(PAPER_MODELS)}"
        ),
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=["huggingface", "aria2"],
        default="huggingface",
        help="Use huggingface_hub, or direct mirror downloads through aria2",
    )
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--concurrent-files", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_presets(values: list[str]) -> tuple[str, ...]:
    if not values or values == ["all"]:
        return DEFAULT_PRESETS
    if "all" in values:
        raise ValueError("Use 'all' alone or list individual presets")
    unknown = sorted(set(values) - set(PAPER_MODELS))
    if unknown:
        raise ValueError(f"Unknown model presets: {unknown}")
    return tuple(dict.fromkeys(values))


def validate_download(destination: Path, spec: ModelSpec) -> None:
    if not (destination / "config.json").is_file():
        raise FileNotFoundError(f"Downloaded snapshot has no config.json: {destination}")
    if not any(
        path.is_file()
        for pattern in spec.weight_patterns
        for path in destination.glob(pattern)
    ):
        raise FileNotFoundError(
            f"Downloaded snapshot has no model weights matching "
            f"{spec.weight_patterns}: {destination}"
        )


def mirror_file_list(
    payload: dict, spec: ModelSpec
) -> list[tuple[str, int, str | None]]:
    selected = []
    for sibling in payload.get("siblings", []):
        filename = sibling.get("rfilename")
        size = sibling.get("size")
        if not isinstance(filename, str) or not isinstance(size, int):
            continue
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe repository path: {filename!r}")
        if any(fnmatch.fnmatch(filename, pattern) for pattern in spec.allow_patterns):
            lfs = sibling.get("lfs")
            sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
            selected.append(
                (filename, size, sha256 if isinstance(sha256, str) else None)
            )
    if not selected:
        raise ValueError(f"Mirror metadata contains no selected files for {spec.repo_id}")
    return selected


def pending_mirror_files(
    files: list[tuple[str, int, str | None]], destination: Path
) -> list[tuple[str, int, str | None]]:
    pending = []
    for filename, expected_size, sha256 in files:
        output = destination / filename
        control = output.with_name(output.name + ".aria2")
        if (
            output.is_file()
            and output.stat().st_size == expected_size
            and not control.exists()
        ):
            continue
        pending.append((filename, expected_size, sha256))
    return pending


def write_aria2_input(
    *,
    files: list[tuple[str, int, str | None]],
    destination: Path,
    endpoint: str,
    repo_id: str,
    revision: str,
    input_path: Path,
) -> None:
    records = []
    for filename, _, _ in files:
        output = destination / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"{endpoint.rstrip('/')}/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
        )
        records.extend(
            [
                url,
                f"  dir={output.parent}",
                f"  out={output.name}",
            ]
        )
    input_path.write_text("\n".join(records) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mirror_checksums(
    files: list[tuple[str, int, str | None]], destination: Path
) -> None:
    checked = 0
    for filename, _, expected in files:
        if expected is None:
            continue
        actual = sha256_file(destination / filename)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {destination / filename}: "
                f"{actual} != {expected}"
            )
        checked += 1
    if checked:
        print(f"  verified SHA-256 for {checked} downloaded weight file(s)")


def aria2_command(
    *,
    input_path: Path,
    log_path: Path,
    connections: int,
    concurrent_files: int,
) -> list[str]:
    return [
        "aria2c",
        f"--input-file={input_path}",
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        f"--max-concurrent-downloads={concurrent_files}",
        "--min-split-size=1M",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--file-allocation=none",
        "--max-tries=0",
        "--retry-wait=3",
        "--summary-interval=5",
        "--console-log-level=warn",
        "--log-level=notice",
        f"--log={log_path}",
    ]


def download_with_aria2(
    *,
    spec: ModelSpec,
    destination: Path,
    endpoint: str,
    revision: str,
    connections: int,
    concurrent_files: int,
) -> None:
    if shutil.which("aria2c") is None:
        raise FileNotFoundError("Mirror backend requires aria2c in PATH")
    if not 1 <= connections <= 16:
        raise ValueError("--connections must be between 1 and 16")
    if not 1 <= concurrent_files <= 16:
        raise ValueError("--concurrent-files must be between 1 and 16")

    endpoint = endpoint.rstrip("/")
    api_url = (
        f"{endpoint}/api/models/{quote(spec.repo_id, safe='/')}/revision/"
        f"{quote(revision, safe='')}?blobs=true"
    )
    request = Request(api_url, headers={"User-Agent": "clear-uav-model-downloader/1"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    files = mirror_file_list(payload, spec)

    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / ".aria2_download_urls.txt"
    log_path = destination / ".aria2_download.log"
    pending = pending_mirror_files(files, destination)
    files_to_verify = list(pending)
    if not pending:
        print(f"  mirror files already complete: {len(files)}")
        return

    write_aria2_input(
        files=pending,
        destination=destination,
        endpoint=endpoint,
        repo_id=spec.repo_id,
        revision=revision,
        input_path=input_path,
    )
    command = aria2_command(
        input_path=input_path,
        log_path=log_path,
        connections=connections,
        concurrent_files=concurrent_files,
    )
    print(
        f"  aria2: {len(pending)} pending files, "
        f"{connections} connections/file, {concurrent_files} concurrent files",
        flush=True,
    )
    result = subprocess.run(command, check=False)
    if result.returncode:
        pending = pending_mirror_files(files, destination)
        if pending:
            fallback_endpoint = "https://huggingface.co"
            print(
                f"  mirror left {len(pending)} incomplete file(s); retrying through "
                f"{fallback_endpoint} with one connection. log: {log_path}",
                flush=True,
            )
            write_aria2_input(
                files=pending,
                destination=destination,
                endpoint=fallback_endpoint,
                repo_id=spec.repo_id,
                revision=revision,
                input_path=input_path,
            )
            fallback_command = aria2_command(
                input_path=input_path,
                log_path=log_path,
                connections=1,
                concurrent_files=1,
            )
            subprocess.run(fallback_command, check=True)

    pending = pending_mirror_files(files, destination)
    if pending:
        raise RuntimeError(
            f"aria2 left {len(pending)} incomplete file(s); see {log_path}"
        )
    validate_mirror_checksums(files_to_verify, destination)
    input_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    try:
        presets = selected_presets(args.presets)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    models_root = args.models_root.resolve()
    print(f"models_root: {models_root}")
    print(f"revision: {args.revision}")
    print(f"backend: {args.backend} ({args.endpoint})")
    for index, preset in enumerate(presets, 1):
        spec = PAPER_MODELS[preset]
        destination = models_root / spec.directory
        print(
            f"[{index}/{len(presets)}] {preset}: {spec.repo_id} -> {destination}\n"
            f"  {spec.note}",
            flush=True,
        )
        if args.dry_run:
            continue
        if args.backend == "aria2":
            download_with_aria2(
                spec=spec,
                destination=destination,
                endpoint=args.endpoint,
                revision=args.revision,
                connections=args.connections,
                concurrent_files=args.concurrent_files,
            )
        else:
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec.repo_id,
                revision=args.revision,
                local_dir=destination,
                allow_patterns=list(spec.allow_patterns),
                max_workers=args.max_workers,
            )
        validate_download(destination, spec)

    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
        return

    manifest = {
        "models_root": str(models_root),
        "revision": args.revision,
        "backend": args.backend,
        "endpoint": args.endpoint,
        "models": [
            {
                "preset": preset,
                "repo_id": PAPER_MODELS[preset].repo_id,
                "path": str(models_root / PAPER_MODELS[preset].directory),
                "note": PAPER_MODELS[preset].note,
            }
            for preset in presets
        ],
    }
    manifest_path = models_root / "paper_models_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded and validated {len(presets)} model snapshots.")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
