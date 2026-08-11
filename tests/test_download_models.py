from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/download_models.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_download_plan_contains_all_models_and_geochat_dependency() -> None:
    module = load_module()

    assert module.selected_presets([]) == module.DEFAULT_PRESETS
    assert module.PAPER_MODELS["qwen3-vl"].repo_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert module.PAPER_MODELS["openclip"].repo_id == "openai/clip-vit-large-patch14"
    assert module.PAPER_MODELS["geochat"].repo_id == "MBZUAI/geochat-7B"
    assert module.PAPER_MODELS["uavit"].repo_id == "ZhanYang-nwpu/GeoChat-UAV"
    assert (
        module.PAPER_MODELS["geochat-vision-tower"].repo_id
        == "openai/clip-vit-large-patch14-336"
    )


def test_all_models_dry_run_does_not_create_destination(tmp_path: Path) -> None:
    destination = tmp_path / "models"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "all",
            "--models-root",
            str(destination),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert not destination.exists()
    assert "[1/5] qwen3-vl" in result.stdout
    assert "[5/5] geochat-vision-tower" in result.stdout
    assert "Dry run complete" in result.stdout


def test_mirror_file_list_filters_formats_and_rejects_unsafe_paths() -> None:
    module = load_module()
    spec = module.PAPER_MODELS["openclip"]
    payload = {
        "siblings": [
            {"rfilename": "config.json", "size": 10},
            {
                "rfilename": "model.safetensors",
                "size": 20,
                "lfs": {"sha256": "abc"},
            },
            {"rfilename": "pytorch_model.bin", "size": 30},
        ]
    }
    assert module.mirror_file_list(payload, spec) == [
        ("config.json", 10, None),
        ("model.safetensors", 20, "abc"),
    ]

    payload["siblings"].append({"rfilename": "../escape.json", "size": 1})
    try:
        module.mirror_file_list(payload, spec)
    except ValueError as exc:
        assert "Unsafe repository path" in str(exc)
    else:
        raise AssertionError("unsafe mirror path was accepted")


def test_aria2_sidecar_keeps_full_sized_file_pending(tmp_path: Path) -> None:
    module = load_module()
    files = [("model.safetensors", 4, None)]
    output = tmp_path / "model.safetensors"
    output.write_bytes(b"1234")

    assert module.pending_mirror_files(files, tmp_path) == []
    output.with_name(output.name + ".aria2").write_bytes(b"state")
    assert module.pending_mirror_files(files, tmp_path) == files


def test_downloaded_mirror_weight_checksum_is_verified(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "model.safetensors"
    output.write_bytes(b"weight")
    digest = module.sha256_file(output)

    module.validate_mirror_checksums(
        [("model.safetensors", output.stat().st_size, digest)], tmp_path
    )
    try:
        module.validate_mirror_checksums(
            [("model.safetensors", output.stat().st_size, "0" * 64)], tmp_path
        )
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("invalid mirror checksum was accepted")
