from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/run_openclip_multi_gpu.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_openclip_multi_gpu", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_memory_profiles_use_same_20gb_safe_batch_on_every_gpu() -> None:
    module = load_module()
    first = module.memory_profile(0)
    second = module.memory_profile(1)

    assert first.configured_mib == second.configured_mib == 20_000
    assert (first.full_batch, first.full_accumulation) == (2, 8)
    assert (second.full_batch, second.full_accumulation) == (2, 8)
    assert first.linear_batch == second.linear_batch == 256
    assert first.linear_feature_batch == second.linear_feature_batch == 32
    assert first.zero_batch == second.zero_batch == 16
    assert first.minimum_free_mib == second.minimum_free_mib == 8_000
    assert first.full_batch * first.full_accumulation == 16
    assert second.full_batch * second.full_accumulation == 16


def test_gpu_ids_are_preserved_verbatim_without_inventory_lookup() -> None:
    module = load_module()
    profiles = module.detect_gpus([0, 2, 1])

    assert [profile.index for profile in profiles] == [0, 2, 1]
    assert all(profile.configured_mib == 20_000 for profile in profiles)


def test_oom_reduction_preserves_effective_full_batch() -> None:
    module = load_module()
    profile = module.memory_profile(0)
    reduced = module.reduced_batch_profile(profile, "full_finetune")

    assert reduced is not None
    assert reduced.full_batch == profile.full_batch // 2
    assert reduced.full_batch * reduced.full_accumulation == 16
    assert reduced.minimum_free_mib < profile.minimum_free_mib


def test_jobs_cover_zero_shot_and_every_protocol_seed_finetune() -> None:
    module = load_module()
    jobs = module.jobs_for(["forward_temporal", "unseen_site"], [42, 43])

    assert len(jobs) == 10
    assert sum(job.kind == "zero" for job in jobs) == 2
    assert sum(job.kind == "linear_probe" for job in jobs) == 4
    assert sum(job.kind == "full_finetune" for job in jobs) == 4


def test_finetune_shard_skips_duplicate_zero_shot(tmp_path: Path) -> None:
    module = load_module()
    gpu = module.memory_profile(2)
    job = module.Job(
        "full_finetune.unseen_site.seed42", "full_finetune", "unseen_site", 42
    )
    command = module.command_for(
        job,
        gpu,
        args=SimpleNamespace(num_workers=2),
        data_root=tmp_path / "data",
        models_root=tmp_path / "models",
        output_root=tmp_path / "outputs",
        results_root=tmp_path / "results",
        labels_file=tmp_path / "labels.txt",
    )

    assert "--skip-zero-shot" in command
    assert "--openclip-finetuning" in command
    assert command[command.index("--openclip-finetuning-modes") + 1] == "full_finetune"
    assert command[command.index("--cuda-devices") + 1] == "2"
    assert command[command.index("--openclip-full-batch-size") + 1] == "2"
    assert command[command.index("--openclip-full-epochs") + 1] == "20"


def test_child_tqdm_carriage_returns_are_exposed_as_live_status() -> None:
    module = load_module()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('epoch 1 10%\\repoch 1 20%\\nfinished\\n')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = io.BytesIO()
    statuses = []

    return_code = module.stream_process(process, log, on_status=statuses.append)

    assert return_code == 0
    assert statuses == ["epoch 1 10%", "epoch 1 20%", "finished"]
    assert b"epoch 1 10%\r" in log.getvalue()
