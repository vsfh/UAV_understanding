from __future__ import annotations

import importlib.util
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


def test_memory_profiles_adapt_batch_to_each_gpu() -> None:
    module = load_module()
    large = module.memory_profile(0, "large", 24_564)
    small = module.memory_profile(1, "small", 12_288)

    assert (large.full_batch, large.full_accumulation) == (4, 4)
    assert (small.full_batch, small.full_accumulation) == (1, 16)
    assert large.minimum_free_mib < large.total_mib
    assert small.minimum_free_mib < small.total_mib


def test_jobs_cover_zero_shot_and_every_protocol_seed_finetune() -> None:
    module = load_module()
    jobs = module.jobs_for(["forward_temporal", "unseen_site"], [42, 43])

    assert len(jobs) == 6
    assert sum(job.kind == "zero" for job in jobs) == 2
    assert sum(job.kind == "finetune" for job in jobs) == 4


def test_finetune_shard_skips_duplicate_zero_shot(tmp_path: Path) -> None:
    module = load_module()
    gpu = module.memory_profile(2, "RTX", 16_384)
    job = module.Job("finetune.unseen_site.seed42", "finetune", "unseen_site", 42)
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
    assert command[command.index("--cuda-devices") + 1] == "2"
    assert command[command.index("--openclip-full-batch-size") + 1] == "2"
