from __future__ import annotations

from pathlib import Path

from clear_uav.experiment_config import experiment_runs, load_yaml


ROOT = Path(__file__).parents[1]
YAML_ROOT = ROOT / "configs/yaml"


def test_main_training_configs_cover_three_protocols_with_seed_43() -> None:
    expected_epochs = {
        "openclip_linear_probe": 20,
        "openclip_full_finetune": 20,
        "qwen_lora": 5,
    }
    for name, epochs in expected_epochs.items():
        config = load_yaml(YAML_ROOT / f"{name}.yaml")
        runs = list(experiment_runs(config))
        assert len(runs) == 3
        assert {protocol for protocol, _ in runs} == {
            "forward_temporal",
            "session_disjoint",
            "unseen_site",
        }
        assert {seed for _, seed in runs} == {43}
        assert config["train"]["epochs"] == epochs


def test_geochat_config_has_seed_43_and_both_prompts() -> None:
    config = load_yaml(YAML_ROOT / "geochat_test.yaml")
    prompts = {prompt["name"]: prompt["template"] for prompt in config["prompts"]}

    assert config["runtime"]["seed"] == 43
    assert config["test"]["view"] == "context"
    assert set(prompts) == {"direct", "definition"}
    assert "{label_list}" in prompts["direct"]
    assert "{definition_list}" in prompts["definition"]


def test_openclip_modes_use_the_expected_memory_profiles() -> None:
    linear = load_yaml(YAML_ROOT / "openclip_linear_probe.yaml")["train"]
    full = load_yaml(YAML_ROOT / "openclip_full_finetune.yaml")["train"]

    assert linear["batch_size"] * linear["gradient_accumulation"] == 256
    assert full["batch_size"] * full["gradient_accumulation"] == 16
    assert full["gradient_checkpointing"] is True


def test_every_entrypoint_reads_one_yaml_without_subprocess_or_command_extend() -> None:
    names = (
        "openclip_linear_probe.py",
        "openclip_finetune.py",
        "qwen_lora.py",
        "test_openclip.py",
        "test_qwen.py",
        "test_geochat.py",
    )
    for name in names:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "--config" in source
        assert "subprocess" not in source
        assert "command.extend" not in source


def test_tensorboard_is_enabled_for_both_training_stacks() -> None:
    openclip = (ROOT / "src/clear_uav/openclip_training.py").read_text(encoding="utf-8")
    qwen = (ROOT / "scripts/qwen_lora.py").read_text(encoding="utf-8")

    assert "SummaryWriter" in openclip
    assert 'report_to=["tensorboard"]' in qwen
