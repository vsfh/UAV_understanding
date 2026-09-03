from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_yaml_with_base(path: str | Path) -> dict:
    override = load_yaml(path)
    base = override.pop("base_config", None)
    if base is None:
        return override
    config = load_yaml_with_base(project_path(base))

    def merge(target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(config, override)
    return config


def project_path(value: str, **values: object) -> Path:
    path = Path(value.format(**values)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def experiment_runs(config: dict):
    for protocol in config["data"]["protocols"]:
        for seed in config["train"]["seeds"]:
            yield protocol, seed
