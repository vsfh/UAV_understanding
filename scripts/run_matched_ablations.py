#!/usr/bin/env python3
"""Run the full control and six matched ablations using the current environment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

CHOICES = ("full", "no_heatmap", "single_scale", "fixed_classifier",
           "no_roi", "no_curriculum", "no_global")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clear_uav.run_progress import phase, run_with_progress


def stage_names(names):
    stages = ["full_ms", "full_cls"]
    for name in CHOICES[1:]:
        if name in names:
            stages.extend([name + "_ms", name + "_cls"] if name in
                          ("no_heatmap", "single_scale") else [name])
    return stages


def fingerprint(config):
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def check_budget(config):
    expected = {"epochs": 12, "learning_rate": 1e-4,
                "vision_learning_rate": 1e-6, "seeds": [43]}
    for key, value in expected.items():
        if config["train"].get(key) != value:
            raise ValueError(f"Unmatched budget: train.{key} must be {value}")
    if config["data"]["protocols"] != ["session_disjoint"]:
        raise ValueError("Matched ablations must use only session_disjoint")


def check_epoch(epoch, expected=12):
    if epoch != expected:
        raise ValueError(f"Checkpoint epoch {epoch}, expected {expected}. "
                         "Do not skip/evaluate this partial or mismatched run. "
                         "Preserve it and explicitly resolve restart/resume first.")


def load_epoch(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["epoch"]


def check_result(payload, config, reference_ids):
    if payload.get("experiment") != config["experiment"]:
        raise ValueError("Wrong result experiment")
    if payload.get("protocol") != "session_disjoint" or payload.get("seed") != 43:
        raise ValueError("Wrong result protocol/seed")
    check_epoch(payload.get("checkpoint_epoch"))
    rows = payload.get("rows", [])
    ids = [r["record_uid"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != reference_ids:
        raise ValueError("Result UIDs are duplicated or differ from frozen test")
    metrics = payload.get("metrics", {}).get("table4", {})
    for key in ("ap50", "c_f1", "g_map50", "n_fpr", "p_recall"):
        if not isinstance(metrics.get(key), (int, float)):
            raise ValueError(f"Missing unified Table IV metric: {key}")


def state(config, checkpoint, result, receipt, reference_ids, epoch_loader=load_epoch):
    if checkpoint.exists():
        check_epoch(epoch_loader(checkpoint))
        if not receipt.exists():
            raise ValueError(f"Missing resolved-config receipt for {checkpoint}")
        saved = json.loads(receipt.read_text())
        if saved.get("config_sha256") != fingerprint(config):
            raise ValueError(f"Resolved config differs from checkpoint receipt: {checkpoint}")
        if result is not None and result.exists():
            check_result(json.loads(result.read_text()), config, reference_ids)
            return "done"
        return "trained"
    if result is not None and result.exists():
        raise ValueError(f"Result exists without its checkpoint: {result}")
    if checkpoint.parent.exists() and any(checkpoint.parent.iterdir()):
        entries = {p.name for p in checkpoint.parent.iterdir()}
        # Ctrl+C during model loading leaves only our config receipt. No model
        # state exists to overwrite; retry preparation with the same config.
        if entries == {receipt.name} and receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved.get("config_sha256") == fingerprint(config):
                return "new"
        raise ValueError(f"Nonempty run directory without checkpoint: {checkpoint.parent}")
    return "new"


def paths(config):
    values = dict(protocol="session_disjoint", seed=43)
    def resolve(value):
        path = Path(value.format(**values))
        return path if path.is_absolute() else ROOT / path
    checkpoint = resolve(config["output"]["checkpoint"])
    return (checkpoint, resolve(config["output"]["test_results"]),
            checkpoint.parent / "matched_run_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=CHOICES, default=list(CHOICES))
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from clear_uav.experiment_config import load_yaml_with_base
    reference_path = ROOT / "results/table4/qwen_ground_cls_t4/session_disjoint/seed43_test.json"
    with phase("Prepare experiment queue: read frozen test record IDs"):
        reference = json.loads(reference_path.read_text())
    reference_ids = {row["record_uid"] for row in reference["rows"]}
    if len(reference_ids) != len(reference["rows"]):
        raise ValueError("Duplicate IDs in reference result")
    stages = []
    for name in stage_names(args.only):
        relative = Path(f"configs/yaml/table4_matched_{name}.yaml")
        config = load_yaml_with_base(ROOT / relative)
        check_budget(config)
        checkpoint, result, receipt = paths(config)
        is_ms = name.endswith("_ms")
        if not is_ms:
            init_name = name[:-4] + "_ms" if name.endswith("_cls") else "full_ms"
            expected_init = ROOT / f"outputs/table4_matched_ablations/{init_name}/session_disjoint/seed43/last.pt"
            actual_init = Path(config["initialization"]["grounding_checkpoint"].format(
                protocol="session_disjoint", seed=43))
            if not actual_init.is_absolute():
                actual_init = ROOT / actual_init
            if actual_init.resolve() != expected_init.resolve():
                raise ValueError(f"Wrong MS initialization for {name}")
        status = state(config, checkpoint, None if is_ms else result, receipt, reference_ids)
        train_script = "qwen_ground_ms.py" if is_ms else "qwen_ground_cls.py"
        commands = []
        if status == "new":
            commands.append([sys.executable, f"scripts/{train_script}", "--config", str(relative)])
        if not is_ms and status != "done":
            commands.append([sys.executable, "scripts/test_qwen_ground_cls.py", "--config", str(relative)])
        stages.append((name, config, checkpoint, result, receipt, status, commands))
        if status == "new" and receipt.exists():
            print(f"[retry] {name}: previous attempt stopped before any checkpoint; retrying setup.", flush=True)
    total = sum(len(stage[-1]) for stage in stages)
    print(f"[queue] {len(stages)} stages; {total} remaining train/evaluation tasks", flush=True)
    task_index = 0
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + child_env.get("PYTHONPATH", "")
    for name, config, checkpoint, result, receipt, status, commands in stages:
        is_ms = name.endswith("_ms")
        current = state(config, checkpoint, None if is_ms else result, receipt, reference_ids)
        if current != status:
            raise ValueError(f"Run state changed during preflight: {name}")
        if not is_ms:
            initial = Path(config["initialization"]["grounding_checkpoint"].format(
                protocol="session_disjoint", seed=43))
            check_epoch(load_epoch(ROOT / initial))
        if status == "new":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not receipt.exists():
                with receipt.open("x", encoding="utf-8") as out:
                    json.dump({"config_sha256": fingerprint(config), "resolved_config": config},
                              out, indent=2)
        for command in commands:
            task_index += 1
            action = "evaluate" if Path(command[1]).name.startswith("test_") else "train"
            run_with_progress(command, cwd=ROOT, env=child_env,
                              label=f"{name} / {action}", index=task_index, total=total)
            check_epoch(load_epoch(checkpoint))
        if not is_ms:
            check_result(json.loads(result.read_text()), config, reference_ids)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, BlockingIOError) as error:
        raise SystemExit(f"STOP: {error}") from error
