#!/usr/bin/env python3
"""Run unseen-site and forward-time experiments using the current environment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clear_uav.experiment_config import load_yaml_with_base
from clear_uav.run_progress import run_with_progress
from run_matched_ablations import fingerprint, load_epoch, check_epoch


def plan():
    steps = []
    for protocol in ("unseen_site", "forward_temporal"):
        entries = [
            ("ground_ms", "scripts/qwen_ground_ms.py", None, "checkpoint"),
            ("ground_cls", "scripts/qwen_ground_cls.py", None, "checkpoint"),
            ("ground_cls", "scripts/test_qwen_ground_cls.py", None, "test_results"),
            ("qwen", "scripts/paper_qwen_stage.py", "train", "calibration"),
            ("qwen", "scripts/paper_qwen_stage.py", "test", "results"),
            ("tiling", "scripts/paper_qwen_stage.py", "tiling-val", "calibration"),
            ("tiling", "scripts/paper_qwen_stage.py", "tiling-test", "results"),
        ]
        for model, script, action, artifact_key in entries:
            cfg_path = f"configs/yaml/paper_shift_{model}_{protocol}.yaml"
            config = load_yaml_with_base(ROOT / cfg_path)
            if config["data"]["protocols"] != [protocol]:
                raise ValueError(f"Wrong protocol: {cfg_path}")
            cmd = [sys.executable, script, "--config", cfg_path]
            if action:
                cmd += ["--action", action]
            output = Path(config["output"][artifact_key].format(protocol=protocol, seed=43))
            if not output.is_absolute():
                output = ROOT / output
            name = f"{protocol}_{model}_{artifact_key}"
            marker = ROOT / f"reports/paper_shift_runs/{name}.json"
            steps.append((cmd, config, output, marker))
    return steps


def completed(config, artifact, marker):
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt.get("config_sha256") != fingerprint(config) or not artifact.is_file():
            raise ValueError(f"Changed config or missing output for {marker}")
        return True
    if artifact.exists():
        raise ValueError(f"Existing output has no completion receipt: {artifact}. "
                         "Preserve it and inspect before rerunning.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    steps = plan()
    total = sum(not completed(config, artifact, marker) for _, config, artifact, marker in steps)
    print(f"[queue] {total} remaining cross-domain tasks", flush=True)
    task_index = 0
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + child_env.get("PYTHONPATH", "")
    for cmd, config, artifact, marker in steps:
        if completed(config, artifact, marker):
            continue
        # Training must start in a fresh run directory. Never silently restart
        # an interrupted run that lacks optimizer/scheduler resume state.
        is_train = cmd[1] in ("scripts/qwen_ground_ms.py", "scripts/qwen_ground_cls.py")
        is_qwen_train = "--action" in cmd and cmd[-1] == "train"
        if is_train or is_qwen_train:
            protocol = config["data"]["protocols"][0]
            run_dir = ROOT / config["output"]["root"].format(protocol=protocol, seed=43)
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(f"Nonempty unfinished training directory: {run_dir}")
        task_index += 1
        run_with_progress(cmd, cwd=ROOT, env=child_env, label=marker.stem,
                          index=task_index, total=total)
        if not artifact.is_file():
            raise ValueError(f"Expected output missing: {artifact}")
        if artifact.suffix == ".pt":
            check_epoch(load_epoch(artifact))
        else:
            json.loads(artifact.read_text())  # reject incomplete JSON
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"config_sha256": fingerprint(config),
                       "resolved_config": config, "artifact": str(artifact)},
                      stream, indent=2)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, BlockingIOError) as error:
        raise SystemExit(f"STOP: {error}") from error
