#!/usr/bin/env python3
"""Run unseen-site and forward-time experiments using the current environment."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clear_uav.run_progress import run_with_progress
from clear_uav.experiment_guard import (artifact_identity, assert_unlocked,
                                       stage_lock, verify_artifact, write_json_atomic)
from run_matched_ablations import fingerprint, load_epoch, check_epoch

PROTOCOLS = ("unseen_site", "forward_temporal")
GROUPS = {"ground": ("ground_ms", "ground_cls"), "ground_ms": ("ground_ms",),
          "ground_cls": ("ground_cls",), "qwen": ("qwen",), "tiling": ("tiling",)}


def load_yaml_with_base(path):
    from clear_uav.experiment_config import load_yaml_with_base as load
    return load(path)


def plan(protocols=PROTOCOLS, only=("ground", "qwen", "tiling")):
    steps = []
    models = {model for group in only for model in GROUPS[group]}
    for protocol in protocols:
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
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
            if model not in models:
                continue
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


def lock_path(marker):
    return ROOT / "reports/experiment_locks" / ("shift_" + marker.stem + ".lock")


def dependency_names(marker):
    name = marker.stem
    protocol = next(p for p in PROTOCOLS if name.startswith(p + "_"))
    suffix = name[len(protocol) + 1:]
    dependencies = {
        "ground_cls_checkpoint": ("ground_ms_checkpoint",),
        "ground_cls_test_results": ("ground_cls_checkpoint",),
        "qwen_results": ("qwen_calibration",),
        "tiling_calibration": ("qwen_calibration",),
        "tiling_results": ("qwen_results", "tiling_calibration"),
    }
    return [protocol + "_" + d for d in dependencies.get(suffix, ())]


def require_dependencies(marker, catalog, scheduled=()):
    for name in dependency_names(marker):
        if name in scheduled:
            continue
        _, config, artifact, receipt = catalog[name]
        assert_unlocked(lock_path(receipt), name)
        if not completed(config, artifact, receipt):
            raise ValueError(f"{marker.stem} requires completed {name}; "
                             "finish that experiment first. No dependency is started automatically.")


def validate_artifact(config, artifact, marker):
    if artifact.suffix == ".pt":
        check_epoch(load_epoch(artifact), config["train"]["epochs"])
        return
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an output object: {artifact}")
    if marker.stem.endswith(("_results", "_test_results")):
        if (payload.get("experiment") != config["experiment"]
                or payload.get("protocol") != config["data"]["protocols"][0]
                or payload.get("seed") != 43):
            raise ValueError(f"Wrong result experiment/protocol/seed: {artifact}")
        if marker.stem.endswith("_ground_cls_test_results"):
            check_epoch(payload.get("checkpoint_epoch"), config["train"]["epochs"])
        rows = payload.get("rows", [])
        ids = [row.get("record_uid") for row in rows]
        if not ids or any(not uid for uid in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Empty, missing or duplicate result IDs: {artifact}")
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid result metrics: {artifact}")
        threshold = metrics.get("table4", metrics).get("threshold")
    else:
        threshold = payload.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)):
        raise ValueError(f"Missing or nonfinite calibration threshold: {artifact}")


def supporting_artifacts(config, marker):
    """Bind Qwen/tiling receipts to their actual adapter and full-frame cache."""
    if "adapter" not in config["output"]:
        return {}
    values = {"protocol": config["data"]["protocols"][0], "seed": 43}
    def resolve(value):
        path = Path(value.format(**values))
        return path if path.is_absolute() else ROOT / path
    adapter = resolve(config["output"]["adapter"])
    if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"Incomplete shared Qwen adapter: {adapter}")
    files = {"adapter/" + path.relative_to(adapter).as_posix(): path
             for path in sorted(adapter.rglob("*")) if path.is_file()}
    files["training_data"] = adapter.parent / "training_data.json"
    split = "validation" if marker.stem.endswith("_calibration") else "test"
    files["full_frame_predictions"] = resolve(config["output"][f"{split}_predictions"])
    return {name: artifact_identity(path, ROOT) for name, path in files.items()}


def completed(config, artifact, marker):
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt.get("config_sha256") != fingerprint(config) or not artifact.is_file():
            raise ValueError(f"Changed config or missing output for {marker}")
        if (receipt.get("stage") != marker.stem or receipt.get("seed") != 43
                or receipt.get("protocol") != config["data"]["protocols"][0]
                or "artifact_identity" not in receipt):
            raise ValueError(f"Legacy/unbound completion receipt: {marker}. Preserve it and "
                             "verify the existing output before adopting it; do not rerun training.")
        verify_artifact(artifact, receipt["artifact_identity"], ROOT)
        if supporting_artifacts(config, marker) != receipt.get("supporting_artifacts", {}):
            raise ValueError(f"Adapter/cache differs from completion receipt: {marker}")
        for identity in receipt.get("dependencies", {}).values():
            dependency = Path(identity["path"])
            if not dependency.is_absolute():
                dependency = ROOT / dependency
            verify_artifact(dependency, identity, ROOT)
        validate_artifact(config, artifact, marker)
        return True
    if artifact.exists():
        protocol = config["data"]["protocols"][0]
        hint = (f" Inspect/preserve it, then explicitly verify existing Qwen outputs with: "
                f"python scripts/run_paper_shifts.py --protocol {protocol} --only qwen --recover-existing"
                if marker.stem.endswith(("_qwen_calibration", "_qwen_results")) else
                " Preserve it and inspect before rerunning.")
        raise ValueError(f"Existing output has no completion receipt: {artifact}." + hint)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--only", nargs="+", choices=tuple(GROUPS), default=["ground", "qwen", "tiling"],
                        help="Run only the selected models; GPU selection is inherited from your environment.")
    parser.add_argument("--recover-existing", action="store_true",
                        help="With --only qwen, validate and register completed existing outputs; never train or infer.")
    args = parser.parse_args()
    steps = plan(tuple(dict.fromkeys(args.protocol)), args.only)
    catalog = {step[3].stem: step for step in plan()}
    if args.recover_existing:
        if set(args.only) != {"qwen"}:
            raise ValueError("--recover-existing requires --only qwen; it never starts missing experiments")
        from recover_paper_qwen import recover_existing_qwen
        for protocol in dict.fromkeys(args.protocol):
            recover_existing_qwen(protocol, catalog, sys.modules[__name__])
        return
    scheduled = {step[3].stem for step in steps}
    for _, config, artifact, marker in steps:
        assert_unlocked(lock_path(marker), marker.stem)
        if not completed(config, artifact, marker):
            require_dependencies(marker, catalog, scheduled)
    total = sum(not completed(config, artifact, marker) for _, config, artifact, marker in steps)
    print(f"[queue] {total} remaining cross-domain tasks", flush=True)
    task_index = 0
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + child_env.get("PYTHONPATH", "")
    for cmd, config, artifact, marker in steps:
        with stage_lock(lock_path(marker), marker.stem):
            if completed(config, artifact, marker):
                continue
            require_dependencies(marker, catalog)
            dependencies = {name: artifact_identity(catalog[name][2], ROOT)
                            for name in dependency_names(marker)}
            # Never silently restart an interrupted run without optimizer state.
            is_train = cmd[1] in ("scripts/qwen_ground_ms.py", "scripts/qwen_ground_cls.py")
            is_qwen_train = "--action" in cmd and cmd[-1] == "train"
            if is_train or is_qwen_train:
                protocol = config["data"]["protocols"][0]
                run_dir = ROOT / config["output"]["root"].format(protocol=protocol, seed=43)
                if run_dir.exists() and any(run_dir.iterdir()):
                    hint = (" If Qwen training finished but calibration stopped, preserve the adapter "
                            "and training_data.json and verify training completion before a calibration-only "
                            "recovery. This runner will not infer completion from adapter_config.json alone."
                            if is_qwen_train else " Preserve the checkpoint; partial runs need explicit resume/restart review.")
                    raise ValueError(f"Nonempty unfinished training directory: {run_dir}." + hint)
            task_index += 1
            run_with_progress(cmd, cwd=ROOT, env=child_env, label=marker.stem,
                              index=task_index, total=total)
            if not artifact.is_file():
                raise ValueError(f"Expected output missing: {artifact}")
            validate_artifact(config, artifact, marker)
            # A dependency changed during this stage is not a valid completed run.
            for name, identity in dependencies.items():
                verify_artifact(catalog[name][2], identity, ROOT)
            marker.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(marker, {"config_sha256": fingerprint(config),
                                      "resolved_config": config, "stage": marker.stem,
                                      "protocol": config["data"]["protocols"][0], "seed": 43,
                                      "artifact_identity": artifact_identity(artifact, ROOT),
                                      "dependencies": dependencies,
                                      "supporting_artifacts": supporting_artifacts(config, marker)})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, BlockingIOError) as error:
        raise SystemExit(f"STOP: {error}") from error
