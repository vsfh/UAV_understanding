"""Evaluate What / Where / Verify with validation-only, mode-specific calibration."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


DEFAULT_CONFIG = "configs/yaml/perception_agents.yaml"
MODES = ("full", "no_verify", "verify_only")
CALIBRATION_VERSION = 1


def evaluation_configuration(saved, requested):
    """The checkpoint's training config owns all prompts and architecture settings."""
    config = copy.deepcopy(saved)
    config["training_sample_limits"] = {
        name: config.pop(name, None)
        for name in ("max_train_samples", "max_val_samples", "max_test_samples")
    }
    for name in ("max_val_samples", "max_test_samples"):
        if requested.get(name) is not None:
            config[name] = requested[name]
    for name in ("output", "device"):
        if requested.get(name) is not None:
            config[name] = requested[name]
    config.setdefault("test", {})["batch_size"] = requested.get("test", {}).get(
        "batch_size", config.get("test", {}).get("batch_size", 1)
    )
    return config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_signature(config, mode, labels, definitions, source_hashes=None):
    """Bind calibration to inference semantics, including external label contents."""
    values = {
        name: config.get(name)
        for name in (
            "protocol", "seed", "model", "spatial", "input", "prompt", "agents",
            "data", "validation", "test",
        )
    }
    # Only batch size can change between val and test without changing semantics.
    values["test"] = {
        name: value for name, value in (values["test"] or {}).items()
        if name != "batch_size"
    }
    values.update(mode=mode, labels=list(labels), definitions=definitions,
                  source_hashes=source_hashes or {})
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_calibration(calibration, config, checkpoint, fingerprint, signature, mode):
    if calibration.get("schema_version") != CALIBRATION_VERSION or calibration.get("split") != "val":
        raise ValueError("Run validation to create a current calibration file")
    if calibration.get("mode") != mode:
        raise ValueError("Calibration belongs to another agent mode; run validation for this mode")
    if (calibration.get("checkpoint") != str(Path(checkpoint).resolve())
            or calibration.get("checkpoint_sha256") != fingerprint):
        raise ValueError("Checkpoint content/path changed; rerun validation")
    if calibration.get("inference_signature") != signature:
        raise ValueError("Inference config, prompts, labels, definitions or source changed; rerun validation")
    if calibration.get("limited_run") and config.get("max_test_samples") is None and not config["data"].get("max_samples"):
        raise ValueError("Full test requires full validation calibration; limited validation is only for smoke runs")
    threshold = calibration.get("threshold")
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= math.nextafter(1.0, math.inf)):
        raise ValueError("Invalid calibrated threshold")
    return float(threshold)


def agent_metrics(predictions):
    count = len(predictions)
    calls = {role: [] for role in ("what", "where", "verify")}
    revisions, abstentions, verification = [], [], []
    for prediction in predictions:
        trace = prediction.get("trace", {})
        trace = trace if isinstance(trace, dict) else {}
        explicit = prediction.get("agent_calls")
        for role in calls:
            calls[role].append(
                int(explicit.get(role, 0)) if isinstance(explicit, dict)
                else sum(call.get("role") == role for call in trace.get("calls", []))
            )
        revisions.append(int(prediction.get("revisions", trace.get("revisions", 0))))
        abstentions.append(bool(prediction.get("abstained", trace.get("abstained", False))))
        verification.append(bool(prediction.get("verified", False)))
    return {
        "agent_mean_calls": {role: sum(values) / count for role, values in calls.items()},
        "agent_call_rates": {role: sum(value > 0 for value in values) / count for role, values in calls.items()},
        "revision_rate": sum(value > 0 for value in revisions) / count,
        "mean_revisions": sum(revisions) / count,
        "abstention_rate": sum(abstentions) / count,
        "verified_rate": sum(verification) / count,
    }


def write_traces(path, samples, predictions):
    with Path(path).open("w", encoding="utf-8") as stream:
        for sample, prediction in zip(samples, predictions):
            row = {"record_uid": sample.record_uid}
            for name in ("trace", "agent_calls", "revisions", "abstained", "verified",
                         "what_presence_score", "verify_score", "num_calls"):
                if name in prediction:
                    row[name] = prediction[name]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(requested, split, mode):
    from perception_agents_runtime import AgentRuntime, build_model, output_path, read_config
    from perception_spatial_head import checkpoint_fingerprint, final_output_fpr
    from clear_uav.table4 import (
        definitions_from_config, read_discovery_samples, save_results,
        select_threshold, table4_metrics,
    )

    output = output_path(requested)
    config = evaluation_configuration(read_config(output / "config.yaml"), requested)
    checkpoint = output / "best"
    for name in ("agent_schema.json", "box_head.pt", "spatial_head.pt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Missing trained agent checkpoint file: {checkpoint / name}")
    fingerprint = checkpoint_fingerprint(checkpoint)
    labels, definitions = definitions_from_config(config)
    source_hashes = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("perception_agents_core.py", "perception_agents_runtime.py")
    }
    import train_perception_spatial, perception_spatial_head, train_perception_qwen
    from clear_uav import generation_constraints, table4
    for dependency in (train_perception_spatial, perception_spatial_head, train_perception_qwen,
                       generation_constraints, table4):
        source_hashes[dependency.__name__] = file_sha256(dependency.__file__)
    modes = MODES if mode == "all" else (mode,)
    signatures = {current: inference_signature(config, current, labels, definitions, source_hashes)
                  for current in modes}
    for name in ("max_val_samples", "max_test_samples"):
        if config.get(name) is not None and config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if split == "all" and config.get("max_val_samples") is not None and config.get("max_test_samples") is None and not config["data"].get("max_samples"):
        raise ValueError("Limit both splits for a smoke run, or use full validation before full test")
    # Check every requested mode before allocating the model for standalone test.
    if split == "test":
        for current in modes:
            path = output / "evaluation" / current / "calibration.json"
            validate_calibration(json.loads(path.read_text(encoding="utf-8")), config,
                                 checkpoint, fingerprint, signatures[current], current)

    model, processor = build_model(config, checkpoint=checkpoint, is_trainable=False)
    runtime = AgentRuntime(model, processor, config, labels)
    for current in modes:
        destination = output / "evaluation" / current
        destination.mkdir(parents=True, exist_ok=True)
        calibration_path = destination / "calibration.json"
        for part in (("val", "test") if split == "all" else (split,)):
            if part == "test":
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                threshold = validate_calibration(calibration, config, checkpoint, fingerprint,
                                                 signatures[current], current)
            samples = read_discovery_samples(config, config["protocol"], part)
            limit = config.get(f"max_{part}_samples")
            samples = samples[:limit] if limit is not None else samples
            if not samples:
                raise ValueError(f"Empty {part} split")
            predictions = runtime.predict(samples, mode=current)
            if len(predictions) != len(samples):
                raise ValueError("Agent runtime returned a different number of predictions")
            if part == "val":
                threshold, selection = select_threshold(samples, predictions, config["validation"])
                calibration = {
                    "schema_version": CALIBRATION_VERSION,
                    "split": "val", "mode": current, "threshold": threshold,
                    "selection": selection, "policy": "max_recall_at_fpr",
                    "max_n_fpr": config["validation"]["max_n_fpr"],
                    "num_samples": len(samples), "sample_limit": limit,
                    "limited_run": limit is not None or bool(config["data"].get("max_samples")),
                    "protocol": config["protocol"], "seed": config["seed"],
                    "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint,
                    "agent_schema_sha256": file_sha256(checkpoint / "agent_schema.json"),
                    "inference_signature": signatures[current],
                }
                calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
            metrics["final_output_fpr"] = final_output_fpr(samples, predictions, threshold, labels)
            metrics["final_output_n_fpr"] = metrics["final_output_fpr"]
            metrics.update(agent_metrics(predictions))
            from perception_agents_core import bbox_iou
            positive_rows = [(s, p) for s, p in zip(samples, predictions) if s.presence]
            initial_hits = []
            final_hits = []
            for sample, prediction in positive_rows:
                initial = prediction['trace']['initial_hypothesis']
                initial_hits.append(initial['category'] == sample.label and bbox_iou(initial['bbox_1000'], sample.bbox_1000) >= .5)
                final_hits.append(prediction['category'] == sample.label and bbox_iou(prediction['bbox_1000'], sample.bbox_1000) >= .5)
            metrics['agent_initial_joint50'] = sum(initial_hits)/len(positive_rows) if positive_rows else None
            metrics['agent_final_joint50'] = sum(final_hits)/len(positive_rows) if positive_rows else None
            metrics['agent_joint50_recovered'] = sum(not before and after for before,after in zip(initial_hits,final_hits))
            metrics['agent_joint50_lost'] = sum(before and not after for before,after in zip(initial_hits,final_hits))
            metrics["agent_mode"] = current
            config.setdefault("train", {})["seeds"] = [config["seed"]]
            # Traces are kept separately so old result readers remain usable.
            result_predictions = [{key: value for key, value in prediction.items() if key != "trace"}
                                  for prediction in predictions]
            save_results(destination / f"{part}_results.json", config, config["protocol"], samples,
                         result_predictions, metrics, seed=config["seed"])
            write_traces(destination / f"{part}_traces.jsonl", samples, predictions)
            print(json.dumps({"split": part, "mode": current, **metrics}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="full")
    parser.add_argument("--output")
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    from perception_agents_runtime import read_config
    config = read_config(args.config)
    for name in ("output", "max_val_samples", "max_test_samples", "device"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.batch_size is not None:
        config.setdefault("test", {})["batch_size"] = args.batch_size
    evaluate(config, args.split, args.mode)


if __name__ == "__main__":
    main()
