"""Evaluate matched Perception continuation with the existing Spatial protocol.

Generation, metrics and validation calibration are unchanged. The model and
checkpoint fingerprint are baseline-only; no spatial head is required or loaded.
"""
import argparse
import json

from train_perception_continue import DEFAULT_CONFIG, build_model, checkpoint_fingerprint
from train_perception_qwen import output_path, read_config
from test_perception_qwen import predict
from clear_uav.table4 import box_iou, definitions_from_config, read_discovery_samples, save_results, select_threshold, table4_metrics
from perception_spatial_head import evaluation_configuration, evaluation_signature, validate_calibration, final_output_fpr


def positive_localization_metrics(samples, predictions):
    """Paper metrics on all positive candidates, with missing boxes as failures."""
    positive = [(sample, prediction) for sample, prediction in zip(samples, predictions) if sample.presence]
    overlaps = [box_iou(sample.bbox_1000, prediction["bbox_1000"])
                if prediction.get("valid", True) else 0.0 for sample, prediction in positive]
    return {
        "mean_iou_positive": sum(overlaps) / len(positive),
        "localization_at_50": sum(value >= 0.5 for value in overlaps) / len(positive),
        "joint_at_50": sum(value >= 0.5 and sample.label == prediction["category"]
                           for value, (sample, prediction) in zip(overlaps, positive)) / len(positive),
    }


def evaluate(config, split):
    output = output_path(config)
    config = evaluation_configuration(read_config(output / "config.yaml"), config)
    if "spatial" in config:
        raise ValueError("This evaluator expects the baseline continuation configuration")
    checkpoint = output / "best"
    fingerprint = checkpoint_fingerprint(checkpoint)
    calibration_path = output / "calibration.json"
    if split == "test":
        if not calibration_path.is_file():
            raise FileNotFoundError("Run --split val or --split all before test")
        validate_calibration(json.loads(calibration_path.read_text(encoding="utf-8")), config, checkpoint, fingerprint)
    model, processor = build_model(config, checkpoint=checkpoint)
    labels, _ = definitions_from_config(config)
    for part in (["val", "test"] if split == "all" else [split]):
        if part == "test":
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            validate_calibration(calibration, config, checkpoint, fingerprint)
        samples = read_discovery_samples(config, config["protocol"], part)
        if not samples:
            raise ValueError(f"{part} split is empty")
        predictions = predict(model, processor, samples, config, labels)
        if part == "val":
            threshold, selection = select_threshold(samples, predictions, config["validation"])
            calibration_path.write_text(json.dumps({
                "threshold": threshold, "selection": selection, "policy": "max_recall_at_fpr",
                "max_n_fpr": config["validation"]["max_n_fpr"], "num_samples": len(samples),
                "sample_limit": None, "limited_run": bool(config["data"].get("max_samples")),
                "split": "val", "protocol": config["protocol"], "seed": config["seed"],
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint,
                "evaluation_signature": evaluation_signature(config),
            }, indent=2), encoding="utf-8")
        else:
            threshold = calibration["threshold"]
        metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
        metrics.update(positive_localization_metrics(samples, predictions))
        metrics["final_output_n_fpr"] = final_output_fpr(samples, predictions, threshold, labels)
        config["train"]["seeds"] = [config["seed"]]
        save_results(output / f"{part}_results.json", config, config["protocol"], samples, predictions,
                     metrics, seed=config["seed"])
        print(json.dumps({"experiment": "perception_continue", "split": part, **metrics}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output is not None:
        config["output"] = args.output
    evaluate(config, args.split)


if __name__ == "__main__":
    main()
