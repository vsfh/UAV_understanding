"""Evaluate spatial perception using the unchanged baseline generation and metrics.

Run validation first to select the presence threshold; test never calibrates on
test labels. The saved training configuration defines the model architecture.
"""
import argparse
import json

from train_perception_spatial import DEFAULT_CONFIG, build_model, output_path, read_config, subset
from test_perception_qwen import predict
from clear_uav.table4 import definitions_from_config, read_discovery_samples, save_results, select_threshold, table4_metrics
from perception_spatial_head import (checkpoint_fingerprint, evaluation_configuration,
                                    evaluation_signature, validate_calibration, final_output_fpr)


def evaluate(config, split):
    output = output_path(config)
    saved = read_config(output / "config.yaml")
    config = evaluation_configuration(saved, config)
    checkpoint = output / 'best'
    fingerprint = checkpoint_fingerprint(checkpoint)
    calibration_path = output / 'calibration.json'
    if split == "test":
        if not calibration_path.is_file():
            raise FileNotFoundError("Run --split val or --split all before test: no validation calibration found")
        validate_calibration(json.loads(calibration_path.read_text(encoding='utf-8')), config, checkpoint, fingerprint)
    model, processor = build_model(config, checkpoint=checkpoint, is_trainable=False)
    labels, _ = definitions_from_config(config)
    for part in (["val", "test"] if split == "all" else [split]):
        if part == 'test':
            calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
            validate_calibration(calibration, config, checkpoint, fingerprint)
        samples = subset(read_discovery_samples(config, config["protocol"], part), config.get(f"max_{part}_samples"))
        if not samples:
            raise ValueError(f"{part} split is empty")
        predictions = predict(model, processor, samples, config, labels)
        if part == "val":
            threshold, selection = select_threshold(samples, predictions, config["validation"])
            calibration_path.write_text(json.dumps({
                "threshold": threshold, "selection": selection, "policy": "max_recall_at_fpr",
                "max_n_fpr": config["validation"]["max_n_fpr"], "num_samples": len(samples),
                "sample_limit": config.get("max_val_samples"),
                "limited_run": config.get('max_val_samples') is not None or bool(config['data'].get('max_samples')),
                "split": "val", "protocol": config['protocol'], "seed": config['seed'],
                "checkpoint": str(checkpoint), "checkpoint_sha256": fingerprint,
                "evaluation_signature": evaluation_signature(config),
            }, indent=2), encoding="utf-8")
        else:
            threshold = calibration['threshold']
        metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
        metrics['final_output_n_fpr'] = final_output_fpr(samples, predictions, threshold, labels)
        config["train"]["seeds"] = [config["seed"]]
        save_results(output / f"{part}_results.json", config, config["protocol"], samples, predictions,
                     metrics, seed=config["seed"])
        print(json.dumps({"extension": config.get("experiment", "spatial_interaction"), "split": part, **metrics}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--output")
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ("output", "max_val_samples", "max_test_samples"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    evaluate(config, args.split)


if __name__ == "__main__":
    main()
