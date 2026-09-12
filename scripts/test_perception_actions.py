"""Evaluate learned greedy box actions; calibration uses validation records only."""
import argparse
import json
import time
from pathlib import Path

import torch
from perception_actions_core import ACTION_NAMES, rollout
from perception_actions_evaluation import (CALIBRATION_VERSION, calibration_preflight,
    validate_calibration, valid_final_event)
from train_perception_actions import DEFAULT_CONFIG, build_model, generated_features, output_path, read_config


@torch.inference_mode()
def predict(model, processor, samples, config, labels):
    model.eval()
    predictions = []
    device = torch.device(config["device"])
    from tqdm import tqdm
    for start in tqdm(range(0, len(samples), config["test"]["batch_size"]), desc="actions predict"):
        current = samples[start:start + config["test"]["batch_size"]]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        initial, features, categories, has_vis, scores, raw = generated_features(model, processor, current, config, labels)
        enabled = torch.tensor([category in labels for category in categories], device=device) & has_vis
        trajectory = rollout(model.action_policy, features, initial, config["actions"], greedy=True, enabled=enabled)
        final_boxes = (trajectory["boxes"] * 1000).cpu().tolist()
        initial_boxes = (initial * 1000).cpu().tolist()
        actions, masks = trajectory["actions"].cpu().tolist(), trajectory["mask"].cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - started) * 1000 / len(current)
        for i, category in enumerate(categories):
            positive = bool(enabled[i])
            sequence = [ACTION_NAMES[a] for a, mask in zip(actions[i], masks[i]) if mask]
            predictions.append({"category": category if positive else None,
                "bbox_1000": final_boxes[i] if positive else None,
                "initial_bbox_1000": initial_boxes[i] if positive else None,
                "presence_score": scores[i], "valid": bool(positive or category == "no_event"),
                "latency_ms": latency, "raw_output": raw[i], "num_calls": 2,
                "inference_batch_size": len(current), "action_sequence": sequence,
                "refinement_steps": sum(a != "stop" for a in sequence),
                "timing_scope": "preprocess_generate_replay_and_greedy_actions_amortized"})
    return predictions


def evaluate(config, split="all", stage="grpo", checkpoint=None, max_val_samples=None, max_test_samples=None):
    import train_perception_qwen
    from clear_uav.table4 import (definitions_from_config, read_discovery_samples,
        save_results, select_threshold, table4_metrics)
    output = output_path(config, stage)
    checkpoint = Path(checkpoint) if checkpoint else output / "best"
    if not (checkpoint / "action_policy.pt").is_file():
        raise FileNotFoundError(f"Trained action checkpoint missing: {checkpoint}")
    saved = read_config(checkpoint / "config.yaml")
    saved["device"], saved["test"] = config["device"], config["test"]
    saved["output"] = config["output"]
    config = saved
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = output / "calibration.json"
    # Validate calibration and split limits before loading the 8B model.
    fingerprint = calibration_preflight(split, calibration_path, checkpoint, max_val_samples, max_test_samples)
    model, processor = build_model(config, checkpoint)
    labels, _ = definitions_from_config(config)
    for part in (["val", "test"] if split == "all" else [split]):
        if part == "test":
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            threshold = validate_calibration(calibration, checkpoint, fingerprint, max_test_samples)
        samples = read_discovery_samples(config, config["protocol"], part)
        limit = max_val_samples if part == "val" else max_test_samples
        if limit:
            samples = samples[:limit]
        if not samples:
            raise ValueError(f"Empty {part} split")
        predictions = predict(model, processor, samples, config, labels)
        if part == "val":
            threshold, selection = select_threshold(samples, predictions, config["validation"])
            calibration_path.write_text(json.dumps({"schema_version": CALIBRATION_VERSION,
                "checkpoint_fingerprint": fingerprint, "threshold": threshold, "selection": selection,
                "policy": "max_recall_at_fpr", "max_n_fpr": config["validation"]["max_n_fpr"],
                "checkpoint": str(checkpoint.resolve()), "sample_count": len(samples), "limited_run": limit is not None}, indent=2), encoding="utf-8")
        metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
        negatives = [(sample, pred) for sample, pred in zip(samples, predictions) if not sample.presence]
        metrics["final_output_n_fpr"] = (sum(valid_final_event(pred, threshold, labels)
            for _, pred in negatives) / len(negatives)) if negatives else None
        metrics["mean_refinement_steps"] = sum(p["refinement_steps"] for p in predictions) / len(predictions)
        config["train"]["seeds"] = [config["seed"]]
        save_results(output / f"{part}_results.json", config, config["protocol"], samples, predictions, metrics, seed=config["seed"])
        print(json.dumps({"split": part, "stage": stage, **metrics}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--stage", choices=["sft", "grpo"], default="grpo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output:
        config["output"] = args.output
    evaluate(config, args.split, args.stage, args.checkpoint, args.max_val_samples, args.max_test_samples)


if __name__ == "__main__":
    main()
