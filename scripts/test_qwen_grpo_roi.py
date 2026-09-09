"""Evaluate generated versus residual-corrected ROIs with identical text outputs."""
import argparse
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm
from train_qwen_grpo_roi import DEFAULT, build_model, read_config, rollout
from clear_uav.table4 import (QwenDiscoveryCollator, event_probability,
    labels_from_config, read_discovery_samples, save_results, select_threshold, table4_metrics)


@torch.no_grad()
def predict(model, processor, samples, config):
    model.eval()
    labels = labels_from_config(config)
    collator = QwenDiscoveryCollator(processor, config, False)
    original, refined = [], []
    device = torch.device(config["device"])
    for example in tqdm(samples, desc="Qwen generation + ROI residual", unit="image"):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        inputs = {k: v.to(device) for k, v in collator([example]).items()}
        row = rollout(model, processor, inputs, labels, config, sample=False)[0]
        base = row["prediction"] | {"presence_score": event_probability(row["first_scores"], processor.tokenizer, labels),
            "valid": True, "raw_output": processor.tokenizer.decode(row["tokens"], skip_special_tokens=False),
            "num_calls": 1, "inference_batch_size": 1, "timing_scope": "preprocess_and_generate"}
        torch.cuda.synchronize(device)
        base["latency_ms"] = (time.perf_counter() - started) * 1000
        prediction = dict(base)
        if base["bbox_1000"] is not None:
            box = torch.tensor([base["bbox_1000"]], device=device) / 1000
            _, boxes, _ = model.replay(inputs, row["tokens"], box)
            prediction["bbox_1000"] = (boxes[0].float() * 1000).cpu().tolist()
            prediction["num_calls"] = 2
        torch.cuda.synchronize(device)
        prediction["latency_ms"] = (time.perf_counter() - started) * 1000
        prediction["timing_scope"] = "preprocess_generate_and_conditional_roi_replay"
        original.append(base)
        refined.append(prediction)
    return {"generated_roi": original, "residual_roi": refined}


def evaluate(args):
    current = read_config(args.config)
    checkpoint = Path(args.checkpoint or (Path(current["output"]) / "latest.txt").read_text().strip())
    config = read_config(checkpoint / "config.yaml")
    config["device"] = current["device"]
    model, processor = build_model(config, checkpoint)
    labels = labels_from_config(config)
    # Probe runs never create or overwrite the full-validation calibration.
    output = checkpoint / (f"probe_{args.limit}" if args.limit else "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    for split in (["val", "test"] if args.split == "all" else [args.split]):
        samples = read_discovery_samples(config, config["protocol"], split)
        if args.limit:
            samples = samples[:args.limit]
        calibration = output / "calibration.json"
        if split == "test":
            threshold = json.loads(calibration.read_text())["threshold"]
        predictions = predict(model, processor, samples, config)
        if split == "val":
            threshold, selection = select_threshold(samples, predictions["generated_roi"], config["validation"])
            calibration.write_text(json.dumps({"threshold": threshold, "selection": selection,
                "n_val": len(samples), "checkpoint": str(checkpoint)}, indent=2))
        summary = {"split": split, "threshold": threshold}
        for name, rows in predictions.items():
            metrics = table4_metrics(samples, rows, labels, threshold, classification=True)
            save_results(output / f"{split}_{name}.json", config, config["protocol"], samples, rows, metrics, seed=config["seed"])
            summary[name] = {k: v for k, v in metrics.items() if k in ["c_f1", "g_map50", "n_fpr", "p_recall"]}
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--checkpoint", help="Defaults to latest.txt from the YAML output")
    parser.add_argument("--split", choices=["val", "test", "all"], default="val")
    parser.add_argument("--limit", type=int, help="Smoke probe only; stored separately from full results")
    evaluate(parser.parse_args())
