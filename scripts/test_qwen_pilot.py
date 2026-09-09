"""Fixed validation-only screening; never reads test labels or retunes thresholds."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import statistics
import sys

import torch
from tqdm import tqdm

from train_qwen_pilot import DEFAULT, settings, diverse_order, load_pilot
from clear_uav.table4 import (labels_from_config, load_qwen_adapter,
    predict_qwen_discovery, read_discovery_samples, table4_metrics,
    QwenDiscoveryCollator, parse_grounding_tokens, location_token_strings,
    grounding_prefix_allowed_tokens)


def iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def roi_prior(pilot, config):
    # CPU-only geometry intervention. Fit class-wise center/size priors on TRAIN labels only.
    root = Path(pilot["output"])
    train = read_discovery_samples(config, pilot["protocol"], "train")
    labels = labels_from_config(config)
    def shape(b):
        return [(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]]
    priors = {}
    for label in labels:
        boxes = [shape(s.bbox_1000) for s in train if s.presence and s.label == label]
        priors[label] = [statistics.median(row[i] for row in boxes) for i in range(4)]
    base = json.loads((root / "baseline_fresh_probe.json").read_text())
    by_id = {s.record_uid: s for s in read_discovery_samples(config, pilot["protocol"], "val")}
    samples = [by_id[r["uid"]] for r in base["rows"]]
    result = {"fit_split": "train", "eval_split": "validation_probe", "priors": priors, "variants": {}}
    for mode in ["center", "size", "both"]:
        for alpha in [.25, .5]:
            predictions = copy.deepcopy([r["prediction"] for r in base["rows"]])
            for p in predictions:
                if p.get("bbox_1000") is None or p.get("category") not in priors:
                    continue
                values = shape(p["bbox_1000"])
                axes = range(2) if mode == "center" else range(2, 4) if mode == "size" else range(4)
                for i in axes:
                    values[i] = (1-alpha)*values[i] + alpha*priors[p["category"]][i]
                x, y, w, h = values
                p["bbox_1000"] = [max(0., x-w/2), max(0., y-h/2), min(1000., x+w/2), min(1000., y+h/2)]
            metrics = table4_metrics(samples, predictions, labels, base["metrics"]["threshold"], True)
            joint = sum(s.presence and s.label == p.get("category") and iou(p.get("bbox_1000"), s.bbox_1000) >= .5
                        for s, p in zip(samples, predictions)) / sum(s.presence for s in samples)
            # The probe is exactly class-balanced, so micro and macro joint success coincide.
            result["variants"][f"{mode}_{alpha}"] = {"metrics": metrics, "joint50": joint}
    (root / "roi_prior_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {"joint50": v["joint50"], "g_map50": v["metrics"]["g_map50"]}
                      for k, v in result["variants"].items()}, indent=2))


@torch.inference_mode()
def rollout_probe(pilot, config, loaded=None):
    # Training examples only. Oracle best-of-4 is a feasibility diagnostic, not an inference score.
    root = Path(pilot["output"])
    samples = read_discovery_samples(config, pilot["protocol"], "train")
    selected = []
    for label in labels_from_config(config) + ["no_event"]:
        selected.extend(diverse_order([s for s in samples if (s.label or "no_event") == label])[:2])
    config["output"]["adapter"] = pilot["checkpoint"]
    torch.manual_seed(pilot["seed"])
    model, processor = loaded or load_pilot(config, pilot["checkpoint"])
    model.eval()
    collator = QwenDiscoveryCollator(processor, config, False)
    labels = labels_from_config(config)
    ids = processor.tokenizer.convert_tokens_to_ids(location_token_strings(config["generation"]["location_tokens"]))
    rows = []
    for sample in tqdm(selected, desc="GRPO train-only feasibility", unit="image"):
        inputs = processor.apply_chat_template([collator.messages(sample)], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "size": {"longest_edge": config["input"]["max_pixels"],
                                                        "shortest_edge": config["input"]["min_pixels"]}}).to("cuda")
        length = inputs["input_ids"].shape[1]
        constraint = grounding_prefix_allowed_tokens(processor.tokenizer, labels,
            location_token_ids=ids, prompt_length=length)
        candidates = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Serial candidates bound KV memory while sharing GPU with the existing job.
            for _ in range(4):
                generated = model.generate(**inputs, do_sample=True, temperature=.8, top_p=.95,
                    max_new_tokens=config["generation"]["max_new_tokens"], prefix_allowed_tokens_fn=constraint)
                prediction = parse_grounding_tokens(generated[0, length:], processor.tokenizer, labels, ids)
                overlap = iou(prediction.get("bbox_1000"), sample.bbox_1000)
                correct = prediction.get("category") == sample.label
                reward = (.25 + .75 * overlap) * correct if sample.presence else float(prediction.get("category") in [None, "no_event"])
                candidates.append({"prediction": prediction, "iou": overlap, "reward": reward})
        rows.append({"uid": sample.record_uid, "positive": sample.presence, "candidates": candidates})
    reward_sets = [[p["reward"] for p in r["candidates"]] for r in rows]
    summary = {"n_train_images": len(rows), "samples_per_image": 4,
        "nonzero_reward_variance_fraction": sum(max(r)-min(r) > 1e-6 for r in reward_sets) / len(rows),
        "mean_sample_reward": sum(sum(r)/4 for r in reward_sets)/len(rows),
        "oracle_best_reward": sum(max(r) for r in reward_sets)/len(rows),
        "all_zero_reward_fraction": sum(max(r) == 0 for r in reward_sets)/len(rows)}
    (root / "grpo_feasibility.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


def evaluate(pilot, config, arm, full_val=False, loaded=None, confirm=False):
    root = Path(pilot["output"])
    samples = read_discovery_samples(config, pilot["protocol"], "val")
    if not full_val or confirm:
        ids = set(json.loads((root / "val_probe.json").read_text()))
        samples = [s for s in samples if (s.record_uid not in ids if confirm else s.record_uid in ids)]
        if not confirm:
            assert len(samples) == len(ids)
    calibration_path = config["output"]["calibration"].format(protocol=pilot["protocol"], seed=pilot["seed"])
    threshold = json.loads(Path(calibration_path).read_text())["threshold"]
    if arm == "baseline":
        cache = Path(config["output"]["validation_predictions"].format(protocol=pilot["protocol"], seed=pilot["seed"]))
        by_id = {r["record_uid"]: r["prediction"] for r in json.loads(cache.read_text())}
        predictions = [by_id[s.record_uid] for s in samples]
    else:
        config["output"]["adapter"] = pilot["checkpoint"] if arm == "baseline_fresh" else str(root / arm / "adapter")
        config["test"]["batch_size"] = pilot.get("eval_batch_size", 4)
        model, processor = loaded or load_pilot(config, config["output"]["adapter"])
        model.eval()
        processor.tokenizer.padding_side = "left"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = predict_qwen_discovery(model, processor, samples, config, torch.device("cuda"), arm + " validation")
    metrics = table4_metrics(samples, predictions, labels_from_config(config), threshold, True)
    rows = [{"uid": s.record_uid, "label": s.label, "presence": s.presence, "group": s.group_id,
             "target": s.bbox_1000, "prediction": p, "iou": iou(p.get("bbox_1000"), s.bbox_1000)}
            for s, p in zip(samples, predictions)]
    positives = [r for r in rows if r["presence"]]
    per_class = {}
    for label in labels_from_config(config):
        selected = [r for r in positives if r["label"] == label]
        if selected:
            per_class[label] = {"n": len(selected), "iou50": sum(r["iou"] >= .5 for r in selected) / len(selected),
                "joint50": sum(r["iou"] >= .5 and r["label"] == r["prediction"].get("category") for r in selected) / len(selected)}
    diagnostics = {"positive_n": len(positives),
        "candidate_iou50": sum(r["iou"] >= .5 for r in positives) / len(positives),
        "candidate_class_accuracy": sum(r["label"] == r["prediction"].get("category") for r in positives) / len(positives),
        "macro_joint50": sum(v["joint50"] for v in per_class.values()) / len(per_class),
        "per_class": per_class}
    suffix = "confirm" if confirm else "full_val" if full_val else "probe"
    (root / f"{arm}_{suffix}.json").write_text(json.dumps({"split": "val", "threshold_source": calibration_path,
        "metrics": metrics, "diagnostics": diagnostics, "rows": rows}, indent=2))
    print(json.dumps({"arm": arm, "n": len(rows), "metrics": metrics,
                      "diagnostics": {k: v for k, v in diagnostics.items() if k != "per_class"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--full-val", action="store_true")
    parser.add_argument("--rollouts", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="Remaining validation images, excluding screening probe")
    parser.add_argument("--roi-prior", action="store_true", help="CPU-only train-fitted geometry-prior ablation")
    args = parser.parse_args()
    pilot, config = settings(args.config)
    if args.roi_prior:
        roi_prior(pilot, config)
    elif args.rollouts:
        rollout_probe(pilot, config)
    else:
        evaluate(pilot, config, args.arm, args.full_val, confirm=args.confirm)
