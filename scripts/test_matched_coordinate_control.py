"""Evaluate fixed-epoch matched controls with validation-only alert calibration."""
import argparse
import json
import time

import torch
from tqdm import tqdm

from train_matched_coordinate_control import (
    ARMS, DEFAULT_CONFIG, VIS, Collator, build_model, output_path, read_config,
)
from clear_uav.generation_constraints import (
    grounding_prefix_allowed_tokens, label_prefix_allowed_tokens, location_token_strings,
)
from clear_uav.table4 import (
    definitions_from_config, event_probability, parse_grounding_tokens,
    read_discovery_samples, save_results, select_threshold, table4_metrics,
)


@torch.inference_mode()
def predict(model, processor, samples, config, labels, arm):
    model.eval()
    collator = Collator(processor, config, arm, training=False)
    tokenizer = processor.tokenizer
    location_ids = tokenizer.convert_tokens_to_ids(location_token_strings(config["model"]["location_tokens"]))
    answers = [label + VIS for label in labels] + ["no_event"]
    max_tokens = max(len(tokenizer.encode(a, add_special_tokens=False)) for a in answers) + 1
    if arm == "token":
        max_tokens += 4
    predictions = []
    batch_size, device = config["test"]["batch_size"], torch.device(config["device"])
    for start in tqdm(range(0, len(samples), batch_size), desc=f"{arm} predict", unit="batch"):
        current = samples[start:start + batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        inputs, _, _ = collator(current)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        length = inputs["input_ids"].shape[1]
        if arm == "token":
            grammar = grounding_prefix_allowed_tokens(
                tokenizer, [label + VIS for label in labels],
                location_token_ids=location_ids, prompt_length=length)
        else:
            grammar = label_prefix_allowed_tokens(tokenizer, answers, prompt_length=length)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            generated = model.vlm.generate(
                **inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                prefix_allowed_tokens_fn=grammar, return_dict_in_generate=True, output_scores=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        tokens = generated.sequences[:, length:]
        first_scores = generated.scores[0]
        if arm == "continuous":
            eos = tokens == tokenizer.eos_token_id
            mask = (eos.cumsum(1) - eos.long()) == 0
            box_inputs = inputs | {"input_ids": generated.sequences,
                                   "attention_mask": torch.cat((inputs["attention_mask"], mask.long()), 1),
                                   "logits_to_keep": 1}
            if "mm_token_type_ids" in inputs:
                box_inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tokens)), 1)
            del generated
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, boxes = model(box_inputs)
            boxes = (boxes.float() * 1000).cpu().tolist()
        else:
            del generated
        token_rows = tokens.cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - started) * 1000 / len(current)
        for i, ids in enumerate(token_rows):
            if arm == "token":
                pred = parse_grounding_tokens([t for t in ids if t != model.vis_id], tokenizer,
                                              labels, location_ids)
                valid = pred["category"] is None or model.vis_id in ids
            else:
                category = tokenizer.decode(ids, skip_special_tokens=True).strip()
                positive = category in labels
                valid = (positive and model.vis_id in ids) or category == "no_event"
                pred = {"category": category if positive else None,
                        "bbox_1000": boxes[i] if positive else None}
            pred.update(presence_score=event_probability(first_scores[i], tokenizer, labels),
                        valid=bool(valid), latency_ms=latency,
                        raw_output=tokenizer.decode(ids, skip_special_tokens=False),
                        num_calls=1 if arm == "token" else 2, inference_batch_size=len(current),
                        timing_scope="preprocess_generate_and_optional_roi_replay_amortized")
            predictions.append(pred)
    return predictions


def compare(config):
    paths = {arm: output_path(config, arm) for arm in ARMS}
    if not all((p / "evaluation_complete.json").exists() for p in paths.values()):
        return
    manifests = {arm: json.loads((p / "matched_manifest.json").read_text()) for arm, p in paths.items()}
    fields = ["seed", "epochs", "selected_epoch", "checkpoint_selection", "completed",
              "initial_trainable_sha256", "tokenizer_sha256", "recipe_sha256", "training_records_sha256",
              "samples_per_epoch", "optimizer_steps", "actual_optimizer_steps", "lr_trace_sha256",
              "epoch_sample_order_sha256"]
    mismatches = [key for key in fields if manifests["token"][key] != manifests["continuous"][key]]
    if mismatches or not all(m["completed"] for m in manifests.values()):
        raise ValueError(f"The runs are not matched: {mismatches}")
    results = {arm: json.loads((p / "test_results.json").read_text()) for arm, p in paths.items()}
    assert ([(r["record_uid"], r["target"]) for r in results["token"]["rows"]]
            == [(r["record_uid"], r["target"]) for r in results["continuous"]["rows"]])
    summary = {"matched": True, "matched_fields": fields,
               "checkpoint_selection": "fixed_final_epoch", "epoch": config["train"]["epochs"],
               "contrast": "coordinate-token CE vs continuous-head L1+GIoU, with shared semantic-prefix CE",
               "metrics": {arm: value["metrics"] for arm, value in results.items()}}
    destination = paths["token"].parent / "comparison.json"
    destination.write_text(json.dumps(summary, indent=2))
    print(f"Verified matched pair: {destination}", flush=True)


def evaluate(config, arm):
    output = output_path(config, arm)
    saved = read_config(output / "config.yaml")
    saved["device"], saved["test"] = config["device"], config["test"]
    config = saved
    manifest = json.loads((output / "matched_manifest.json").read_text())
    if not manifest["completed"]:
        raise ValueError("Training is not complete. This control evaluates the fixed final epoch only.")
    model, processor = build_model(config, output / f"epoch_{config['train']['epochs']}")
    labels, _ = definitions_from_config(config)
    for split in ("val", "test"):
        samples = read_discovery_samples(config, config["protocol"], split)
        predictions = predict(model, processor, samples, config, labels, arm)
        if split == "val":
            threshold, selection = select_threshold(samples, predictions, config["validation"])
            (output / "calibration.json").write_text(json.dumps({
                "threshold": threshold, "selection": selection, "policy": "max_recall_at_fpr",
                "max_n_fpr": config["validation"]["max_n_fpr"],
                "checkpoint": f"epoch_{config['train']['epochs']}",
            }, indent=2))
        metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
        config["train"]["seeds"] = [config["seed"]]
        save_results(output / f"{split}_results.json", config, config["protocol"],
                     samples, predictions, metrics, seed=config["seed"])
        print(json.dumps({"arm": arm, "split": split, **metrics}, indent=2), flush=True)
    (output / "evaluation_complete.json").write_text(json.dumps({
        "arm": arm, "epoch": config["train"]["epochs"], "splits": ["val", "test"],
    }))
    compare(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.compare_only:
        compare(config)
    else:
        if args.arm is None:
            parser.error("--arm is required for evaluation")
        evaluate(config, args.arm)
