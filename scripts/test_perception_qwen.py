"""Generate categories, regress contextual ROIs, and run the existing paper metrics."""
import argparse
import json
import time

import torch
from tqdm import tqdm

from train_perception_qwen import (
    VIS, DEFAULT_CONFIG, Collator, build_model, output_path, read_config,
)
from clear_uav.generation_constraints import label_prefix_allowed_tokens
from clear_uav.table4 import (
    definitions_from_config, event_probability, read_discovery_samples,
    save_results, select_threshold, table4_metrics,
)


@torch.inference_mode()
def predict(model, processor, samples, config, labels):
    model.eval()
    collator = Collator(processor, config, training=False)
    tokenizer = processor.tokenizer
    answers = [label + VIS for label in labels] + ["no_event"]
    max_tokens = max(len(tokenizer.encode(a, add_special_tokens=False)) for a in answers) + 1
    predictions = []
    batch_size = config["test"]["batch_size"]
    device = torch.device(config["device"])
    for start in tqdm(range(0, len(samples), batch_size), desc="predict", unit="batch"):
        current = samples[start:start + batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        inputs, _, _ = collator(current)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        length = inputs["input_ids"].shape[1]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            generated = model.vlm.generate(
                **inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                prefix_allowed_tokens_fn=label_prefix_allowed_tokens(tokenizer, answers, prompt_length=length),
                return_dict_in_generate=True, output_scores=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        tokens = generated.sequences[:, length:]
        categories = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        # Replay predicted text so <vis> itself has been processed by the LLM.
        # generate()'s final state alone can correspond to the PRECEDING token.
        completion_mask = (tokens == tokenizer.eos_token_id).cumsum(1)
        completion_mask = (completion_mask - (tokens == tokenizer.eos_token_id).long()) == 0
        box_inputs = inputs | {
            "input_ids": generated.sequences,
            "attention_mask": torch.cat((inputs["attention_mask"], completion_mask.long()), dim=1),
            "logits_to_keep": 1,
        }
        if "mm_token_type_ids" in inputs:
            box_inputs["mm_token_type_ids"] = torch.cat((
                inputs["mm_token_type_ids"], torch.zeros_like(tokens),
            ), dim=1)
        # Release the generation KV cache before replaying the full sequence.
        first_scores = generated.scores[0]
        del generated
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, boxes = model(box_inputs)
        boxes = (boxes.float() * 1000).cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - started) * 1000 / len(current)
        for i, text in enumerate(categories):
            category = text.strip()
            positive = category in labels
            valid = (positive and model.vis_id in tokens[i]) or category == "no_event"
            predictions.append({
                "category": category if positive else None,
                "bbox_1000": boxes[i] if positive else None,
                "presence_score": event_probability(first_scores[i], tokenizer, labels),
                "valid": bool(valid), "latency_ms": latency,
                "raw_output": tokenizer.decode(tokens[i], skip_special_tokens=False),
                "num_calls": 2, "inference_batch_size": len(current),
                "timing_scope": "preprocess_generate_and_box_replay_amortized",
            })
    return predictions


def evaluate(config, split):
    output = output_path(config)
    # Always use the checkpoint's configuration, not a subsequently edited YAML.
    saved = read_config(output / "config.yaml")
    saved["device"], saved["test"] = config["device"], config["test"]
    config = saved
    model, processor = build_model(config, output / "best")
    labels, _ = definitions_from_config(config)
    for part in (["val", "test"] if split == "all" else [split]):
        samples = read_discovery_samples(config, config["protocol"], part)
        predictions = predict(model, processor, samples, config, labels)
        if part == "val":
            threshold, selection = select_threshold(samples, predictions, config["validation"])
            (output / "calibration.json").write_text(json.dumps({
                "threshold": threshold, "selection": selection,
                "policy": "max_recall_at_fpr", "max_n_fpr": config["validation"]["max_n_fpr"],
            }, indent=2))
        else:
            threshold = json.loads((output / "calibration.json").read_text())["threshold"]
        metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
        # The existing result writer expects this field; no separate config layer.
        config["train"]["seeds"] = [config["seed"]]
        save_results(output / f"{part}_results.json", config, config["protocol"],
                     samples, predictions, metrics, seed=config["seed"])
        print(json.dumps({"split": part, **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    args = parser.parse_args()
    evaluate(read_config(args.config), args.split)
