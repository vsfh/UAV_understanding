#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from tqdm.auto import tqdm

from clear_uav.evaluation import evaluation_samples, save_result
from clear_uav.experiment_config import load_yaml, project_path
from clear_uav.metrics import classification_metrics, group_sets
from clear_uav.modeling import load_qwen
from clear_uav.ontology import load_label_subset, load_ontology
from clear_uav.prompts import conversation


def parse_events(text: str, labels: set[str]) -> set[str]:
    return set(json.loads(text)["events"]) & labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Qwen3-VL base models and LoRA adapters")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    ontology = load_ontology(project_path(config["data"]["ontology"]))
    labels = list(load_label_subset(project_path(config["data"]["labels"]), ontology))
    label_set = set(labels)
    model_path = project_path(config["model"]["path"])

    for model_config in config["adapters"]:
        for protocol in config["data"]["protocols"]:
            for seed in model_config["seeds"]:
                values = {"protocol": protocol, "seed": seed, "name": model_config["name"]}
                samples = evaluation_samples(config, protocol, labels)
                model, processor = load_qwen(model_path, device_map="auto")
                adapter = model_config.get("path")
                if adapter:
                    model = PeftModel.from_pretrained(
                        model, project_path(adapter, **values), local_files_only=True
                    )
                model.eval()
                predictions = []
                rows = []
                with torch.inference_mode():
                    for sample in tqdm(
                        samples,
                        desc=f"{model_config['name']} {protocol} seed{seed}",
                        unit="sample",
                    ):
                        inputs = processor.apply_chat_template(
                            conversation(
                                sample.context_path,
                                sample.evidence_path,
                                config["test"]["view"],
                            ),
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            processor_kwargs={
                                "size": {
                                    "longest_edge": config["test"]["max_pixels"],
                                    "shortest_edge": 65_536,
                                }
                            },
                        ).to(model.device)
                        generated = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=config["test"]["max_new_tokens"],
                        )
                        text = processor.decode(
                            generated[0, inputs["input_ids"].shape[1] :],
                            skip_special_tokens=True,
                        ).strip()
                        try:
                            prediction = parse_events(text, label_set)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            prediction = set()
                        predictions.append(prediction)
                        rows.append({label: float(label in prediction) for label in labels})

                targets = [{sample.label} for sample in samples]
                pair_metrics = classification_metrics(targets, predictions, labels)
                set_targets, set_predictions = group_sets(
                    [sample.content_group_id for sample in samples],
                    [sample.label for sample in samples],
                    predictions,
                )
                metrics = {
                    "pair": pair_metrics,
                    "set_union": classification_metrics(set_targets, set_predictions, labels),
                }
                output = project_path(config["output"]["path"], **values)
                save_result(output, values, samples, predictions, rows, metrics)
                print(f"{model_config['name']} {protocol} seed{seed}: {metrics}")
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
