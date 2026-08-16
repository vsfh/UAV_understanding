#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from clear_uav.evaluation import evaluation_samples, save_result, scored_metrics
from clear_uav.experiment_config import load_yaml, project_path
from clear_uav.openclip_finetune import load_finetuned_checkpoint, pooled_features
from clear_uav.ontology import load_label_subset, load_ontology


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenCLIP checkpoints")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    ontology = load_ontology(project_path(config["data"]["ontology"]))
    labels = list(load_label_subset(project_path(config["data"]["labels"]), ontology))
    device = torch.device(config["runtime"]["device"])
    model_path = project_path(config["model"]["path"])

    for model_config in config["checkpoints"]:
        for protocol in config["data"]["protocols"]:
            for seed in model_config["seeds"]:
                values = {"protocol": protocol, "seed": seed, "name": model_config["name"]}
                checkpoint_path = project_path(model_config["path"], **values)
                samples = evaluation_samples(config, protocol, labels)
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForZeroShotImageClassification.from_pretrained(
                    model_path, local_files_only=True, dtype=torch.bfloat16
                ).to(device)
                head, checkpoint = load_finetuned_checkpoint(
                    checkpoint_path, model=model, device=device
                )
                view = checkpoint["view"]
                model.eval()
                predictions, scores = [], []
                batch_size = config["test"]["batch_size"]
                with torch.inference_mode():
                    for start in tqdm(
                        range(0, len(samples), batch_size),
                        desc=f"{model_config['name']} {protocol} seed{seed}",
                        unit="batch",
                    ):
                        batch = samples[start : start + batch_size]
                        paths = [
                            sample.context_path if view == "context" else sample.evidence_path
                            for sample in batch
                        ]
                        images = []
                        for path in paths:
                            with Image.open(path) as source:
                                images.append(source.convert("RGB"))
                        inputs = processor(images=images, return_tensors="pt").to(device)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            features = pooled_features(model.get_image_features(**inputs))
                            logits = head(features).float().cpu()
                        indices = logits.argmax(-1).tolist()
                        predictions.extend({labels[index]} for index in indices)
                        scores.extend(
                            {label: float(value) for label, value in zip(labels, row.tolist())}
                            for row in logits
                        )
                metrics = scored_metrics(samples, predictions, scores, labels, ontology)
                output = project_path(config["output"]["path"], **values)
                save_result(output, values, samples, predictions, scores, metrics)
                print(f"{model_config['name']} {protocol} seed{seed}: {metrics}")
                del model, head
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
