#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from clear_uav.evaluation import evaluation_samples
from clear_uav.experiment_config import load_yaml, project_path
from clear_uav.metrics import classification_metrics, group_sets
from clear_uav.ontology import Ontology, load_label_subset, load_ontology


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def render_prompt(template: str, labels: list[str], ontology: Ontology) -> str:
    label_list = "\n".join(f"- {label}" for label in labels)
    definition_list = "\n".join(
        f"- {label}: {ontology.definitions[label]}" for label in labels
    )
    try:
        return template.format(label_list=label_list, definition_list=definition_list)
    except KeyError as error:
        raise ValueError(f"Unknown prompt placeholder: {error.args[0]}") from error


def _canonical_text(value: str) -> str:
    value = value.casefold().replace("-", " ").replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def parse_label(text: str, labels: list[str]) -> set[str]:
    """Parse one unambiguous canonical label from a free-form GeoChat response."""
    normalized = _canonical_text(text)
    aliases = {_canonical_text(label): label for label in labels}
    if normalized in aliases:
        return {aliases[normalized]}

    occurrences = [
        (match.start(), match.end(), label)
        for alias, label in aliases.items()
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized
        )
    ]
    # Prefer a specific label over its contained parent phrase, e.g.
    # "muck_truck_covered" over "muck_truck". Distinct maximal labels remain ambiguous.
    matches = {
        label
        for start, end, label in occurrences
        if not any(
            other_start <= start
            and end <= other_end
            and (other_start, other_end) != (start, end)
            for other_start, other_end, _ in occurrences
        )
    }
    return matches if len(matches) == 1 else set()


def _require_checkpoint(path: Path) -> None:
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"GeoChat config is missing: {path / 'config.json'}")
    if not any(path.glob("pytorch_model*.bin")) and not any(path.glob("*.safetensors")):
        raise FileNotFoundError(f"GeoChat weights are missing from {path}")


@contextmanager
def local_checkpoint(model_path: Path, vision_tower: Path) -> Iterator[Path]:
    """Expose a temporary checkpoint config pointing at the local vision tower."""
    _require_checkpoint(model_path)
    if not (vision_tower / "config.json").is_file():
        raise FileNotFoundError(f"GeoChat vision tower is missing: {vision_tower}")

    with tempfile.TemporaryDirectory(prefix="geochat-eval-") as temporary:
        patched_path = Path(temporary) / "geochat"
        patched_path.mkdir()
        for source in model_path.iterdir():
            if source.name == "config.json":
                continue
            target = patched_path / source.name
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())

        model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        model_config["mm_vision_tower"] = str(vision_tower.resolve())
        (patched_path / "config.json").write_text(
            json.dumps(model_config, indent=2), encoding="utf-8"
        )
        yield patched_path


def import_geochat(code_root: Path):
    if code_root.is_dir():
        sys.path.insert(0, str(code_root))
    try:
        from geochat.conversation import Chat, conv_templates
        from geochat.mm_utils import get_model_name_from_path
        from geochat.model.builder import load_pretrained_model
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Could not import the official GeoChat package and its dependencies. "
            f"No usable source was found at {code_root}. Create the environment described by "
            "GeoChat, then set GEOCHAT_PYTHON and, if needed, GEOCHAT_ROOT."
        ) from error
    return Chat, conv_templates, get_model_name_from_path, load_pretrained_model


def generate_response(
    *,
    chat,
    conv_templates,
    conv_mode: str,
    model,
    tokenizer,
    image_path: Path,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
) -> str:
    if conv_mode not in conv_templates:
        raise ValueError(f"Unknown GeoChat conversation mode: {conv_mode}")

    conversation = conv_templates[conv_mode].copy()
    image_list = []
    with Image.open(image_path) as image:
        chat.upload_img(image.convert("RGB"), conversation, image_list)
    chat.ask(prompt, conversation)
    chat.encode_img(image_list)
    generation = chat.answer_prepare(
        conversation,
        image_list,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    input_length = generation["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=generation["input_ids"],
        images=generation["images"],
        stopping_criteria=generation["stopping_criteria"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    return tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()


def save_result(
    path: Path,
    run_config: dict,
    prompt: str,
    samples,
    predictions: list[set[str]],
    responses: list[str],
    labels: list[str],
    metrics: dict,
) -> None:
    rows = []
    for sample, prediction, response in zip(samples, predictions, responses):
        rows.append(
            {
                "record_uid": sample.record_uid,
                "target": sample.label,
                "prediction": sorted(prediction),
                "response": response,
                "parse_success": len(prediction) == 1,
                # These are generated-label indicators, not calibrated ranking scores.
                "scores": {label: float(label in prediction) for label in labels},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": run_config,
                "prompt": prompt,
                "num_samples": len(samples),
                "metrics": metrics,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot GeoChat evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    seed = int(config["runtime"]["seed"])
    if seed != 43:
        raise ValueError(f"GeoChat evaluation seed must be 43, got {seed}")
    set_seed(seed)

    view = config["test"]["view"]
    if view not in {"context", "evidence"}:
        raise ValueError("GeoChat accepts one image; test.view must be 'context' or 'evidence'")

    ontology = load_ontology(project_path(config["data"]["ontology"]))
    labels = list(load_label_subset(project_path(config["data"]["labels"]), ontology))
    prompts = config.get("prompts", [])
    prompt_names = [item["name"] for item in prompts]
    if not prompts or len(prompt_names) != len(set(prompt_names)):
        raise ValueError("prompts must contain at least one uniquely named prompt")

    code_root_value = os.environ.get("GEOCHAT_ROOT", config["model"]["code_root"])
    code_root = project_path(code_root_value)
    Chat, conv_templates, get_model_name, load_model = import_geochat(code_root)
    model_path = project_path(config["model"]["path"])
    vision_tower = project_path(config["model"]["vision_tower"])
    device = config["runtime"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but CUDA is unavailable")

    with local_checkpoint(model_path, vision_tower) as checkpoint:
        tokenizer, model, image_processor, _ = load_model(
            str(checkpoint),
            None,
            get_model_name(str(checkpoint)),
            load_8bit=bool(config["model"].get("load_8bit", False)),
            load_4bit=bool(config["model"].get("load_4bit", False)),
            device_map={"": device},
            device=device,
        )
        model.eval()
        chat = Chat(model, image_processor, tokenizer, device=device)

        for protocol in config["data"]["protocols"]:
            samples = evaluation_samples(config, protocol, labels)
            for prompt_config in prompts:
                prompt_name = prompt_config["name"]
                prompt = render_prompt(prompt_config["template"], labels, ontology)
                predictions: list[set[str]] = []
                responses: list[str] = []
                with torch.inference_mode():
                    for sample in tqdm(
                        samples,
                        desc=f"geochat {prompt_name} {protocol} seed{seed}",
                        unit="sample",
                    ):
                        image_path = (
                            sample.context_path if view == "context" else sample.evidence_path
                        )
                        response = generate_response(
                            chat=chat,
                            conv_templates=conv_templates,
                            conv_mode=config["test"]["conv_mode"],
                            model=model,
                            tokenizer=tokenizer,
                            image_path=image_path,
                            prompt=prompt,
                            max_length=int(config["test"]["max_length"]),
                            max_new_tokens=int(config["test"]["max_new_tokens"]),
                        )
                        responses.append(response)
                        predictions.append(parse_label(response, labels))

                targets = [{sample.label} for sample in samples]
                set_targets, set_predictions = group_sets(
                    [sample.content_group_id for sample in samples],
                    [sample.label for sample in samples],
                    predictions,
                )
                metrics = {
                    "pair": classification_metrics(targets, predictions, labels),
                    "set_union": classification_metrics(set_targets, set_predictions, labels),
                    "parse_success_rate": (
                        sum(len(prediction) == 1 for prediction in predictions) / len(predictions)
                        if predictions
                        else 0.0
                    ),
                }
                values = {"prompt": prompt_name, "protocol": protocol, "seed": seed}
                output = project_path(config["output"]["path"], **values)
                run_config = {
                    **values,
                    "model": "geochat",
                    "view": view,
                    "decoder": "greedy_free_generation",
                }
                save_result(
                    output,
                    run_config,
                    prompt,
                    samples,
                    predictions,
                    responses,
                    labels,
                    metrics,
                )
                print(f"geochat {prompt_name} {protocol} seed{seed}: {metrics}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
