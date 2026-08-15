from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


CHECKPOINT_VERSION = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pooled_features(output: object) -> torch.Tensor:
    """Normalize CLIP feature outputs across Transformers releases."""
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError(f"Unsupported CLIP feature output: {type(output).__name__}")


def label_prompts(labels: list[str], definitions: dict[str, str], prompt: str) -> list[str]:
    if prompt == "direct":
        return [f"a UAV image of {label.replace('_', ' ')}" for label in labels]
    if prompt == "definition":
        return [f"a UAV image showing {definitions[label]}" for label in labels]
    raise ValueError(f"Unknown prompt: {prompt}")


class OpenCLIPClassifier(nn.Module):
    """A normalized linear classifier initialized from CLIP text embeddings."""

    def __init__(self, feature_dim: int, num_labels: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(nn.functional.normalize(features.float(), dim=-1))

    @torch.no_grad()
    def initialize_from_text(self, text_features: torch.Tensor) -> None:
        normalized = nn.functional.normalize(text_features.float(), dim=-1)
        if normalized.shape != self.head.weight.shape:
            raise ValueError(
                f"Text feature shape {tuple(normalized.shape)} does not match classifier "
                f"shape {tuple(self.head.weight.shape)}"
            )
        self.head.weight.copy_(normalized)
        self.head.bias.zero_()


def set_trainable_mode(model: nn.Module, mode: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "linear_probe":
        return []
    if mode != "full_finetune":
        raise ValueError(f"Unknown OpenCLIP training mode: {mode}")

    vision_model = getattr(model, "vision_model", None)
    visual_projection = getattr(model, "visual_projection", None)
    if vision_model is None or visual_projection is None:
        raise TypeError(
            "OpenCLIP model must expose vision_model and visual_projection for full fine-tuning"
        )
    parameters = []
    for module in (vision_model, visual_projection):
        for parameter in module.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)
    return parameters


def trainable_parameter_count(*modules: nn.Module) -> int:
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def checkpoint_payload(
    *,
    model: nn.Module,
    classifier: OpenCLIPClassifier,
    mode: str,
    labels: list[str],
    prompt: str,
    view: str,
    epoch: int,
    best_macro_f1: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "mode": mode,
        "labels": labels,
        "prompt": prompt,
        "view": view,
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "classifier_state_dict": {
            key: value.detach().cpu() for key, value in classifier.state_dict().items()
        },
    }
    if mode == "full_finetune":
        payload["vision_state_dict"] = {
            key: value.detach().cpu()
            for key, value in model.vision_model.state_dict().items()
        }
        payload["visual_projection_state_dict"] = {
            key: value.detach().cpu()
            for key, value in model.visual_projection.state_dict().items()
        }
    return payload


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_finetuned_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    device: torch.device,
) -> tuple[OpenCLIPClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported OpenCLIP checkpoint version in {path}")
    labels = checkpoint.get("labels")
    state = checkpoint.get("classifier_state_dict")
    if not isinstance(labels, list) or not isinstance(state, dict):
        raise ValueError(f"Invalid OpenCLIP checkpoint: {path}")
    weight = state.get("head.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"Invalid classifier weight in {path}")

    if checkpoint.get("mode") == "full_finetune":
        model.vision_model.load_state_dict(checkpoint["vision_state_dict"], strict=True)
        model.visual_projection.load_state_dict(
            checkpoint["visual_projection_state_dict"], strict=True
        )
    classifier = OpenCLIPClassifier(weight.shape[1], weight.shape[0])
    classifier.load_state_dict(state, strict=True)
    classifier.to(device)
    classifier.eval()
    return classifier, checkpoint
