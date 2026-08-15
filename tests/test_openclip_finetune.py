from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from clear_uav.openclip_finetune import (
    OpenCLIPClassifier,
    atomic_torch_save,
    checkpoint_payload,
    load_finetuned_checkpoint,
    pooled_features,
    set_trainable_mode,
)


class DummyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(4, 3)
        self.visual_projection = nn.Linear(3, 2, bias=False)
        self.text_model = nn.Linear(4, 3)


def test_pooled_features_accepts_tensor_and_pooler_output() -> None:
    tensor = torch.randn(2, 3)
    assert pooled_features(tensor) is tensor
    assert pooled_features(SimpleNamespace(pooler_output=tensor)) is tensor


def test_trainable_modes_freeze_text_and_select_visual_backbone() -> None:
    model = DummyCLIP()
    assert set_trainable_mode(model, "linear_probe") == []
    assert not any(parameter.requires_grad for parameter in model.parameters())

    visual_parameters = set_trainable_mode(model, "full_finetune")
    assert visual_parameters
    assert all(parameter.requires_grad for parameter in model.vision_model.parameters())
    assert all(
        parameter.requires_grad for parameter in model.visual_projection.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.text_model.parameters())


def test_full_finetune_checkpoint_restores_visual_and_classifier(tmp_path: Path) -> None:
    model = DummyCLIP()
    classifier = OpenCLIPClassifier(2, 2)
    path = tmp_path / "openclip_classifier.pt"
    atomic_torch_save(
        checkpoint_payload(
            model=model,
            classifier=classifier,
            mode="full_finetune",
            labels=["one", "two"],
            prompt="definition",
            view="context",
            epoch=2,
            best_macro_f1=0.75,
        ),
        path,
    )

    restored_model = DummyCLIP()
    for parameter in restored_model.parameters():
        nn.init.zeros_(parameter)
    restored_classifier, metadata = load_finetuned_checkpoint(
        path, model=restored_model, device=torch.device("cpu")
    )

    assert metadata["mode"] == "full_finetune"
    assert metadata["labels"] == ["one", "two"]
    assert torch.equal(
        restored_model.vision_model.weight, model.vision_model.weight
    )
    assert torch.equal(restored_classifier.head.weight, classifier.head.weight)
