from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


LANGUAGE_LORA_PATTERN = (
    r".*language_model\..*\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
PROJECTOR_LORA_PATTERN = (
    r".*visual\.(?:merger|deepstack_merger_list\.\d+)\."
    r"(?:linear_fc1|linear_fc2)"
)
LORA_PATTERNS = {
    "llm": LANGUAGE_LORA_PATTERN,
    "projector_llm": (
        rf"(?:{LANGUAGE_LORA_PATTERN}|{PROJECTOR_LORA_PATTERN})"
    ),
}


def enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def require_local_model(path: str | Path) -> Path:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in local model directory: {model_path}")
    return model_path


def load_qwen(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
):
    model_path = require_local_model(model_path)
    enable_offline_mode()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )
    return model, processor
