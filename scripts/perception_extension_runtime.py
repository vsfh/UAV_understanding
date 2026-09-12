"""Offline Qwen loading for the Perception extensions on the SSHFS model cache.

Transformers 5.8 supports disable_mmap=True. Eager weight reads avoid the tiny
random mmap page reads observed with this repository's SSHFS-mounted hf_cache.
This changes loading only; no checkpoint is copied or modified on disk.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clear_uav.modeling import enable_offline_mode, require_local_model


def load_qwen(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
    disable_mmap: bool = True,
):
    """Load existing local weights; default to SSHFS-compatible eager reads."""
    model_path = require_local_model(path)
    enable_offline_mode()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    print(f"[perception loader] {model_path}; disable_mmap={disable_mmap}; offline=True", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        local_files_only=True,
        disable_mmap=disable_mmap,
    )
    return model, processor


def build_baseline(config, checkpoint):
    """Load a frozen baseline-compatible checkpoint without its old loader.

    Compatible extension callers may attach their independently saved heads
    after this function returns. The baseline box head, adapter and special
    token processor are fully restored; all parameters are frozen for inference.
    """
    from peft import PeftModel
    from train_perception_qwen import PerceptionQwen, VIS

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "box_head.pt").is_file() or not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"A complete Perception adapter/box checkpoint is required: {checkpoint}")
    vlm, _ = load_qwen(config["model"]["path"], disable_mmap=config["model"].get("disable_mmap", True))
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    vis_id = processor.tokenizer.convert_tokens_to_ids(VIS)
    if vis_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"Checkpoint tokenizer is missing {VIS}: {checkpoint}")
    vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    vlm = PeftModel.from_pretrained(vlm, checkpoint, is_trainable=False)
    model = PerceptionQwen(vlm, vis_id, config["model"]["head_dim"])
    model.box_head.load_state_dict(torch.load(checkpoint / "box_head.pt", map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    model.eval()
    return model.to(config["device"]), processor
