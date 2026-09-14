"""Continue the original Perception checkpoint with Spatial's exact SFT recipe.

The existing baseline training loop is reused in this process. Only its model
factory is replaced so the checkpoint's adapter and ROI head remain trainable.
No SpatialInteraction module is constructed, and no shared source is edited.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor

import train_perception_qwen as baseline
from perception_extension_runtime import load_qwen
from clear_uav.table4 import training_data_state

DEFAULT_CONFIG = "configs/yaml/perception_continue.yaml"
MATCHED_KEYS = ("protocol", "seed", "initial_checkpoint", "data", "model", "prompt",
                "input", "train", "loss", "test", "validation")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint):
    """Same content fingerprint as Spatial evaluation, for a baseline checkpoint."""
    checkpoint = Path(checkpoint)
    for name in ("box_head.pt", "adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if (checkpoint / "spatial_head.pt").exists():
        raise ValueError(f"This control requires a baseline checkpoint: {checkpoint}")
    digest = hashlib.sha256()
    for path in sorted(p for p in checkpoint.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(checkpoint)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def inspect_recipe(config):
    """CPU-only: compare the saved Spatial recipe and inspect current inputs."""
    reference_path = Path(config["matched_spatial_config"].format(**config))
    reference = baseline.read_config(reference_path)
    differences = [key for key in MATCHED_KEYS if config.get(key) != reference.get(key)]
    if differences:
        raise ValueError(f"Recipe differs from saved Spatial training: {differences}")
    if config["train"]["epochs"] != 3 or "spatial" in config:
        raise ValueError("This control requires three epochs and no spatial configuration")
    if any(config.get(key) is not None for key in ("max_train_samples", "max_val_samples", "max_test_samples")):
        raise ValueError("Matched continuation must use the complete splits")
    checkpoint = Path(config["initial_checkpoint"].format(**config)).resolve()
    output = baseline.output_path(config).resolve()
    reference_output = baseline.output_path(reference).resolve()
    if output.is_relative_to(reference_output) or output.is_relative_to(checkpoint.parent) or checkpoint.is_relative_to(output):
        raise ValueError("Continuation requires an independent output directory")
    fingerprint = checkpoint_fingerprint(checkpoint)
    train_samples = baseline.read_discovery_samples(config, config["protocol"], "train")
    val_samples = baseline.read_discovery_samples(config, config["protocol"], "val")
    if not train_samples or not val_samples:
        raise ValueError("Training and validation splits must both be nonempty")
    sampler = baseline.discovery_sampler(train_samples, config["train"], config["seed"])
    batches = math.ceil(len(sampler) / config["train"]["batch_size"])
    updates = math.ceil(batches / config["train"]["gradient_accumulation"]) * 3
    code_dir = Path(__file__).resolve().parent
    return {
        "experiment": "perception_continue", "spatial_module": False,
        "reference_config": str(reference_path.resolve()),
        "reference_config_sha256": sha256(reference_path), "matched_keys": list(MATCHED_KEYS),
        "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": fingerprint,
        "initial_file_sha256": {name: sha256(checkpoint / name) for name in ("adapter_model.safetensors", "box_head.pt")},
        "output": str(output), "output_exists": output.exists(),
        "train_records": len(train_samples), "val_records": len(val_samples),
        "train_positives": sum(sample.presence for sample in train_samples),
        "samples_per_epoch": len(sampler), "batches_per_epoch": batches,
        "optimizer_updates": updates, "warmup_updates": int(updates * config["train"]["warmup_ratio"]),
        "checkpoint_selection": "minimum validation language + L1 + GIoU loss",
        "training_data": training_data_state(config, config["protocol"]),
        "source_sha256": {name: sha256(code_dir / name) for name in (
            "train_perception_continue.py", "train_perception_qwen.py",
            "train_perception_spatial.py", "perception_extension_runtime.py")},
        "rng_note": "Same seed and independent weighted-sampler generator; removing Spatial changes dropout RNG consumption, so outputs are not bitwise paired.",
    }


def build_model(config, checkpoint=None):
    is_trainable = checkpoint is None
    source = Path(checkpoint or config["initial_checkpoint"].format(**config))
    vlm, _ = load_qwen(config["model"]["path"], disable_mmap=config["model"].get("disable_mmap", True))
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    vis_id = processor.tokenizer.convert_tokens_to_ids(baseline.VIS)
    if vis_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"Warm-start tokenizer does not contain {baseline.VIS}")
    vlm = PeftModel.from_pretrained(vlm, source, is_trainable=is_trainable)
    if is_trainable:
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = baseline.PerceptionQwen(vlm, vis_id, config["model"]["head_dim"])
    model.box_head.load_state_dict(torch.load(source / "box_head.pt", map_location="cpu", weights_only=True))
    if not is_trainable:
        model.requires_grad_(False)
    return model.to(config["device"]), processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true", help="Inspect the matched recipe, data and hashes without loading the VLM or writing outputs")
    args = parser.parse_args()
    config = baseline.read_config(args.config)
    if args.output is not None:
        config["output"] = args.output
    receipt = inspect_recipe(config)
    print(json.dumps(receipt, indent=2), flush=True)
    if args.check_only:
        return
    output = baseline.output_path(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "matched_recipe.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    original_builder = baseline.build_model
    baseline.build_model = build_model
    try:
        baseline.train(config)
    finally:
        baseline.build_model = original_builder


if __name__ == "__main__":
    main()
