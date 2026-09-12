"""SFT with <vis>-to-image spatial cross-attention on final Qwen LLM states.

This is an architecture-inspired adaptation, not an MVP-LM reproduction. It
uses one image and one LLM pass, without raw ViT or multiscale feature fusion.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, set_seed
from tqdm import tqdm
import yaml

from train_perception_qwen import (
    VIS, Collator, LORA_PATTERNS, read_config, output_path,
    read_discovery_samples, discovery_sampler, losses, validation_loss,
)
from perception_extension_runtime import load_qwen
from perception_spatial_head import SpatialInteraction, spatial_memory, save_spatial_heads, load_spatial_heads

DEFAULT_CONFIG = "configs/yaml/perception_spatial.yaml"


class SpatialPerceptionQwen(nn.Module):
    def __init__(self, vlm, vis_id, config):
        super().__init__()
        self.vlm, self.vis_id = vlm, vis_id
        hidden_dim = vlm.config.text_config.hidden_size
        head_dim = config["model"]["head_dim"]
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(),
                                      nn.Linear(head_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 4))
        spatial = config["spatial"]
        self.spatial_head = SpatialInteraction(
            hidden_dim, spatial["interaction_dim"], spatial["num_heads"],
            spatial.get("dropout", 0.1), spatial.get("residual_gate_init", 0.0))
        self.image_token_id = vlm.config.image_token_id
        self.merge_size = vlm.config.vision_config.spatial_merge_size
        self.max_image_tokens = spatial.get("max_image_tokens", 2048)
        self._capture_context = None
        self.visual_state = None
        self._capture_hook = vlm.get_base_model().model.language_model.register_forward_hook(self.capture)

    def capture(self, module, args, output):
        if self._capture_context is None:
            return
        inputs, positions = self._capture_context
        hidden = output[0]
        query = hidden[torch.arange(len(hidden), device=hidden.device), positions]
        memory, coords, mask = spatial_memory(hidden, inputs["input_ids"], inputs["attention_mask"],
                                              self.image_token_id, inputs.get("image_grid_thw"),
                                              self.merge_size, self.max_image_tokens)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            self.visual_state = self.spatial_head(query, memory, coords, mask)

    def forward(self, inputs):
        ids = inputs["input_ids"]
        # Choose the LAST <vis>; the prompt also contains this special token.
        positions = ((ids == self.vis_id) * torch.arange(ids.shape[1], device=ids.device)).amax(1)
        self._capture_context = (inputs, positions)
        try:
            output = self.vlm(**inputs, use_cache=False)
            if self.visual_state is None:
                raise RuntimeError("Qwen language-model hook did not capture spatial states")
            with torch.autocast(device_type=ids.device.type, enabled=False):
                raw = self.box_head(self.visual_state.float()).sigmoid()
            lo, hi = torch.minimum(raw[:, :2], raw[:, 2:]), torch.maximum(raw[:, :2], raw[:, 2:])
            return output.loss, torch.cat((lo, hi), dim=-1)
        finally:
            self._capture_context = None
            self.visual_state = None


def build_model(config, checkpoint=None, is_trainable=False):
    """Load a full spatial checkpoint, or use initial_checkpoint for SFT warm start."""
    source = checkpoint or (config.get("initial_checkpoint") if is_trainable else None)
    vlm, processor = load_qwen(config["model"]["path"], disable_mmap=config["model"].get("disable_mmap", True))
    if source:
        source = Path(str(source).format(protocol=config["protocol"], seed=config["seed"]))
        processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    else:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [VIS]})
    processor.tokenizer.padding_side = "left"
    vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    vis_id = processor.tokenizer.convert_tokens_to_ids(VIS)
    if vis_id == processor.tokenizer.unk_token_id:
        raise ValueError("Warm-start tokenizer does not contain <vis>")
    if source:
        vlm = PeftModel.from_pretrained(vlm, source, is_trainable=is_trainable)
    else:
        settings = config["train"]
        vlm = get_peft_model(vlm, LoraConfig(
            r=settings["lora_r"], lora_alpha=settings["lora_alpha"], lora_dropout=settings["lora_dropout"],
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_PATTERNS["projector_llm"],
            trainable_token_indices={"model.language_model.embed_tokens": [vis_id], "lm_head": [vis_id]},
        ))
    if is_trainable:
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = SpatialPerceptionQwen(vlm, vis_id, config)
    if source:
        load_spatial_heads(model, source, require_spatial=checkpoint is not None)
    if not is_trainable:
        model.requires_grad_(False)
    return model.to(config["device"]), processor


def subset(samples, maximum):
    if maximum is None:
        return samples
    if maximum < 1:
        raise ValueError("Sample limits must be at least 1")
    return samples[:maximum]


def train(config):
    set_seed(config["seed"])
    settings = config["train"]
    if settings["epochs"] < 1:
        raise ValueError("epochs must be at least 1")
    output = output_path(config)
    if (output / "best").exists() or (output / "history.json").exists():
        raise FileExistsError(f"Refusing to overwrite an experiment: {output}. Choose --output.")
    samples = subset(read_discovery_samples(config, config["protocol"], "train"), config.get("max_train_samples"))
    val_samples = subset(read_discovery_samples(config, config["protocol"], "val"), config.get("max_val_samples"))
    if not samples or not val_samples:
        raise ValueError("Training and validation splits must both be nonempty")
    model, processor = build_model(config, is_trainable=True)
    collator = Collator(processor, config)
    loader = DataLoader(samples, batch_size=settings["batch_size"],
                        sampler=discovery_sampler(samples, settings, config["seed"]),
                        collate_fn=collator, num_workers=settings["num_workers"])
    val_loader = DataLoader(val_samples, batch_size=settings["batch_size"], collate_fn=collator,
                            num_workers=settings["num_workers"])
    if len(loader) == 0:
        raise ValueError("Sampler produced zero training batches")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": settings["learning_rate"]},
        {"params": list(model.box_head.parameters()) + list(model.spatial_head.parameters()),
         "lr": settings["head_learning_rate"]},
    ], weight_decay=settings["weight_decay"])
    accumulation = settings["gradient_accumulation"]
    updates = math.ceil(len(loader) / accumulation) * settings["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * settings["warmup_ratio"]), updates)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    best, history = float("inf"), []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"spatial train {epoch}/{settings['epochs']}")
        total = 0.0
        for step, batch in enumerate(progress):
            loss, parts = losses(model, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {step}")
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (loss / group_size).backward()
            total += loss.item()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.3f}", gate=f"{model.spatial_head.residual_gate.item():.4f}")
        val = validation_loss(model, val_loader, config)
        if not math.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": total / len(loader), "val_loss": val,
                        "residual_gate": model.spatial_head.residual_gate.detach().item()})
        if val < best:
            best = val
            checkpoint = output / "best"
            model.vlm.save_pretrained(checkpoint, save_embedding_layers=False)
            processor.save_pretrained(checkpoint)
            save_spatial_heads(model, checkpoint)
            (checkpoint / "extension.json").write_text(json.dumps({
                "extension": "spatial_interaction", "feature_source": "final_llm_image_tokens",
                "spatial": config["spatial"], "epoch": epoch, "val_loss": val,
            }, indent=2), encoding="utf-8")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch}: val_loss={val:.4f}, best={best:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = read_config(args.config)
    for name in ("initial_checkpoint", "max_train_samples", "max_val_samples", "output"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    train(config)


if __name__ == "__main__":
    main()
