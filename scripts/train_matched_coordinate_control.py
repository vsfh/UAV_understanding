"""Matched coordinate tokens vs continuous ROI head, from the same Qwen base.

Both arms instantiate identical parameters/tokenizers and share semantic-prefix
CE. Only coordinate serialization, its native loss, and decoding differ.
No warm start, continued stage, early stopping, or best-loss selection.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from torchvision.ops import generalized_box_iou_loss
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, set_seed

from train_perception_qwen import VIS, PerceptionQwen, read_config
from clear_uav.modeling import LORA_PATTERNS, load_qwen
from clear_uav.generation_constraints import location_token_strings
from clear_uav.table4 import category_block, discovery_sampler, read_discovery_samples

DEFAULT_CONFIG = "configs/yaml/matched_coordinate_control.yaml"
ARMS = ("token", "continuous")


def output_path(config, arm):
    return Path(config["output"].format(
        protocol=config["protocol"], seed=config["seed"], arm=arm))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_manifest(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def quantize_box(box, count):
    """Ordered xyxy on the existing 0..count-1 token grid."""
    x1, y1, x2, y2 = [max(0, min(count - 1, round(v * (count - 1) / 1000)))
                       for v in box]
    x1, y1 = min(x1, count - 2), min(y1, count - 2)
    return [x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)]


def supervision_masks(targets, positive, location_ids, eos):
    valid = targets != -100
    ends = (targets == eos) & valid
    # Exclude template suffixes after the first assistant EOS.
    valid &= (ends.cumsum(1) - ends.long()) == 0
    locations = torch.isin(targets, torch.as_tensor(location_ids, device=targets.device))
    # Identical semantic target: category + <vis>, or no_event + EOS.
    semantic = valid & ~locations & ~(positive[:, None] & ends)
    coordinates = valid & positive[:, None] & (locations | ends)
    return semantic, coordinates


class Collator:
    def __init__(self, processor, config, arm, training=True):
        self.processor, self.config = processor, config
        self.arm, self.training = arm, training
        self.count = config["model"]["location_tokens"]
        self.location_ids = processor.tokenizer.convert_tokens_to_ids(location_token_strings(self.count))
        self.prompt = config["prompt"]["user"].replace("{categories}", category_block(config))

    def __call__(self, samples):
        conversations, answers = [], []
        for sample in samples:
            messages = [
                {"role": "system", "content": self.config["prompt"]["system"]},
                {"role": "user", "content": [
                    {"type": "image", "image": str(sample.image_path)},
                    {"type": "text", "text": self.prompt},
                ]},
            ]
            if self.training:
                answer = sample.label + VIS if sample.presence else "no_event"
                if sample.presence and self.arm == "token":
                    answer += "".join(f"<loc_{i}>" for i in quantize_box(sample.bbox_1000, self.count))
                answers.append(answer)
            conversations.append(messages)
        inputs = dict(self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", processor_kwargs={
                "padding": True, "size": {"longest_edge": self.config["input"]["max_pixels"],
                                           "shortest_edge": self.config["input"]["min_pixels"]},
            }))
        positive = torch.tensor([bool(s.presence) for s in samples])
        boxes = torch.tensor([s.bbox_1000 or (0, 0, 0, 0) for s in samples]).float() / 1000
        if not self.training:
            return inputs, boxes, positive
        tokenizer = self.processor.tokenizer
        # Process identical prompts first. Append answers with common right-padding
        # so shared prefix positions and dropout tensor shapes also match.
        encoded = [tokenizer.encode(a, add_special_tokens=False) + [tokenizer.eos_token_id]
                   for a in answers]
        lengths = [len(ids) + (4 if s.presence and self.arm == "continuous" else 0)
                   for ids, s in zip(encoded, samples)]
        completion = torch.full((len(samples), max(lengths)), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(completion)
        for row, ids in enumerate(encoded):
            completion[row, :len(ids)] = torch.tensor(ids)
            mask[row, :len(ids)] = 1
        targets = torch.cat((torch.full_like(inputs["input_ids"], -100),
                             completion.masked_fill(mask == 0, -100)), 1)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], completion), 1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], mask), 1)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(completion)), 1)
        semantic, coordinates = supervision_masks(targets, positive, self.location_ids,
                                                   self.processor.tokenizer.eos_token_id)
        if self.arm == "continuous":
            coordinates.zero_()
        # Only materialize vocabulary logits at supervised answer positions.
        positions = (semantic[:, 1:] | coordinates[:, 1:]).any(0).nonzero().flatten()
        inputs["logits_to_keep"] = positions
        return (inputs, boxes, positive, targets[:, positions + 1],
                semantic[:, positions + 1], coordinates[:, positions + 1])


class MatchedQwen(PerceptionQwen):
    def forward(self, inputs):
        ids = inputs["input_ids"]
        self.positions = ((ids == self.vis_id) * torch.arange(ids.shape[1], device=ids.device)).amax(1)
        output = self.vlm(**inputs, use_cache=False)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            raw = self.box_head(self.visual_state.float()).sigmoid()
        self.positions = self.visual_state = None
        boxes = torch.cat((torch.minimum(raw[:, :2], raw[:, 2:]),
                           torch.maximum(raw[:, :2], raw[:, 2:])), -1)
        return output.logits, boxes


def build_model(config, checkpoint=None):
    # This complete initialization path is independent of the requested arm.
    set_seed(config["seed"])
    vlm, processor = load_qwen(config["model"]["path"])
    if checkpoint is None:
        processor.tokenizer.add_special_tokens({"additional_special_tokens":
            [VIS] + location_token_strings(config["model"]["location_tokens"])})
    else:
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    token_ids = processor.tokenizer.convert_tokens_to_ids(
        [VIS] + location_token_strings(config["model"]["location_tokens"]))
    settings = config["train"]
    if checkpoint is None:
        vlm = get_peft_model(vlm, LoraConfig(
            r=settings["lora_r"], lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules=LORA_PATTERNS["projector_llm"],
            trainable_token_indices={"model.language_model.embed_tokens": token_ids,
                                     "lm_head": token_ids},
        ))
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        vlm = PeftModel.from_pretrained(vlm, checkpoint)
    model = MatchedQwen(vlm, token_ids[0], config["model"]["head_dim"])
    if checkpoint is not None:
        model.box_head.load_state_dict(torch.load(Path(checkpoint) / "box_head.pt",
                                                  map_location="cpu", weights_only=True))
    return model.to(config["device"]), processor


def initial_fingerprint(model):
    checksum = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            checksum.update(name.encode())
            checksum.update(str((tuple(param.shape), param.dtype)).encode())
            checksum.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def masked_ce(logits, targets, mask):
    if not mask.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[mask].float(), targets[mask])


def losses(model, batch, config, arm):
    inputs, target, positive, labels, semantic, coordinates = batch
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}
    target, positive, labels, semantic, coordinates = [
        v.to(config["device"]) for v in (target, positive, labels, semantic, coordinates)]
    with torch.autocast(device_type=torch.device(config["device"]).type, dtype=torch.bfloat16):
        logits, boxes = model(inputs)
    language = masked_ce(logits, labels, semantic)
    if arm == "token":
        geometry = masked_ce(logits, labels, coordinates)
    elif positive.any():
        geometry = (F.l1_loss(boxes[positive].float(), target[positive])
                    + generalized_box_iou_loss(boxes[positive].float(), target[positive], reduction="mean"))
    else:
        geometry = boxes.sum() * 0
    return language + geometry, {"semantic": language.item(), "coordinate": geometry.item()}


def train(config, arm):
    settings = config["train"]
    output = output_path(config, arm)
    # A separate output namespace prevents overwriting paper checkpoints/results.
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    samples = read_discovery_samples(config, config["protocol"], "train")
    model, processor = build_model(config)
    manifest = {
        "arm": arm, "seed": config["seed"], "epochs": settings["epochs"],
        "initial_trainable_sha256": initial_fingerprint(model),
        "tokenizer_sha256": digest(processor.tokenizer.get_vocab()),
        "recipe_sha256": digest({k: v for k, v in config.items() if k != "output"}),
        "training_records_sha256": digest([(s.record_uid, s.presence, s.label, s.bbox_1000) for s in samples]),
        "trainable_parameters_instantiated": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inactive_roi_parameters": sum(p.numel() for p in model.box_head.parameters()) if arm == "token" else 0,
        "checkpoint_selection": "fixed_final_epoch", "completed": False,
    }
    sampler = discovery_sampler(samples, settings, config["seed"])
    batches = math.ceil(len(sampler) / settings["batch_size"])
    accumulation = settings["gradient_accumulation"]
    updates = math.ceil(batches / accumulation) * settings["epochs"]
    manifest.update(samples_per_epoch=len(sampler), optimizer_steps=updates)
    save_manifest(output / "matched_manifest.json", manifest)
    other_arm = "continuous" if arm == "token" else "token"
    other_path = output_path(config, other_arm) / "matched_manifest.json"
    if other_path.exists():
        other = json.loads(other_path.read_text())
        checks = ["initial_trainable_sha256", "tokenizer_sha256", "recipe_sha256", "training_records_sha256"]
        different = [key for key in checks if manifest[key] != other[key]]
        if different:
            raise ValueError(f"Initialization/recipe differs from {other_arm}: {different}")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": settings["learning_rate"]},
        {"params": model.box_head.parameters(), "lr": settings["head_learning_rate"]},
    ], weight_decay=settings["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * settings["warmup_ratio"]), updates)
    history, learning_rates = [], []
    set_seed(config["seed"])
    collator = Collator(processor, config, arm)
    for epoch in range(1, settings["epochs"] + 1):
        # Independent sampler generators make order invariant to model RNG use.
        indices = list(discovery_sampler(samples, settings, config["seed"] + epoch - 1))
        loader = DataLoader(samples, batch_size=settings["batch_size"], sampler=indices,
                            collate_fn=collator, num_workers=settings["num_workers"],
                            generator=torch.Generator().manual_seed(config["seed"] + epoch - 1))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started, total = time.perf_counter(), 0.0
        progress = tqdm(loader, desc=f"{arm} train {epoch}/{settings['epochs']}")
        for step, batch in enumerate(progress):
            loss, parts = losses(model, batch, config, arm)
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (loss / group_size).backward()
            total += loss.item()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                learning_rates.append([group["lr"] for group in optimizer.param_groups])
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.3f}", **{k: f"{v:.3f}" for k, v in parts.items()})
        checkpoint = output / f"epoch_{epoch}"
        model.vlm.save_pretrained(checkpoint, save_embedding_layers=False)
        processor.save_pretrained(checkpoint)
        torch.save(model.box_head.state_dict(), checkpoint / "box_head.pt")
        history.append({"epoch": epoch, "train_loss": total / len(loader),
                        "seconds": time.perf_counter() - started,
                        "sample_order_sha256": digest([samples[i].record_uid for i in indices]),
                        "optimizer_steps": len(learning_rates)})
        (output / "history.json").write_text(json.dumps(history, indent=2))
    manifest.update(completed=True, selected_epoch=settings["epochs"],
                    actual_optimizer_steps=len(learning_rates), lr_trace_sha256=digest(learning_rates),
                    epoch_sample_order_sha256=[row["sample_order_sha256"] for row in history])
    save_manifest(output / "matched_manifest.json", manifest)
    print(f"Saved fixed final checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    train(read_config(args.config), args.arm)
