"""Full Qwen3-VL + a PerceptionGPT-style continuous ROI decoder.

This is an adaptation, not a full PerceptionGPT reproduction: no mask branch,
box input encoder, or extra ViT layer fusion. Categories remain text outputs.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import generalized_box_iou_loss
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clear_uav.modeling import LORA_PATTERNS, assistant_only_labels, load_qwen
from clear_uav.table4 import category_block, discovery_sampler, read_discovery_samples


VIS = "<vis>"
DEFAULT_CONFIG = "configs/yaml/perception_qwen.yaml"


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def output_path(config):
    return Path(config["output"].format(protocol=config["protocol"], seed=config["seed"]))


class Collator:
    def __init__(self, processor, config, training=True):
        self.processor = processor
        self.config = config
        self.training = training
        self.prompt = config["prompt"]["user"].replace("{categories}", category_block(config))

    def __call__(self, samples):
        conversations = []
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
                messages.append({"role": "assistant", "content": answer})
            conversations.append(messages)
        encoded = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=not self.training,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "size": {
                "longest_edge": self.config["input"]["max_pixels"],
                "shortest_edge": self.config["input"]["min_pixels"],
            }},
        )
        if self.training:
            encoded["labels"] = assistant_only_labels(
                encoded["input_ids"], encoded["attention_mask"], self.processor.tokenizer
            )
        positive = torch.tensor([s.presence for s in samples])
        boxes = torch.tensor([s.bbox_1000 or (0, 0, 0, 0) for s in samples]).float() / 1000
        return dict(encoded), boxes, positive


class PerceptionQwen(nn.Module):
    def __init__(self, vlm, vis_id, head_dim):
        super().__init__()
        self.vlm = vlm
        self.vis_id = vis_id
        hidden_dim = vlm.config.text_config.hidden_size
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(),
            nn.Linear(head_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 4),
        )
        self.positions = None
        self.visual_state = None
        # Capture only the last multimodal layer, not every layer's hidden states.
        vlm.get_base_model().model.language_model.register_forward_hook(self.capture)

    def capture(self, module, args, output):
        if self.positions is not None:
            hidden = output[0]
            self.visual_state = hidden[torch.arange(hidden.shape[0], device=hidden.device), self.positions]

    def forward(self, inputs):
        ids = inputs["input_ids"]
        # Select the assistant's LAST <vis>; the instruction also mentions <vis>.
        self.positions = ((ids == self.vis_id) * torch.arange(ids.shape[1], device=ids.device)).amax(1)
        output = self.vlm(**inputs, use_cache=False)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            raw = self.box_head(self.visual_state.float()).sigmoid()
        self.positions = self.visual_state = None
        # Ordered xyxy in [0,1]; losses operate in original-image coordinates.
        lo = torch.minimum(raw[:, :2], raw[:, 2:])
        hi = torch.maximum(raw[:, :2], raw[:, 2:])
        return output.loss, torch.cat((lo, hi), dim=-1)


def build_model(config, checkpoint=None):
    vlm, processor = load_qwen(config["model"]["path"])
    if checkpoint is None:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [VIS]})
    else:
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    vis_id = processor.tokenizer.convert_tokens_to_ids(VIS)
    train = config["train"]
    if checkpoint is None:
        vlm = get_peft_model(vlm, LoraConfig(
            r=train["lora_r"], lora_alpha=train["lora_alpha"],
            lora_dropout=train["lora_dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules=LORA_PATTERNS["projector_llm"],
            trainable_token_indices={
                "model.language_model.embed_tokens": [vis_id], "lm_head": [vis_id],
            },
        ))
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        vlm = PeftModel.from_pretrained(vlm, checkpoint)
    model = PerceptionQwen(vlm, vis_id, config["model"]["head_dim"])
    if checkpoint is not None:
        model.box_head.load_state_dict(torch.load(Path(checkpoint) / "box_head.pt", map_location="cpu", weights_only=True))
    return model.to(config["device"]), processor


def losses(model, batch, config):
    inputs, target, positive = batch
    device = config["device"]
    inputs = {key: value.to(device) for key, value in inputs.items()}
    target, positive = target.to(device), positive.to(device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        language, boxes = model(inputs)
    l1 = giou = boxes.sum() * 0
    if positive.any():
        l1 = F.l1_loss(boxes[positive].float(), target[positive])
        giou = generalized_box_iou_loss(boxes[positive].float(), target[positive], reduction="mean")
    parts = {"language": language, "l1": l1, "giou": giou}
    total = sum(config["loss"][name] * value for name, value in parts.items())
    return total, {name: value.detach().item() for name, value in parts.items()}


@torch.no_grad()
def validation_loss(model, loader, config):
    model.eval()
    total = count = 0
    for batch in tqdm(loader, desc="validation loss"):
        loss, _ = losses(model, batch, config)
        size = len(batch[2])
        total += loss.item() * size
        count += size
    return total / count


def train(config):
    set_seed(config["seed"])
    settings = config["train"]
    samples = read_discovery_samples(config, config["protocol"], "train")
    val_samples = read_discovery_samples(config, config["protocol"], "val")
    model, processor = build_model(config)
    collator = Collator(processor, config)
    loader = DataLoader(samples, batch_size=settings["batch_size"],
                        sampler=discovery_sampler(samples, settings, config["seed"]),
                        collate_fn=collator, num_workers=settings["num_workers"])
    val_loader = DataLoader(val_samples, batch_size=settings["batch_size"],
                            collate_fn=collator, num_workers=settings["num_workers"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": settings["learning_rate"]},
        {"params": model.box_head.parameters(), "lr": settings["head_learning_rate"]},
    ], weight_decay=settings["weight_decay"])
    accumulation = settings["gradient_accumulation"]
    updates = math.ceil(len(loader) / accumulation) * settings["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * settings["warmup_ratio"]), updates)
    output = output_path(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    best, history = float("inf"), []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"train {epoch}/{settings['epochs']}")
        total = 0
        for step, batch in enumerate(progress):
            loss, parts = losses(model, batch, config)
            # Correctly normalize the final, possibly shorter accumulation group.
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (loss / group_size).backward()
            total += loss.item()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.3f}", **{k: f"{v:.3f}" for k, v in parts.items()})
        val = validation_loss(model, val_loader, config)
        history.append({"epoch": epoch, "train_loss": total / len(loader), "val_loss": val})
        if val < best:
            best = val
            checkpoint = output / "best"
            model.vlm.save_pretrained(checkpoint, save_embedding_layers=False)
            processor.save_pretrained(checkpoint)
            torch.save(model.box_head.state_dict(), checkpoint / "box_head.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch}: val_loss={val:.4f}, best={best:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    train(read_config(parser.parse_args().config))
