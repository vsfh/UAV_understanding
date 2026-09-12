"""Supervised utility-gated, two-stage active local review for Perception-Qwen.

This is a runnable adaptation of active zoom, NOT a reproduction of ZoomEarth
and NOT GRPO. A frozen, train-only baseline proposes a ROI and image-conditioned
state. A learned budget-conditioned gate selects global-only or global+crop
refinement. Both branches share one VLM/box head and predict ORIGINAL image
coordinates. Prepare is explicit; training never prepares or reads test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CONFIG = "configs/yaml/perception_zoom.yaml"
CACHE_VERSION = 1
VIS = "<vis>"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_digest(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files or not (path / "box_head.pt").is_file():
        raise FileNotFoundError(f"Expected a Perception-Qwen checkpoint: {path}")
    return digest([(str(p.relative_to(path)), file_digest(p)) for p in files])


def read_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def output_path(config):
    return Path(config["output"].format(protocol=config["protocol"], seed=config["seed"]))


def checkpoint_path(config, key="proposal_checkpoint"):
    return Path(config["zoom"][key].format(protocol=config["protocol"], seed=config["seed"]))


def crop_bounds(box, image_size, *, expansion=1.8, min_side=96, min_fraction=0.12):
    """Return a clipped integer PIL crop from a predicted normalized xyxy ROI.

    Invalid/missing/negative-event proposals produce None, hence a forced stop.
    Small *valid* proposals are expanded to a minimum usable field of view.
    """
    width, height = image_size
    if width <= 0 or height <= 0 or box is None or len(box) != 4:
        return None
    values = [float(v) for v in box]
    if not all(math.isfinite(v) for v in values):
        return None
    x0, y0, x1, y1 = [max(0.0, min(1.0, v)) for v in values]
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) * width / 2, (y0 + y1) * height / 2
    cw = min(width, max((x1 - x0) * width * expansion, min_side, width * min_fraction))
    ch = min(height, max((y1 - y0) * height * expansion, min_side, height * min_fraction))
    left = max(0.0, min(width - cw, cx - cw / 2))
    top = max(0.0, min(height - ch, cy - ch / 2))
    bounds = (max(0, math.floor(left)), max(0, math.floor(top)),
              min(width, math.ceil(left + cw)), min(height, math.ceil(top + ch)))
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def crop_to_original(box, bounds, image_size, clamp=True):
    """Map normalized crop xyxy into normalized original-image xyxy."""
    w, h = image_size
    left, top, right, bottom = bounds
    out = [(left + box[0] * (right - left)) / w,
           (top + box[1] * (bottom - top)) / h,
           (left + box[2] * (right - left)) / w,
           (top + box[3] * (bottom - top)) / h]
    return [min(1.0, max(0.0, v)) for v in out] if clamp else out


def original_to_crop(box, bounds, image_size, clamp=False):
    """Inverse mapping; unclamped by default so out-of-crop evidence is retained."""
    w, h = image_size
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise ValueError("Degenerate crop")
    out = [(box[0] * w - left) / (right - left),
           (box[1] * h - top) / (bottom - top),
           (box[2] * w - left) / (right - left),
           (box[3] * h - top) / (bottom - top)]
    return [min(1.0, max(0.0, v)) for v in out] if clamp else out


def inference_sample(sample):
    """Erase targets at the inference boundary, including cache preparation."""
    return SimpleNamespace(record_uid=sample.record_uid, image_path=sample.image_path)


def gate_metadata(prediction, bounds, image_size, review_cost):
    """Image/prediction-only scalar inputs, deliberately no sample/target argument."""
    score = float(prediction.get("presence_score", 0.0))
    score = max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
    if bounds is None:
        xyxy = [0.0] * 4
    else:
        w, h = image_size
        xyxy = [bounds[0] / w, bounds[1] / h, bounds[2] / w, bounds[3] / h]
    area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
    return [score, *xyxy, area, float(bounds is not None), float(review_cost)]


def should_review(probability, valid_crop, threshold=0.5):
    return bool(valid_crop and math.isfinite(float(probability)) and probability >= threshold)


def sample_signature(samples):
    # Image membership and file metadata only: no target labels or target boxes.
    rows = []
    for sample in samples:
        path = Path(sample.image_path).resolve()
        info = path.stat()
        rows.append((sample.record_uid, str(path), info.st_size, info.st_mtime_ns))
    return digest(rows)


def source_config(config):
    path = checkpoint_path(config).parent / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Proposal training provenance is required: {path}")
    source = read_config(path)
    if source["protocol"] != config["protocol"]:
        raise ValueError("Proposal checkpoint protocol differs; create a matching baseline first")
    if source.get("experiment") != "perception_qwen":
        raise ValueError("Use a baseline perception_qwen proposal checkpoint, never an RL/zoom/test-fitted model")
    if source["data"] != config["data"]:
        raise ValueError("Proposal and zoom data configuration differ; audited matching splits are required")
    source["device"] = config["device"]
    source["test"] = config["test"]
    return source


def cache_metadata(config, split, samples, source_hash=None):
    source = source_config(config)
    # Device/batch size are runtime choices, not proposal semantics.
    semantic_source = {k: v for k, v in source.items() if k not in {"device", "test"}}
    return {"version": CACHE_VERSION, "split": split, "protocol": config["protocol"],
            "source_checkpoint": str(checkpoint_path(config).resolve()),
            "source_checkpoint_sha256": source_hash or checkpoint_digest(checkpoint_path(config)),
            "source_config_sha256": digest(semantic_source), "split_sha256": sample_signature(samples),
            "count": len(samples), "targets_in_cache": False,
            "feature_source": "frozen_baseline_last_layer_predicted_vis_or_last_prompt_vis",
            "proposal_policy": "constrained_greedy_category_and_continuous_box; no target access"}


def cache_path(config, split):
    return output_path(config) / "proposal_cache" / f"{split}.pt"


def load_cache(config, split, samples):
    import torch
    path = cache_path(config, split)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run the explicit prepare mode for {split}")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    expected = cache_metadata(config, split, samples)
    if cache["metadata"] != expected:
        changed = [k for k in expected if cache["metadata"].get(k) != expected[k]]
        raise ValueError(f"Stale/incompatible {split} cache: {changed}; prepare it again explicitly")
    keys = [s.record_uid for s in samples]
    if len(set(keys)) != len(keys) or set(keys) != set(cache["records"]):
        raise ValueError("Cache sample IDs are missing, repeated, or belong to another split")
    if not all(torch.isfinite(row["feature"]).all() for row in cache["records"].values()):
        raise ValueError("Non-finite proposal features")
    return cache


def samples_for(config, split):
    from clear_uav.table4 import read_discovery_samples
    samples = read_discovery_samples(config, config["protocol"], split)
    maximum = config.get("limits", {}).get(split)
    if maximum is not None:
        samples = samples[:maximum]
    if not samples:
        raise ValueError(f"Empty {split} split")
    return samples


def require_disjoint(train_samples, val_samples):
    train_paths = {str(Path(s.image_path).resolve()) for s in train_samples}
    val_paths = {str(Path(s.image_path).resolve()) for s in val_samples}
    if train_paths & val_paths:
        raise ValueError("Train and validation image memberships overlap")


def dependencies():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_model(config, checkpoint=None, training=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor
    from clear_uav.modeling import LORA_PATTERNS
    from perception_extension_runtime import load_qwen, build_baseline
    from train_perception_qwen import PerceptionQwen
    if checkpoint is not None and not training:
        return build_baseline(config, checkpoint)
    if checkpoint is None:
        vlm, processor = load_qwen(config["model"]["path"], disable_mmap=config["model"].get("disable_mmap", True))
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [VIS]})
        processor.tokenizer.padding_side = "left"
        vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        vis_id = processor.tokenizer.convert_tokens_to_ids(VIS)
        settings = config["train"]
        vlm = get_peft_model(vlm, LoraConfig(
            r=settings["lora_r"], lora_alpha=settings["lora_alpha"], lora_dropout=settings["lora_dropout"],
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_PATTERNS["projector_llm"],
            trainable_token_indices={"model.language_model.embed_tokens": [vis_id], "lm_head": [vis_id]},
        ))
        model = PerceptionQwen(vlm, vis_id, config["model"]["head_dim"]).to(config["device"])
    else:
        vlm, _ = load_qwen(config["model"]["path"], disable_mmap=config["model"].get("disable_mmap", True))
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        vlm = PeftModel.from_pretrained(vlm, checkpoint, is_trainable=training)
        vis_id = processor.tokenizer.convert_tokens_to_ids(VIS)
        if vis_id == processor.tokenizer.unk_token_id:
            raise ValueError("The initialization checkpoint has no <vis> token")
        model = PerceptionQwen(vlm, vis_id, config["model"]["head_dim"])
        model.box_head.load_state_dict(torch.load(Path(checkpoint) / "box_head.pt", map_location="cpu", weights_only=True))
        model = model.to(config["device"])
    if training:
        model.vlm.enable_input_require_grads()
        model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not any(p.requires_grad for p in model.vlm.parameters()):
            raise RuntimeError("Warm-start adapter is frozen")
    else:
        model.requires_grad_(False)
        model.eval()
    return model, processor


def make_gate(feature_dim, config):
    import torch
    from torch import nn
    hidden = config["zoom"]["gate_hidden_dim"]
    class UtilityGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_norm = nn.LayerNorm(feature_dim)
            self.layers = nn.Sequential(nn.Linear(feature_dim + 8, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        def forward(self, inputs):
            # Preserve the physical 0..1 geometry/probability/cost scales instead
            # of normalizing them jointly with thousands of visual features.
            inputs = torch.cat((self.visual_norm(inputs[:, :feature_dim]), inputs[:, feature_dim:]), dim=-1)
            return self.layers(inputs)
    return UtilityGate().to(config["device"])


def proposal_bounds(record, config):
    prediction = record["prediction"]
    if not prediction.get("valid") or prediction.get("category") is None:
        return None
    box = prediction.get("bbox_1000")
    box = [float(x) / 1000 for x in box] if box is not None else None
    settings = config["zoom"]
    return crop_bounds(box, record["image_size"], expansion=settings["crop_expansion"],
                       min_side=settings["min_crop_side"], min_fraction=settings["min_crop_fraction"])


def gate_input(record, bounds, cost, device):
    import torch
    extra = gate_metadata(record["prediction"], bounds, record["image_size"], cost)
    return torch.cat((record["feature"].float(), torch.tensor(extra))).to(device).unsqueeze(0)


class ZoomCollator:
    def __init__(self, processor, config, training=False):
        from clear_uav.table4 import category_block
        self.processor, self.config, self.training = processor, config, training
        self.prompt = config["prompt"]["user"].replace("{categories}", category_block(config))

    def __call__(self, samples, records=None, use_crop=False):
        import torch
        from PIL import Image
        from clear_uav.modeling import assistant_only_labels
        conversations = []
        for i, sample in enumerate(samples):
            with Image.open(sample.image_path) as handle:
                full = handle.convert("RGB")
            content = [{"type": "text", "text": "Image 1 is the full original UAV image."},
                       {"type": "image", "image": full}]
            if use_crop:
                bounds = proposal_bounds(records[i], self.config)
                if bounds is None:
                    raise ValueError("A crop branch requires a valid prediction-only proposal")
                normalized = [round(v, 6) for v in crop_to_original([0, 0, 1, 1], bounds, full.size)]
                content.extend([
                    {"type": "text", "text": f"Image 2 is a high-resolution crop from image 1 at normalized xyxy {normalized}."},
                    {"type": "image", "image": full.crop(bounds)},
                ])
            content.append({"type": "text", "text": self.prompt +
                            " The <vis> region always uses the coordinates of the FULL ORIGINAL image 1."})
            messages = [{"role": "system", "content": self.config["prompt"]["system"]},
                        {"role": "user", "content": content}]
            if self.training:
                answer = sample.label + VIS if sample.presence else "no_event"
                messages.append({"role": "assistant", "content": answer})
            conversations.append(messages)
        encoded = dict(self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=not self.training,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True, "size": {
                "longest_edge": self.config["input"]["max_pixels"],
                "shortest_edge": self.config["input"]["min_pixels"]}}))
        if not self.training:
            return encoded
        encoded["labels"] = assistant_only_labels(encoded["input_ids"], encoded["attention_mask"], self.processor.tokenizer)
        target = torch.tensor([s.bbox_1000 or (0, 0, 0, 0) for s in samples], dtype=torch.float32) / 1000
        positive = torch.tensor([bool(s.presence) for s in samples], dtype=torch.bool)
        return encoded, target, positive


def forward_losses(model, batch, config):
    """Per-example losses make gate utility targets valid even for batch > 1."""
    import torch
    import torch.nn.functional as F
    from torchvision.ops import generalized_box_iou_loss
    inputs, target, positive = batch
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}
    target, positive = target.to(config["device"]), positive.to(config["device"])
    labels = inputs.pop("labels")
    ids = inputs["input_ids"]
    model.positions = ((ids == model.vis_id) * torch.arange(ids.shape[1], device=ids.device)).amax(1)
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda"):
        output = model.vlm(**inputs, use_cache=False)
        state = model.visual_state
    raw = model.box_head(state.float()).sigmoid()
    boxes = torch.cat((torch.minimum(raw[:, :2], raw[:, 2:]), torch.maximum(raw[:, :2], raw[:, 2:])), dim=-1)
    model.positions = model.visual_state = None
    mask = labels[:, 1:] != -100
    selected = output.logits[:, :-1][mask].float()
    ce = F.cross_entropy(selected, labels[:, 1:][mask], reduction="none")
    batch_ids = torch.arange(len(labels), device=ids.device).unsqueeze(1).expand_as(mask)[mask]
    language = torch.zeros(len(labels), device=ids.device).scatter_add(0, batch_ids, ce)
    language = language / mask.sum(1).clamp_min(1)
    l1 = boxes.sum(1) * 0
    giou = boxes.sum(1) * 0
    if positive.any():
        l1 = l1.index_copy(0, positive.nonzero().flatten(), F.l1_loss(boxes[positive], target[positive], reduction="none").mean(1))
        giou = giou.index_copy(0, positive.nonzero().flatten(), generalized_box_iou_loss(boxes[positive], target[positive], reduction="none"))
    parts = {"language": language, "l1": l1, "giou": giou}
    total = sum(config["loss"][k] * value for k, value in parts.items())
    return total, {k: v.detach() for k, v in parts.items()}


def predict_branch(model, processor, samples, config, labels, records=None, use_crop=False, capture_features=False):
    """Inference consumes only record_uid/image_path, never sample targets."""
    import torch
    from clear_uav.generation_constraints import label_prefix_allowed_tokens
    from clear_uav.table4 import event_probability
    model.eval()
    samples = [inference_sample(s) for s in samples]
    collator = ZoomCollator(processor, config, training=False)
    tokenizer = processor.tokenizer
    answers = [label + VIS for label in labels] + ["no_event"]
    max_tokens = max(len(tokenizer.encode(a, add_special_tokens=False)) for a in answers) + 1
    device = torch.device(config["device"])
    captured = []
    hook = None
    if capture_features:
        hook = model.box_head[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().half()))
    try:
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            inputs = collator(samples, records, use_crop)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            length = inputs["input_ids"].shape[1]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                generated = model.vlm.generate(
                    **inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                    prefix_allowed_tokens_fn=label_prefix_allowed_tokens(tokenizer, answers, prompt_length=length),
                    return_dict_in_generate=True, output_scores=True,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            tokens = generated.sequences[:, length:]
            categories = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            completion_mask = (tokens == tokenizer.eos_token_id).cumsum(1)
            completion_mask = (completion_mask - (tokens == tokenizer.eos_token_id).long()) == 0
            box_inputs = inputs | {"input_ids": generated.sequences,
                                   "attention_mask": torch.cat((inputs["attention_mask"], completion_mask.long()), dim=1),
                                   "logits_to_keep": 1}
            if "mm_token_type_ids" in inputs:
                box_inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tokens)), dim=1)
            first_scores = generated.scores[0]
            del generated
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, boxes = model(box_inputs)
            boxes = (boxes.float().clamp(0, 1) * 1000).cpu().tolist()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency = (time.perf_counter() - started) * 1000 / len(samples)
            predictions = []
            for i, text in enumerate(categories):
                category = text.strip()
                positive = category in labels
                valid = (positive and bool((tokens[i] == model.vis_id).any())) or category == "no_event"
                predictions.append({"category": category if positive else None,
                                    "bbox_1000": boxes[i] if positive else None,
                                    "presence_score": event_probability(first_scores[i], tokenizer, labels),
                                    "valid": bool(valid), "latency_ms": latency,
                                    "raw_output": tokenizer.decode(tokens[i], skip_special_tokens=False),
                                    "num_calls": 2, "inference_batch_size": len(samples),
                                    "visual_input_tokens": int(inputs.get("image_grid_thw", torch.zeros(1)).prod(-1).sum().item()
                                                               / processor.image_processor.merge_size ** 2),
                                    "timing_scope": "preprocess_generate_box_replay"})
            return predictions, captured[0] if capture_features else None
    finally:
        if hook is not None:
            hook.remove()


def prepare(config, split):
    import torch
    from PIL import Image
    from tqdm import tqdm
    from clear_uav.table4 import definitions_from_config
    source = source_config(config)
    model, processor = build_model(source, checkpoint_path(config), training=False)
    labels, _ = definitions_from_config(source)
    source_hash = checkpoint_digest(checkpoint_path(config))
    splits = ["train", "val"] if split == "train_val" else [split]
    for part in splits:
        samples = samples_for(config, part)
        records = {}
        # Use the ACTUAL baseline prompt/collator for proposals, matching its checkpoint.
        from test_perception_qwen import predict as baseline_predict
        for sample in tqdm(samples, desc=f"frozen proposals {part}"):
            safe = SimpleNamespace(image_path=sample.image_path, presence=False, bbox_1000=None, label=None)
            states = []
            visual_tokens = []
            hook = model.box_head[0].register_forward_pre_hook(lambda module, args: states.append(args[0].detach().cpu().half()))
            def capture_visual_tokens(module, args, kwargs):
                grid = kwargs.get("image_grid_thw")
                if grid is not None:
                    visual_tokens.append(int(grid.prod(-1).sum().item() / processor.image_processor.merge_size ** 2))
            visual_hook = model.vlm.register_forward_pre_hook(capture_visual_tokens, with_kwargs=True)
            try:
                with torch.inference_mode():
                    prediction = baseline_predict(model, processor, [safe], source, labels)[0]
            finally:
                hook.remove()
                visual_hook.remove()
            prediction["visual_input_tokens"] = max(visual_tokens, default=0)
            with Image.open(sample.image_path) as handle:
                size = list(handle.size)
            records[sample.record_uid] = {"prediction": prediction, "feature": states[-1][0], "image_size": size}
        metadata = cache_metadata(config, part, samples, source_hash)
        path = cache_path(config, part)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"metadata": metadata, "records": records}, temporary)
        temporary.replace(path)
        path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Prepared {part}: {len(samples)} records at {path}", flush=True)


def run_epoch(model, gate, processor, samples, cache, config, *, training=False, optimizer=None, scheduler=None, epoch=0):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from clear_uav.table4 import discovery_sampler
    model.train(training)
    gate.train(training)
    collator = ZoomCollator(processor, config, training=True)
    settings, zoom = config["train"], config["zoom"]
    order = list(discovery_sampler(samples, settings, config["seed"] + epoch)) if training else list(range(len(samples)))
    batch_size = settings["batch_size"]
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    accumulation = settings["gradient_accumulation"]
    totals = {"global_loss": 0.0, "crop_loss": 0.0, "gate_loss": 0.0,
              "policy_loss": 0.0, "review_rate": 0.0, "utility_positive_rate": 0.0}
    count = crop_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, indices in enumerate(tqdm(batches, desc="train zoom" if training else "validate zoom")):
        current = [samples[i] for i in indices]
        records = [cache["records"][s.record_uid] for s in current]
        bounds = [proposal_bounds(row, config) for row in records]
        valid_indices = [i for i, bound in enumerate(bounds) if bound is not None]
        costs = [random.uniform(*zoom["train_review_cost_range"]) if training else zoom["review_cost"] for _ in current]
        group_size = min(accumulation, len(batches) - (step // accumulation) * accumulation)
        scale = group_size if training else 1
        with torch.set_grad_enabled(training):
            full, _ = forward_losses(model, collator(current), config)
            if not torch.isfinite(full).all():
                raise FloatingPointError("Non-finite global loss; optimizer was not updated")
            full_detached = full.detach()
            if training:
                (zoom["global_loss_weight"] * full.mean() / scale).backward()
            # The branches are backpropagated separately to release 8B activations.
            del full
            crop_detached = full_detached.clone()
            if valid_indices:
                cropped, _ = forward_losses(model, collator([current[i] for i in valid_indices],
                                                            [records[i] for i in valid_indices], True), config)
                if not torch.isfinite(cropped).all():
                    raise FloatingPointError("Non-finite crop loss; optimizer was not updated")
                crop_detached[valid_indices] = cropped.detach()
                if training:
                    (zoom["crop_loss_weight"] * cropped.sum() / len(current) / scale).backward()
                crop_count += len(valid_indices)
                totals["crop_loss"] += cropped.detach().sum().item()
                del cropped
            features = torch.cat([gate_input(row, bound, cost, config["device"])
                                  for row, bound, cost in zip(records, bounds, costs)], dim=0)
            logits = gate(features).squeeze(-1)
            valid = torch.tensor([b is not None for b in bounds], device=config["device"], dtype=torch.bool)
            cost_tensor = torch.tensor(costs, device=config["device"])
            gain = full_detached - crop_detached - cost_tensor
            targets = ((gain > zoom["utility_margin"]) & valid).float()
            gate_loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(gate_loss):
                raise FloatingPointError("Non-finite gate loss; optimizer was not updated")
            if training:
                (zoom["gate_loss_weight"] * gate_loss / scale).backward()
        decisions = (logits.detach().sigmoid() >= zoom["gate_threshold"]) & valid
        policy = torch.where(decisions, crop_detached + cost_tensor, full_detached)
        n = len(current)
        totals["global_loss"] += full_detached.sum().item()
        totals["gate_loss"] += gate_loss.item() * n
        totals["policy_loss"] += policy.sum().item()
        totals["review_rate"] += decisions.sum().item()
        totals["utility_positive_rate"] += targets.sum().item()
        count += n
        if training and ((step + 1) % accumulation == 0 or step + 1 == len(batches)):
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(gate.parameters()), settings["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return {k: value / max(1, crop_count if k == "crop_loss" else count) for k, value in totals.items()}


def train(config):
    import torch
    import yaml
    from transformers import get_cosine_schedule_with_warmup, set_seed
    from clear_uav.table4 import discovery_sampler
    output = output_path(config)
    if any((output / name).exists() for name in ("best", "config.yaml", "history.json")):
        raise FileExistsError(f"Refusing to overwrite an existing zoom training run: {output}; use a new --output")
    set_seed(config["seed"])
    samples, val_samples = samples_for(config, "train"), samples_for(config, "val")
    require_disjoint(samples, val_samples)
    train_cache, val_cache = load_cache(config, "train", samples), load_cache(config, "val", val_samples)
    feature_dim = len(next(iter(train_cache["records"].values()))["feature"])
    model, processor = build_model(config, checkpoint_path(config, "initial_checkpoint"), training=True)
    gate = make_gate(feature_dim, config)
    settings = config["train"]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": settings["learning_rate"]},
        {"params": model.box_head.parameters(), "lr": settings["head_learning_rate"]},
        {"params": gate.parameters(), "lr": config["zoom"]["gate_learning_rate"]},
    ], weight_decay=settings["weight_decay"])
    length = len(discovery_sampler(samples, settings, config["seed"]))
    updates = math.ceil(math.ceil(length / settings["batch_size"]) / settings["gradient_accumulation"]) * settings["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * settings["warmup_ratio"]), max(1, updates))
    output.mkdir(parents=True, exist_ok=True)
    config["zoom"]["gate_feature_dim"] = feature_dim
    config["zoom"]["proposal_sha256"] = train_cache["metadata"]["source_checkpoint_sha256"]
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    history, best = [], float("inf")
    for epoch in range(1, settings["epochs"] + 1):
        train_values = run_epoch(model, gate, processor, samples, train_cache, config,
                                 training=True, optimizer=optimizer, scheduler=scheduler, epoch=epoch - 1)
        val_values = run_epoch(model, gate, processor, val_samples, val_cache, config)
        history.append({"epoch": epoch, "train": train_values, "val": val_values})
        if val_values["policy_loss"] < best:
            best = val_values["policy_loss"]
            checkpoint = output / "best"
            model.vlm.save_pretrained(checkpoint, save_embedding_layers=False)
            processor.save_pretrained(checkpoint)
            torch.save(model.box_head.state_dict(), checkpoint / "box_head.pt")
            torch.save({"state_dict": gate.state_dict(), "feature_dim": feature_dim}, checkpoint / "zoom_gate.pt")
            (checkpoint / "zoom_manifest.json").write_text(json.dumps({
                "method": "supervised_utility_gate", "box_frame": "original_image_xyxy_0_1",
                "epoch": epoch, "selection": "validation_policy_loss", "val_policy_loss": best,
                "train_cache": train_cache["metadata"], "val_cache": val_cache["metadata"]}, indent=2), encoding="utf-8")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)


def evaluation_config(runtime):
    saved = read_config(output_path(runtime) / "config.yaml")
    saved["device"], saved["test"] = runtime["device"], runtime["test"]
    # Limits are invocation-local: a tiny training run must not silently force
    # every later evaluation to remain tiny. all-mode passes the same explicit
    # CLI limits to each stage, so its prepared cache still matches exactly.
    # Model/prompt/gate/proposal/data settings come from the checkpoint run.
    saved["output"] = runtime["output"]
    saved["training_limits"] = dict(saved.get("limits", {}))
    saved["limits"] = dict(runtime.get("limits", {}))
    return saved


def validate_calibration(calibration, config, checkpoint, fingerprint, test_limit):
    if calibration.get("checkpoint") != str(Path(checkpoint).resolve()):
        raise ValueError("Calibration checkpoint path differs; rerun validation")
    if calibration.get("checkpoint_sha256") != fingerprint:
        raise ValueError("Calibration checkpoint content differs; rerun validation")
    if calibration.get("protocol") != config["protocol"] or calibration.get("seed") != config["seed"]:
        raise ValueError("Calibration belongs to a different protocol or seed")
    if calibration.get("proposal_sha256") != config["zoom"]["proposal_sha256"]:
        raise ValueError("Calibration proposal source differs; rerun validation")
    if calibration.get("limited_run") and test_limit is None and not config["data"].get("max_samples"):
        raise ValueError("Refusing full test with tiny-run calibration; prepare full val proposals and run full validation first")


def final_output_fpr(samples, predictions, threshold):
    negatives = [(s, p) for s, p in zip(samples, predictions) if not s.presence]
    if not negatives:
        return None
    def emitted(pred):
        box = pred.get("bbox_1000")
        valid_box = (box is not None and len(box) == 4 and all(math.isfinite(float(x)) for x in box)
                     and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
        return bool(pred.get("category") is not None and pred.get("valid") and valid_box
                    and pred["presence_score"] >= threshold)
    return sum(emitted(p) for _, p in negatives) / len(negatives)


def evaluate(runtime, split):
    import torch
    from tqdm import tqdm
    from clear_uav.table4 import definitions_from_config, save_results, select_threshold, table4_metrics
    config = evaluation_config(runtime)
    output = output_path(config)
    checkpoint = output / "best"
    fingerprint = checkpoint_digest(checkpoint)
    calibration_path = output / "calibration.json"
    if split == "test":
        if not calibration_path.is_file():
            raise FileNotFoundError("Run validation first; test never chooses a threshold")
        validate_calibration(json.loads(calibration_path.read_text(encoding="utf-8")), config,
                             checkpoint, fingerprint, config.get("limits", {}).get("test"))
    model, processor = build_model(config, checkpoint, training=False)
    payload = torch.load(output / "best" / "zoom_gate.pt", map_location="cpu", weights_only=True)
    gate = make_gate(payload["feature_dim"], config)
    gate.load_state_dict(payload["state_dict"])
    gate.eval().requires_grad_(False)
    labels, _ = definitions_from_config(config)
    for part in (["val", "test"] if split == "all" else [split]):
        if part == "test":
            if not calibration_path.is_file():
                raise FileNotFoundError("Run validation first; test never chooses a threshold")
            validate_calibration(json.loads(calibration_path.read_text(encoding="utf-8")), config,
                                 checkpoint, fingerprint, config.get("limits", {}).get("test"))
        samples = samples_for(config, part)
        cache = load_cache(config, part, samples)
        if cache["metadata"]["source_checkpoint_sha256"] != config["zoom"]["proposal_sha256"]:
            raise ValueError("The proposal source changed after gate training")
        predictions = []
        for sample in tqdm(samples, desc=f"active review {part}"):
            row = cache["records"][sample.record_uid]
            bounds = proposal_bounds(row, config)
            with torch.inference_mode():
                probability = gate(gate_input(row, bounds, config["zoom"]["review_cost"], config["device"])).sigmoid().item()
            review = should_review(probability, bounds is not None, config["zoom"]["gate_threshold"])
            current, _ = predict_branch(model, processor, [sample], config, labels, [row], review)
            prediction = current[0]
            prediction.update({"reviewed": review, "gate_probability": probability,
                               "crop_bounds_pixels": list(bounds) if review else None,
                               "bbox_coordinate_frame": "original_image_0_1000",
                               "proposal_latency_ms": row["prediction"]["latency_ms"],
                               "refinement_latency_ms": prediction["latency_ms"],
                               "refinement_visual_input_tokens": prediction["visual_input_tokens"],
                               "proposal_visual_input_tokens": row["prediction"].get("visual_input_tokens", 0),
                               "visual_input_tokens": prediction["visual_input_tokens"] + row["prediction"].get("visual_input_tokens", 0),
                               "num_calls": row["prediction"].get("num_calls", 2) + 2,
                               "num_image_views": 3 if review else 2,
                               "latency_ms": prediction["latency_ms"] + row["prediction"]["latency_ms"],
                               "timing_scope": "cached_proposal_measured_time_plus_live_selected_refinement;gate_and_cache_io_excluded"})
            predictions.append(prediction)
        if part == "val":
            threshold, selection = select_threshold(samples, predictions, config["validation"])
            calibration_path.write_text(json.dumps({"threshold": threshold, "selection": selection,
                "split": "val", "split_sha256": cache["metadata"]["split_sha256"],
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint,
                "protocol": config["protocol"], "seed": config["seed"],
                "proposal_sha256": config["zoom"]["proposal_sha256"],
                "sample_count": len(samples), "sample_limit": config.get("limits", {}).get("val"),
                "limited_run": config.get("limits", {}).get("val") is not None or bool(config["data"].get("max_samples")),
                "policy": "max_recall_at_fpr", "max_n_fpr": config["validation"]["max_n_fpr"]}, indent=2), encoding="utf-8")
        else:
            if not calibration_path.is_file():
                raise FileNotFoundError("Run validation first; test never chooses a threshold")
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            validate_calibration(calibration, config, checkpoint, fingerprint, config.get("limits", {}).get("test"))
            threshold = calibration["threshold"]
        metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
        metrics["final_output_n_fpr"] = final_output_fpr(samples, predictions, threshold)
        metrics["zoom_review_rate"] = sum(p["reviewed"] for p in predictions) / len(predictions)
        metrics["zoom_mean_image_views"] = sum(p["num_image_views"] for p in predictions) / len(predictions)
        config["train"]["seeds"] = [config["seed"]]
        save_results(output / f"{part}_results.json", config, config["protocol"], samples, predictions, metrics, seed=config["seed"])
        # Existing writer may select prediction columns; preserve all active-policy telemetry.
        (output / f"{part}_zoom_predictions.json").write_text(json.dumps([
            {"record_uid": s.record_uid, **p} for s, p in zip(samples, predictions)], indent=2), encoding="utf-8")
        print(json.dumps({"split": part, **metrics}, indent=2), flush=True)


def main(default_mode="train"):
    dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], default=default_mode)
    parser.add_argument("--prepare-split", choices=["train", "val", "train_val", "test"], default="train_val")
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = read_config(args.config)
    for part in ["train", "val", "test"]:
        value = getattr(args, f"max_{part}_samples")
        if value is not None:
            if value < 1:
                parser.error("sample limits must be positive")
            config.setdefault("limits", {})[part] = value
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("epochs must be positive")
        config["train"]["epochs"] = args.epochs
    if args.output:
        config["output"] = args.output
    if output_path(config).resolve() == checkpoint_path(config).parent.resolve():
        parser.error("zoom output must be independent of the baseline proposal run")
    if args.mode == "prepare":
        # Test preparation is explicit and uses saved training provenance if available.
        if args.prepare_split == "test" and (output_path(config) / "config.yaml").is_file():
            config = evaluation_config(config)
        prepare(config, args.prepare_split)
    elif args.mode == "train":
        train(config)
    else:
        evaluate(config, args.split)


if __name__ == "__main__":
    main()
