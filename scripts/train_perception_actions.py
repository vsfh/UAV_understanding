"""Warmstart Perception; train GT action imitation, then action-only GRPO.

GRPO freezes the VLM and the initial continuous box head. It samples only the
explicit discrete refinement policy, with a fixed SFT reference and refreshed
old-policy snapshots. It does not train text generation or rejection decisions.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from perception_actions_core import (ActionPolicy, aligned_iou, grpo_loss,
    group_advantages, imitation_loss, rollout, trajectory_reward, update_rollout_buffer)

DEFAULT_CONFIG = "configs/yaml/perception_actions.yaml"


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def formatted(config, value):
    return Path(str(value).format(protocol=config["protocol"], seed=config["seed"]))


def output_path(config, stage):
    return formatted(config, config["output"]) / stage


class ActionsPerception(nn.Module):
    def __init__(self, baseline, config):
        super().__init__()
        self.baseline = baseline
        self.config = config
        self.action_policy = ActionPolicy(baseline.vlm.config.text_config.hidden_size, config["actions"]["hidden_dim"])
        self.feature_state = None
        baseline.vlm.get_base_model().model.language_model.register_forward_hook(self.capture)

    @property
    def vlm(self):
        return self.baseline.vlm

    @property
    def vis_id(self):
        return self.baseline.vis_id

    def capture(self, module, args, output):
        if self.baseline.positions is not None:
            hidden = output[0]
            self.feature_state = hidden[torch.arange(len(hidden), device=hidden.device), self.baseline.positions]

    def encode(self, inputs):
        self.feature_state = None
        language, boxes = self.baseline(inputs)
        if self.feature_state is None:
            raise RuntimeError("Baseline <vis> feature hook did not run")
        features, self.feature_state = self.feature_state, None
        return language, boxes, features.float()

    def forward(self, inputs):
        language, initial, features = self.encode(inputs)
        trajectory = rollout(self.action_policy, features, initial, self.config["actions"], greedy=True)
        return language, trajectory["boxes"]


def build_model(config, checkpoint=None, stage=None):
    from transformers import AutoProcessor
    from peft import PeftModel
    from train_perception_qwen import VIS, PerceptionQwen
    from perception_extension_runtime import load_qwen
    source = formatted(config, checkpoint or config["model"]["initial_checkpoint"])
    if not (source / "box_head.pt").is_file():
        raise FileNotFoundError(f"A trained Perception adapter and box_head.pt are required: {source}")
    if stage == "grpo" and not (source / "action_policy.pt").is_file():
        raise FileNotFoundError(f"GRPO requires the action-SFT checkpoint: {source}")
    vlm, _ = load_qwen(config["model"]["path"], disable_mmap=config["model"].get("disable_mmap", True))
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    vlm.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    train_vlm = stage == "sft" and config["sft"].get("train_backbone", True)
    vlm = PeftModel.from_pretrained(vlm, source, is_trainable=train_vlm)
    if train_vlm:
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    baseline = PerceptionQwen(vlm, processor.tokenizer.convert_tokens_to_ids(VIS), config["model"]["head_dim"])
    baseline.box_head.load_state_dict(torch.load(source / "box_head.pt", map_location="cpu", weights_only=True))
    baseline.box_head.requires_grad_(stage == "sft" and config["sft"].get("train_box_head", True))
    model = ActionsPerception(baseline, config)
    if (source / "action_policy.pt").is_file():
        model.action_policy.load_state_dict(torch.load(source / "action_policy.pt", map_location="cpu", weights_only=True))
    if stage != "sft":
        baseline.requires_grad_(False)
    return model.to(config["device"]), processor


def _to_device(batch, device):
    inputs, target, positive = batch
    return {k: v.to(device) for k, v in inputs.items()}, target.to(device), positive.to(device).bool()


def sft_loss(model, batch, config):
    from torchvision.ops import generalized_box_iou_loss
    import torch.nn.functional as F
    inputs, target, positive = _to_device(batch, config["device"])
    with torch.autocast(device_type=torch.device(config["device"]).type, dtype=torch.bfloat16):
        language, initial, features = model.encode(inputs)
    l1 = giou = initial.sum() * 0
    if positive.any():
        l1 = F.l1_loss(initial[positive], target[positive])
        giou = generalized_box_iou_loss(initial[positive], target[positive], reduction="mean")
    action = imitation_loss(model.action_policy, features, initial, target, positive, config["actions"])
    parts = {"language": language, "l1": l1, "giou": giou, "action": action}
    loss = sum(config["loss"][name] * part for name, part in parts.items())
    return loss, {key: float(value.detach()) for key, value in parts.items()}


@torch.no_grad()
def generated_features(model, processor, samples, config, labels):
    """Generate a category and replay its token sequence; never inspect GT boxes."""
    from train_perception_qwen import Collator, VIS
    from clear_uav.generation_constraints import label_prefix_allowed_tokens
    from clear_uav.table4 import event_probability
    tokenizer = processor.tokenizer
    inputs, _, _ = Collator(processor, config, training=False)(samples)
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}
    answers = [label + VIS for label in labels] + ["no_event"]
    length = inputs["input_ids"].shape[1]
    max_tokens = max(len(tokenizer.encode(a, add_special_tokens=False)) for a in answers) + 1
    with torch.autocast(device_type=torch.device(config["device"]).type, dtype=torch.bfloat16):
        generated = model.vlm.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
            prefix_allowed_tokens_fn=label_prefix_allowed_tokens(tokenizer, answers, prompt_length=length),
            return_dict_in_generate=True, output_scores=True,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    tokens = generated.sequences[:, length:]
    categories = [text.strip() for text in tokenizer.batch_decode(tokens, skip_special_tokens=True)]
    scores = [event_probability(generated.scores[0][i], tokenizer, labels) for i in range(len(samples))]
    raw = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    has_vis = (tokens == model.vis_id).any(-1)
    eos = tokens == tokenizer.eos_token_id
    completion_mask = (eos.cumsum(1) - eos.long()) == 0
    replay = inputs | {"input_ids": generated.sequences, "attention_mask": torch.cat((inputs["attention_mask"], completion_mask.long()), 1), "logits_to_keep": 1}
    if "mm_token_type_ids" in inputs:
        replay["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tokens)), 1)
    del generated
    with torch.autocast(device_type=torch.device(config["device"]).type, dtype=torch.bfloat16):
        _, initial, features = model.encode(replay)
    return initial, features, categories, has_vis, scores, raw


def save_checkpoint(model, processor, config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    model.vlm.save_pretrained(destination, save_embedding_layers=False)
    processor.save_pretrained(destination)
    torch.save(model.baseline.box_head.state_dict(), destination / "box_head.pt")
    torch.save(model.action_policy.state_dict(), destination / "action_policy.pt")
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def reward_provider(config, labels):
    if not config.get("reward_model", {}).get("enabled", False):
        return None
    from perception_reward_model import CachedVisualReward
    return CachedVisualReward(config, labels, torch.device(config["device"]))


def add_visual_reward(rewards, provider, samples, boxes, categories, eligible, config):
    if provider is None:
        return rewards
    group_size = len(boxes) // len(samples)
    values = []
    for i, sample in enumerate(samples):
        if not bool(eligible[i * group_size:(i + 1) * group_size].any()):
            values.append(boxes.new_zeros(group_size))
            continue
        scored = provider.score(sample.record_uid, boxes[i * group_size:(i + 1) * group_size], categories[i])
        values.append(torch.as_tensor(scored, device=boxes.device, dtype=torch.float32).reshape(group_size))
    bonus = torch.cat(values).detach()
    return rewards + config["reward_model"]["weight"] * torch.where(eligible, bonus, torch.zeros_like(bonus))


@torch.no_grad()
def validate(model, processor, loader, config, stage, labels):
    model.eval()
    total, count = 0.0, 0
    for batch, samples in loader:
        if stage == "sft":
            loss, _ = sft_loss(model, batch, config)
            total += float(loss) * len(samples)
        else:
            initial, features, categories, valid, _, _ = generated_features(model, processor, samples, config, labels)
            target = torch.tensor([s.bbox_1000 or (0, 0, 0, 0) for s in samples], device=initial.device).float() / 1000
            positive = torch.tensor([bool(s.presence) for s in samples], device=initial.device)
            correct = torch.tensor([bool(s.presence) and c == s.label for s, c in zip(samples, categories)], device=initial.device) & valid
            refined = rollout(model.action_policy, features, initial, config["actions"], greedy=True, enabled=valid)["boxes"]
            # Count missed or misclassified positives as zero; exclude negatives.
            total -= float((aligned_iou(refined, target) * correct).sum())
            count += int(positive.sum())
            continue
        count += len(samples)
    if count == 0:
        raise ValueError("Validation has no usable records / no positive records")
    return total / count


class SampleCollator:
    def __init__(self, processor, config, training):
        from train_perception_qwen import Collator
        self.collator = Collator(processor, config, training=True) if training else None

    def __call__(self, samples):
        # GRPO prepares the generated (not teacher-forced) conversation later.
        return self.collator(samples) if self.collator is not None else None, samples


def train(config, stage, checkpoint=None):
    from transformers import get_cosine_schedule_with_warmup, set_seed
    # Loads existing project helpers only when running (not for --help).
    import train_perception_qwen
    from clear_uav.table4 import definitions_from_config, discovery_sampler, read_discovery_samples
    settings = config[stage]
    if config["actions"]["max_steps"] < 1:
        raise ValueError("actions.max_steps must be positive")
    if stage == "grpo" and settings["group_size"] < 2:
        raise ValueError("GRPO group_size must be at least 2")
    if stage == "grpo":
        settings.setdefault("updates_per_rollout", 2)
        if settings["updates_per_rollout"] < 1:
            raise ValueError("GRPO updates_per_rollout must be positive")
    if settings["epochs"] < 1 or settings["gradient_accumulation"] < 1:
        raise ValueError("epochs and gradient_accumulation must be positive")
    if settings.get("steps") is not None and settings["steps"] < 1:
        raise ValueError("steps must be positive when provided")
    for key in ["max_train_samples", "max_val_samples"]:
        if config.get(key) is not None and config[key] < 1:
            raise ValueError(f"{key} must be positive when provided")
    output = output_path(config, stage)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}; use a new --output. Optimizer resume is not implemented.")
    set_seed(config["seed"])
    source = checkpoint or (output_path(config, "sft") / "best" if stage == "grpo" else config["model"]["initial_checkpoint"])
    samples = read_discovery_samples(config, config["protocol"], "train")
    val_samples = read_discovery_samples(config, config["protocol"], "val")
    if config.get("max_train_samples"):
        samples = samples[:config["max_train_samples"]]
    if config.get("max_val_samples"):
        val_samples = val_samples[:config["max_val_samples"]]
    if not samples or not val_samples:
        raise ValueError("Train and validation splits must both be nonempty")
    model, processor = build_model(config, source, stage)
    labels, _ = definitions_from_config(config)
    provider = reward_provider(config, labels) if stage == "grpo" else None
    collator = SampleCollator(processor, config, training=stage == "sft")
    loader = DataLoader(samples, batch_size=settings["batch_size"], sampler=discovery_sampler(samples, config["train"], config["seed"]),
                        collate_fn=collator, num_workers=config["train"]["num_workers"])
    val_loader = DataLoader(val_samples, batch_size=settings["batch_size"], collate_fn=collator,
                            num_workers=config["train"]["num_workers"])
    if stage == "sft":
        groups = [{"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": settings["learning_rate"]},
                  {"params": [p for p in model.baseline.box_head.parameters() if p.requires_grad], "lr": settings["head_learning_rate"]},
                  {"params": model.action_policy.parameters(), "lr": settings["policy_learning_rate"]}]
    else:
        groups = [{"params": model.action_policy.parameters(), "lr": settings["policy_learning_rate"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    accumulation = settings["gradient_accumulation"]
    total_updates = math.ceil(len(loader) / accumulation) * settings["epochs"]
    if stage == "grpo":
        total_updates *= settings["updates_per_rollout"]
    if settings.get("steps"):
        total_updates = min(total_updates, settings["steps"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * settings["warmup_ratio"]), total_updates)
    if stage == "grpo":
        reference = copy.deepcopy(model.action_policy).requires_grad_(False).eval()
        old_policy = copy.deepcopy(reference).requires_grad_(False).eval()
    config = copy.deepcopy(config)
    config["run"] = {"stage": stage, "initial_checkpoint": str(formatted(config, source)),
                     "selection": "val_total_sft_loss" if stage == "sft" else "val_generated_category_gated_positive_mean_iou",
                     "rl_scope": "none" if stage == "sft" else "discrete_box_actions_only; frozen category generator and initial box head"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    best, history, updates = float("inf"), [], 0
    from tqdm import tqdm
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        if stage == "grpo" or not config["sft"].get("train_backbone", True):
            model.baseline.eval()
        optimizer.zero_grad(set_to_none=True)
        total, observed, eligible_count, all_count = 0.0, 0, 0, 0
        rollout_buffer = []
        for step, (batch, current) in enumerate(tqdm(loader, desc=f"actions {stage} {epoch}")):
            if stage == "sft":
                loss, parts = sft_loss(model, batch, config)
            else:
                initial, features, categories, valid, _, _ = generated_features(model, processor, current, config, labels)
                target = torch.tensor([s.bbox_1000 or (0, 0, 0, 0) for s in current], device=initial.device).float() / 1000
                eligible = torch.tensor([bool(s.presence) and c == s.label for s, c in zip(current, categories)], device=initial.device) & valid
                eligible_count += int(eligible.sum())
                all_count += len(current)
                k = settings["group_size"]
                grouped_features, grouped_initial = features.repeat_interleave(k, 0), initial.repeat_interleave(k, 0)
                grouped_target, grouped_eligible = target.repeat_interleave(k, 0), eligible.repeat_interleave(k, 0)
                # One behavior snapshot for the entire accumulation buffer.
                # It stays fixed through ALL updates_per_rollout optimizations.
                if not rollout_buffer:
                    old_policy.load_state_dict(model.action_policy.state_dict())
                with torch.no_grad():
                    trajectory = rollout(old_policy, grouped_features, grouped_initial, config["actions"], enabled=grouped_eligible)
                    rewards = trajectory_reward(grouped_initial, trajectory["boxes"], grouped_target, grouped_eligible, trajectory, config["reward"])
                    rewards = add_visual_reward(rewards, provider, current, trajectory["boxes"], categories, grouped_eligible, config)
                    advantages = group_advantages(rewards, k, settings["advantage_epsilon"])
                rollout_buffer.append({"features": features, "initial": initial, "target": target, "eligible": eligible,
                    "grouped_features": grouped_features, "trajectory": trajectory, "rewards": rewards, "advantages": advantages})
                if len(rollout_buffer) == accumulation or step + 1 == len(loader):
                    update_rows = update_rollout_buffer(model.action_policy, reference, rollout_buffer, config["actions"],
                        settings, optimizer, scheduler, max_updates=total_updates - updates)
                    for parts in update_rows:
                        updates += 1
                        total += parts["loss"]
                        observed += 1
                        print(json.dumps({"stage": stage, "update": updates, **parts}), flush=True)
                    rollout_buffer.clear()
                    if updates >= total_updates:
                        break
                continue
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (loss / group_size).backward()
            total += float(loss.detach())
            observed += 1
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                print(json.dumps({"stage": stage, "update": updates, "loss": float(loss.detach()), **parts}), flush=True)
                if updates >= total_updates:
                    break
        metric = validate(model, processor, val_loader, config, stage, labels)
        history.append({"epoch": epoch, "updates": updates, "train_loss": total / max(observed, 1), "selection_loss": metric,
                        "eligible_localization_records": eligible_count, "seen_records": all_count})
        if metric < best:
            best = metric
            save_checkpoint(model, processor, config, output / "best")
        save_checkpoint(model, processor, config, output / "last")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)
        if updates >= total_updates:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["sft", "grpo"], default="sft")
    parser.add_argument("--checkpoint", help="Warmstart source, not optimizer resume")
    parser.add_argument("--output", help="New output root; stages are written under sft/ and grpo/")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int, help="Maximum optimizer updates")
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ["output", "max_train_samples", "max_val_samples"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ["epochs", "steps"]:
        if getattr(args, key) is not None:
            config[args.stage][key] = getattr(args, key)
    train(config, args.stage, args.checkpoint)


if __name__ == "__main__":
    main()
