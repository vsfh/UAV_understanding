"""Full Qwen + short-response GRPO + zero-initialized contextual-ROI residual.

The reference is the random-half adapter on the SAME frozen Qwen base. The
ROI head reads predicted coordinates, not ground-truth coordinates. Its input
features are detached: box regression cannot overwrite language semantics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from peft import PeftModel
from torchvision.ops import box_iou, generalized_box_iou_loss
from tqdm import tqdm
from transformers import set_seed
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clear_uav.modeling import load_qwen
from clear_uav.table4 import (QwenDiscoveryCollator, add_qwen_location_tokens,
    discovery_sampler, grounding_prefix_allowed_tokens, labels_from_config,
    parse_grounding_tokens, read_discovery_samples)

DEFAULT = "configs/yaml/qwen_grpo_roi.yaml"


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class ResidualROI(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.shift, self.scale = config["max_center_shift"], config["max_log_scale"]
        self.net = nn.Sequential(nn.LayerNorm(hidden_size + 4),
            nn.Linear(hidden_size + 4, config["head_dim"]), nn.GELU(),
            nn.Linear(config["head_dim"], 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden, base):
        delta = self.net(torch.cat((hidden.detach().float(), base.float()), -1)).tanh()
        center = (base[:, :2] + base[:, 2:]) / 2 + self.shift * delta[:, :2]
        size = (base[:, 2:] - base[:, :2]) * (self.scale * delta[:, 2:]).exp()
        refined = torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1)
        return refined, delta


class GroundedQwen(nn.Module):
    def __init__(self, vlm, location_ids, config):
        super().__init__()
        self.vlm, self.location_ids = vlm, location_ids
        self.head = ResidualROI(vlm.config.text_config.hidden_size, config["model"])
        self.position = self.hidden = None
        vlm.get_base_model().model.language_model.register_forward_hook(self.capture)

    def capture(self, module, args, output):
        if self.position is not None:
            self.hidden = output[0][:, self.position, :]

    def replay(self, inputs, tokens, base_box=None, grad_logits=True):
        # B=1 replay bounds activation memory. Keep only completion logits.
        length = inputs["input_ids"].shape[1]
        tokens = tokens.unsqueeze(0)
        batch = inputs | {"input_ids": torch.cat((inputs["input_ids"], tokens), 1),
            "attention_mask": torch.cat((inputs["attention_mask"], torch.ones_like(tokens)), 1)}
        if "mm_token_type_ids" in inputs:
            batch["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tokens)), 1)
        if base_box is not None:
            loc = torch.isin(tokens[0], tokens.new_tensor(self.location_ids)).nonzero().flatten()
            self.position = length + int(loc[-1])
        with torch.set_grad_enabled(torch.is_grad_enabled() and grad_logits), torch.autocast(inputs["input_ids"].device.type, dtype=torch.bfloat16):
            result = self.vlm(**batch, logits_to_keep=tokens.shape[1] + 1, use_cache=False)
        hidden = self.hidden
        # Do not retain the hook during checkpointed backward recomputation.
        self.position = self.hidden = None
        boxes = delta = None
        if base_box is not None:
            boxes, delta = self.head(hidden, base_box)
        return result.logits[0, :-1].float(), boxes, delta


def build_model(config, checkpoint=None, training=False):
    progress = tqdm(total=3, desc="load Qwen / adapters / ROI head", unit="stage")
    vlm, processor = load_qwen(config["model"]["path"], device_map=config["device"])
    location_ids = add_qwen_location_tokens(vlm, processor, config["generation"]["location_tokens"])
    progress.update()
    adapter = Path(checkpoint) / "adapter" if checkpoint else config["model"]["initial_adapter"]
    vlm = PeftModel.from_pretrained(vlm, adapter, is_trainable=training, local_files_only=True)
    if training:
        vlm.load_adapter(config["model"]["initial_adapter"], adapter_name="reference",
                         is_trainable=False, local_files_only=True)
        vlm.set_adapter("default")
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Matching rollout/replay distributions requires dropout disabled in both.
    for module in vlm.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    progress.update()
    model = GroundedQwen(vlm, location_ids, config)
    model.head.to(config["device"])
    if checkpoint:
        model.head.load_state_dict(torch.load(Path(checkpoint) / "roi_head.pt", map_location="cpu", weights_only=True))
    processor.tokenizer.padding_side = "left"
    progress.update()
    progress.close()
    return model, processor


def grammar(processor, labels, model, length=0):
    return grounding_prefix_allowed_tokens(processor.tokenizer, labels,
        location_token_ids=model.location_ids, prompt_length=length)


def constrained_logps(logits, tokens, constraint):
    # Apply exactly the rollout grammar. No top-k/top-p/temperature mismatch.
    values = []
    for i, token in enumerate(tokens):
        allowed = constraint(0, tokens[:i])
        subset = logits[i, allowed].log_softmax(-1)
        values.append(subset[allowed.index(int(token))])
    return torch.stack(values)


def group_advantages(rewards, epsilon):
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)


def grpo_loss(logps, old_logps, ref_logps, advantage, config):
    ratio = (logps - old_logps).exp()
    surrogate = torch.minimum(ratio * advantage,
        ratio.clamp(1 - config["clip"], 1 + config["clip"]) * advantage)
    log_ratio = ref_logps - logps
    kl = log_ratio.exp() - log_ratio - 1  # sampled nonnegative KL estimator
    return (-surrogate + config["beta"] * kl).mean(), kl.mean()


def reward(prediction, sample, config):
    if not sample.presence:
        return config["negative_correct"] if prediction["category"] is None else config["negative_false_positive"]
    if prediction["category"] != sample.label:
        return 0.
    overlap = box_iou(torch.tensor([prediction["bbox_1000"]]), torch.tensor([sample.bbox_1000]))[0, 0].item()
    return config["category_weight"] + config["iou_weight"] * overlap


@torch.no_grad()
def rollout(model, processor, inputs, labels, config, count=1, sample=True):
    model.eval()
    length = inputs["input_ids"].shape[1]
    constraint = grammar(processor, labels, model, length)
    with torch.autocast(inputs["input_ids"].device.type, dtype=torch.bfloat16):
        generated = model.vlm.generate(**inputs, do_sample=sample, temperature=1. if sample else None,
            top_p=1. if sample else None, top_k=0 if sample else None, repetition_penalty=1., num_beams=1,
            num_return_sequences=count, max_new_tokens=config["generation"]["max_new_tokens"],
            prefix_allowed_tokens_fn=constraint, use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            return_dict_in_generate=True, output_scores=True)
    rows = []
    for i in range(count):
        tokens = generated.sequences[i, length:]
        end = (tokens == processor.tokenizer.eos_token_id).nonzero().flatten()
        if len(end):
            tokens = tokens[:int(end[0]) + 1]
        old = torch.stack([scores[i].float().log_softmax(-1)[token]
                           for scores, token in zip(generated.scores, tokens)])
        prediction = parse_grounding_tokens(tokens, processor.tokenizer, labels, model.location_ids)
        rows.append({"tokens": tokens.clone(), "old_logps": old.detach(), "prediction": prediction,
                     "first_scores": generated.scores[0][i].detach().clone()})
    return rows  # generated's KV cache is released here, before any backward.


def head_loss(model, inputs, row, sample, config, train_policy=True):
    base = row["prediction"]["bbox_1000"]
    target = inputs["input_ids"].new_tensor([sample.bbox_1000 or [0, 0, 0, 0]], dtype=torch.float32) / 1000
    valid = sample.presence and row["prediction"]["category"] == sample.label and base is not None
    if not valid and not train_policy:
        return None, target.new_zeros(())
    base = target.new_tensor([base]) / 1000 if valid else None
    logits, boxes, delta = model.replay(inputs, row["tokens"], base, grad_logits=train_policy)
    loss = logits.new_zeros(())
    if valid:
        loss = (config["l1"] * F.l1_loss(boxes, target)
            + config["giou"] * generalized_box_iou_loss(boxes, target, reduction="mean")
            + config["residual"] * delta.square().mean())
    return logits, loss


def supervised_loss(model, batch):
    targets = batch["labels"][0]
    positions = (targets != -100).nonzero().flatten()
    # Full-vocabulary CE, but materialize logits only for assistant tokens.
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.autocast(targets.device.type, dtype=torch.bfloat16):
        logits = model.vlm(**inputs, logits_to_keep=positions - 1, use_cache=False).logits[0]
    return F.cross_entropy(logits.float(), targets[positions])


def save_checkpoint(model, processor, optimizer, config, step):
    path = Path(config["output"]) / f"step_{step:06d}"
    path.mkdir(parents=True, exist_ok=False)
    model.vlm.save_pretrained(path / "adapter", selected_adapters=["default"], save_embedding_layers=False)
    processor.save_pretrained(path / "adapter")
    torch.save(model.head.state_dict(), path / "roi_head.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step,
        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, path / "training.pt")
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (Path(config["output"]) / "latest.txt").write_text(str(path))
    return path


def train(config, resume=None):
    set_seed(config["seed"])
    settings, grpo = config["train"], config["grpo"]
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=bool(resume))
    selected = json.loads(Path(config["data"]["subset"]).read_text())
    wanted = set(selected)
    samples = [s for s in read_discovery_samples(config, config["protocol"], "train") if s.record_uid in wanted]
    assert len(samples) == len(wanted) == len(selected), "Subset must contain unique training IDs only"
    (root / "subset.json").write_text(json.dumps(selected, indent=2))
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "provenance.json").write_text(json.dumps({"initial_adapter": config["model"]["initial_adapter"],
        "subset_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "n_train": len(samples), "training": "continuation of random-half pilot, not half-data from scratch"}, indent=2))
    model, processor = build_model(config, resume, training=True)
    labels = labels_from_config(config)
    collator, supervised = QwenDiscoveryCollator(processor, config, False), QwenDiscoveryCollator(processor, config, True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": settings["learning_rate"]},
        {"params": model.head.parameters(), "lr": settings["head_learning_rate"]}], weight_decay=settings["weight_decay"])
    start = 0
    if resume:
        state = torch.load(Path(resume) / "training.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    accumulation = settings["gradient_accumulation"]
    sampling = settings | {"samples_per_epoch": settings["steps"] * accumulation}
    indices = list(discovery_sampler(samples, sampling, config["seed"]))
    constraint = grammar(processor, labels, model)
    progress = tqdm(range(start, settings["steps"]), desc="GRPO + residual ROI", unit="update")
    with (root / "train.jsonl").open("a", buffering=1) as log:
        for step in progress:
            optimizer.zero_grad(set_to_none=True)
            stats = {"reward": 0., "active": 0., "sft": 0., "roi": 0., "kl": 0.}
            for index in indices[step * accumulation:(step + 1) * accumulation]:
                progress.set_description(f"update {step + 1}: sample {grpo['group_size']} candidates")
                example = samples[index]
                inputs = {k: v.to(config["device"]) for k, v in collator([example]).items()}
                rows = []
                for offset in range(0, grpo["group_size"], grpo["rollout_batch_size"]):
                    rows.extend(rollout(model, processor, inputs, labels, config,
                        min(grpo["rollout_batch_size"], grpo["group_size"] - offset)))
                rewards = torch.tensor([reward(r["prediction"], example, config["reward"]) for r in rows], device=config["device"])
                advantages = group_advantages(rewards, grpo["advantage_epsilon"])
                active = rewards.std(unbiased=False).item() > grpo["advantage_epsilon"]
                stats["reward"] += rewards.mean().item() / accumulation
                stats["active"] += int(active) / accumulation
                # Zero-variance groups have no policy signal. Keep one head example
                # and the SFT anchor; do not spend G reference/backward passes on them.
                chosen = list(enumerate(rows)) if active else [(0, rows[0])]
                progress.set_description(f"update {step + 1}: GRPO / ROI / SFT")
                for i, row in chosen:
                    ref_logps = None
                    if active:
                        model.vlm.set_adapter("reference", inference_mode=True)
                        model.eval()
                        with torch.no_grad():
                            ref_logits, _, _ = model.replay(inputs, row["tokens"])
                            ref_logps = constrained_logps(ref_logits, row["tokens"], constraint).detach()
                            del ref_logits
                        model.vlm.set_adapter("default")
                    model.train()
                    logits, roi = head_loss(model, inputs, row, example, config["loss"], train_policy=active)
                    loss = roi
                    if active:
                        logps = constrained_logps(logits, row["tokens"], constraint)
                        policy, kl = grpo_loss(logps, row["old_logps"], ref_logps, advantages[i], grpo)
                        loss = loss + policy
                        stats["kl"] += kl.detach().item() / (len(chosen) * accumulation)
                    if loss.requires_grad:
                        (loss / (len(chosen) * accumulation)).backward()
                    stats["roi"] += roi.detach().item() / (len(chosen) * accumulation)
                    del logits, loss, roi
                # Full-vocabulary teacher forcing anchors category AND ROI tokens;
                # negatives always contribute, including zero-variance GRPO groups.
                batch = {k: v.to(config["device"]) for k, v in supervised([example]).items()}
                sft = supervised_loss(model, batch)
                (config["loss"]["sft"] * sft / accumulation).backward()
                stats["sft"] += sft.detach().item() / accumulation
                del batch, sft, rows, inputs
            nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            optimizer.step()
            log.write(json.dumps({"step": step + 1, **stats}) + "\n")
            progress.set_postfix({k: f"{v:.3f}" for k, v in stats.items()})
            if (step + 1) % settings["save_every"] == 0 or step + 1 == settings["steps"]:
                save_checkpoint(model, processor, optimizer, config, step + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--resume", help="A step_XXXXXX checkpoint directory")
    parser.add_argument("--steps", type=int, help="Total updates, including resumed updates")
    parser.add_argument("--output", help="Separate output for a short pilot")
    args = parser.parse_args()
    config = read_config(Path(args.resume) / "config.yaml" if args.resume else args.config)
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.output:
        config["output"] = args.output
    train(config, args.resume)
