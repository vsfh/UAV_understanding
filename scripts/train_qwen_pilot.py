"""Small, train-only data/QA interventions on the same Direct Qwen adapter."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
import yaml
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clear_uav.modeling import assistant_only_labels, load_qwen
from clear_uav.table4 import (QwenDiscoveryCollator, add_qwen_location_tokens,
    discovery_sampler, grounding_target, read_discovery_samples)

DEFAULT = "configs/yaml/qwen_pilot.yaml"


def settings(path):
    pilot = yaml.safe_load(Path(path).read_text())
    config = yaml.safe_load(Path(pilot["base_config"]).read_text())
    return pilot, config


def load_pilot(config, checkpoint, training=False):
    progress = tqdm(total=3, desc="load Qwen + adapter", unit="stage")
    model, processor = load_qwen(config["model"]["path"], device_map="cuda")
    progress.update()
    add_qwen_location_tokens(model, processor, config["generation"]["location_tokens"])
    progress.update()
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=training, local_files_only=True)
    progress.update()
    progress.close()
    return model, processor


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def spatial_bin(sample):
    if not sample.presence:
        return "negative"
    x1, y1, x2, y2 = sample.bbox_1000
    area = (x2 - x1) * (y2 - y1) / 1e6
    return (min(2, int((x1 + x2) / 2000 * 3)),
            min(2, int((y1 + y2) / 2000 * 3)),
            0 if area < .01 else 1 if area < .05 else 2 if area < .2 else 3)


def diverse_order(samples):
    # Round-robin across content groups AND ROI geometry, never val/test errors.
    buckets = defaultdict(list)
    for sample in sorted(samples, key=lambda s: digest(s.record_uid)):
        buckets[(sample.group_id, spatial_bin(sample))].append(sample)
    keys = sorted(buckets, key=lambda k: digest(str(k)))
    return [buckets[k][i] for i in range(max(map(len, buckets.values())))
            for k in keys if i < len(buckets[k])]


def prepare(pilot, config):
    output = Path(pilot["output"])
    output.mkdir(parents=True, exist_ok=True)
    train = read_discovery_samples(config, pilot["protocol"], "train")
    val = read_discovery_samples(config, pilot["protocol"], "val")
    assert not ({s.record_uid for s in train} & {s.record_uid for s in val})
    pools = defaultdict(list)
    for sample in train:
        pools[sample.label or "no_event"].append(sample)
    random_half, diverse_half = [], []
    for label, samples in sorted(pools.items()):
        count = max(1, round(len(samples) * pilot["subset_fraction"]))
        random_half.extend(sorted(samples, key=lambda s: digest("random43" + s.record_uid))[:count])
        diverse_half.extend(diverse_order(samples)[:count])
    val_pools = defaultdict(list)
    for sample in val:
        val_pools[sample.label or "no_event"].append(sample)
    probe = []
    for label, samples in sorted(val_pools.items()):
        count = pilot["val_negatives"] if label == "no_event" else pilot["val_per_class"]
        probe.extend(diverse_order(samples)[:count])
    manifests = {"control": train, "random_half": random_half,
                 "diverse_half": diverse_half, "diverse_qa": diverse_half, "val_probe": probe}
    summary = {}
    for name, samples in manifests.items():
        ids = [s.record_uid for s in samples]
        (output / f"{name}.json").write_text(json.dumps(ids, indent=2))
        summary[name] = {"n": len(ids), "classes": dict(Counter(s.label or "no_event" for s in samples)),
                         "groups": len({s.group_id for s in samples}),
                         "ids_sha256": digest("\n".join(ids))}
    summary["protocol"] = "train-only selection; validation probe; no test access"
    (output / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({name: {k: v for k, v in row.items() if k != "classes"}
                      for name, row in summary.items() if isinstance(row, dict)}, indent=2))


def auxiliary(collator, sample, task):
    if not sample.presence or task == 0:
        return collator.messages(sample, "yes" if sample.presence else "no",
            prompt="Does this image contain a supported operational event? Answer yes or no. Categories:\n" + collator.prompt.split("Categories:")[-1])
    if task == 1:
        return collator.messages(sample, sample.label, image=str(sample.evidence_path),
            prompt="Identify the supported event in this contextual evidence crop. Return only its canonical category.\n" + collator.prompt.split("Categories:")[-1])
    return collator.messages(sample, grounding_target(sample, collator.location_count),
        prompt=f"Locate the complete contextual evidence region of {sample.label} in this image. "
               "Return its category followed by four location tokens. These are context ROIs, not tight object boxes.")


def train(pilot, config, arm, loaded=None):
    set_seed(pilot["seed"])
    output = Path(pilot["output"]) / arm
    output.mkdir(parents=True, exist_ok=False)
    pool = {"full_resampled": "control", "random_qa": "random_half"}.get(arm, arm)
    selected = set(json.loads((output.parent / f"{pool}.json").read_text()))
    all_train = read_discovery_samples(config, pilot["protocol"], "train")
    samples = [s for s in all_train if s.record_uid in selected]
    model, processor = loaded or load_pilot(config, pilot["checkpoint"], training=True)
    processor.tokenizer.padding_side = "left"
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    collator = QwenDiscoveryCollator(processor, config, True)
    count = pilot["steps"] * pilot["accumulation"]
    sampling = config["train"] | {"samples_per_epoch": count}
    indices = list(discovery_sampler(samples, sampling, pilot["seed"]))
    control_path = output.parent / "control" / "training.json"
    if arm != "control" and control_path.exists():
        # Match the control's per-step categories, not just expected class proportions.
        by_uid = {s.record_uid: s.label for s in all_train}
        by_label = defaultdict(list)
        for i, sample in enumerate(samples):
            by_label[sample.label].append(i)
        rng = random.Random(pilot["seed"])
        exposure = json.loads(control_path.read_text())["exposure"]
        indices = [rng.choice(by_label[by_uid[row["uid"]]]) for row in exposure]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=pilot["learning_rate"], weight_decay=.01)
    history, exposure = [], []
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(enumerate(indices), total=count, desc=arm, unit="image")
    for index, sample_index in progress:
        sample = samples[sample_index]
        aux = arm.endswith("_qa") and index % 4 == 3
        if aux:
            conversation = auxiliary(collator, sample, (index // 4) % 3)
            batch = processor.apply_chat_template([conversation], tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt", processor_kwargs={"padding": True, "size": {
                    "longest_edge": config["input"]["max_pixels"], "shortest_edge": config["input"]["min_pixels"]}})
            batch["labels"] = assistant_only_labels(batch["input_ids"], batch["attention_mask"], processor.tokenizer)
        else:
            batch = collator([sample])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        (loss / pilot["accumulation"]).backward()
        value = loss.detach().item()
        history.append(value)
        exposure.append({"uid": sample.record_uid, "task": "auxiliary" if aux else "grounding"})
        if (index + 1) % pilot["accumulation"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        progress.set_postfix(loss=f"{value:.3f}", update=(index + 1) // pilot["accumulation"])
    model.save_pretrained(output / "adapter", save_embedding_layers=False)
    processor.save_pretrained(output / "adapter")
    (output / "training.json").write_text(json.dumps({"arm": arm, "settings": pilot,
        "base_config": config, "seconds": time.monotonic() - started, "loss": history,
        "exposure": exposure, "unique_images": len({r["uid"] for r in exposure})}, indent=2))
    return model, processor


def remaining(pilot, config):
    # Reuse the frozen 8B backbone in memory, reset ALL adapter/token parameters for each arm.
    from test_qwen_pilot import evaluate, rollout_probe
    model, processor = load_pilot(config, pilot["checkpoint"], training=True)
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    model.eval()
    if not (Path(pilot["output"]) / "baseline_fresh_probe.json").exists():
        evaluate(pilot, config, "baseline_fresh", loaded=(model, processor))
    for arm in pilot["arms"]:
        output = Path(pilot["output"])
        if (output / f"{arm}_probe.json").exists():
            continue
        if (output / arm / "training.json").exists():
            from safetensors.torch import load_file
            state = load_file(str(output / arm / "adapter" / "adapter_model.safetensors"))
            set_peft_model_state_dict(model, state)
        else:
            set_peft_model_state_dict(model, dict(initial))
            current = get_peft_model_state_dict(model)
            assert all(torch.equal(current[k].detach().cpu(), v) for k, v in initial.items())
            train(pilot, config, arm, (model, processor))
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        evaluate(pilot, config, arm, loaded=(model, processor))
    set_peft_model_state_dict(model, dict(initial))
    rollout_probe(pilot, config, loaded=(model, processor))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=["control", "random_half", "diverse_half", "diverse_qa", "full_resampled", "random_qa", "remaining"])
    args = parser.parse_args()
    pilot, config = settings(args.config)
    if args.prepare:
        prepare(pilot, config)
    elif args.arm == "remaining":
        remaining(pilot, config)
    else:
        train(pilot, config, args.arm)
