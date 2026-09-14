"""Frozen Spatial evidence interventions, sharing ONE Qwen replay per image.

No training or generation. Replay the cached model output, never the GT category.
An outer deadline bounds loading + inference + summary to at most 12 hours.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
CONDITIONS = ("full", "no_spatial", "shuffle_xy", "global_memory")
DEFAULT_CONFIG = "configs/yaml/spatial_evidence.yaml"


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def seeded_order(uids, seed):
    return sorted(uids, key=lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest())


def validate_plan(previous, current):
    # Old plans stored the source location; moving identical files is safe.
    previous = {key: value for key, value in previous.items() if key != "source_dir"}
    if previous != current:
        raise ValueError("Checkpoint, cached predictions or intervention plan changed; use a new output directory")


def cached_completion(tokenizer, raw_output):
    tokens = tokenizer.encode(raw_output, add_special_tokens=False)
    if tokenizer.eos_token_id in tokens:
        tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
    return tokens


def read_progress(path):
    """Keep complete paired records; repair only an interrupted final log write."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    ids, offset = set(), 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        if row["record_uid"] in ids or set(row["predictions"]) != set(CONDITIONS):
            raise ValueError("Duplicate or incomplete paired record in progress file")
        ids.add(row["record_uid"])
        offset += len(line)
    else:
        if data and not data.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return ids


class PairedSpatialReadout:
    """Hook the tiny spatial head; do NOT run the backbone once per intervention."""
    def __init__(self, model, seed):
        self.model, self.seed = model, seed
        self.uids, self.boxes = [], None
        self.handle = model.spatial_head.register_forward_hook(self.capture)

    def capture(self, head, args, full):
        import torch
        query, memory, coordinates, padding = args
        with torch.autocast(device_type=query.device.type, enabled=False):
            memory = memory.float()
            shuffled = coordinates.clone()
            for row, uid in enumerate(self.uids):
                valid = (~padding[row]).nonzero().flatten()
                seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{uid}".encode()).digest()[:8], "little")
                generator = torch.Generator(device="cpu").manual_seed(seed)
                order = torch.randperm(len(valid), generator=generator).to(valid.device)
                shuffled[row, valid] = coordinates[row, valid[order]]
            valid_mask = (~padding).unsqueeze(-1)
            mean = (memory * valid_mask).sum(1, keepdim=True) / valid_mask.sum(1, keepdim=True)
            global_memory = mean.expand_as(memory)
            # Calling forward directly avoids recursively invoking this hook.
            states = {"full": full, "no_spatial": query.float(),
                      "shuffle_xy": head.forward(query, memory, shuffled, padding),
                      "global_memory": head.forward(query, global_memory, coordinates, padding)}
            self.boxes = {}
            for name, state in states.items():
                raw = self.model.box_head(state.float()).sigmoid()
                lo = torch.minimum(raw[:, :2], raw[:, 2:])
                hi = torch.maximum(raw[:, :2], raw[:, 2:])
                self.boxes[name] = (torch.cat((lo, hi), dim=-1) * 1000).cpu().tolist()


def paired_record(row, boxes=None, index=None):
    predictions = {name: copy.deepcopy(row["prediction"]) for name in CONDITIONS}
    if boxes is not None:
        for name in CONDITIONS:
            predictions[name]["bbox_1000"] = boxes[name][index]
    return {"record_uid": row["record_uid"], "group_id": row.get("group_id"),
            "target": row["target"], "cached_prediction": row["prediction"],
            "predictions": predictions, "replayed": boxes is not None}


def worker(spec, deadline):
    import torch
    from tqdm import tqdm
    from train_perception_spatial import build_model
    from train_perception_qwen import Collator
    from perception_spatial_head import checkpoint_fingerprint, validate_calibration
    from clear_uav.table4 import definitions_from_config, read_discovery_samples

    source, output = Path(spec["source_dir"]), Path(spec["output"])
    config = read_yaml(source / "config.yaml")
    if config.get("spatial_lora", {}).get("enabled") or "spatial" not in config:
        raise ValueError("Use the current paper Spatial checkpoint, not a different experimental architecture")
    result_path = source / f"{spec['split']}_results.json"
    result = json.loads(result_path.read_text())
    calibration = json.loads((source / "calibration.json").read_text())
    fingerprint = checkpoint_fingerprint(source / "best")
    validate_calibration(calibration, config, source / "best", fingerprint)
    if (result["protocol"] != config["protocol"] or result["seed"] != config["seed"]
            or result["metrics"]["threshold"] != calibration["threshold"]):
        raise ValueError("Cached results and the checkpoint calibration disagree")
    rows = {row["record_uid"]: row for row in result["rows"]}
    if len(rows) != len(result["rows"]):
        raise ValueError("Duplicate source records")
    order = seeded_order(rows, spec["permutation_seed"])
    head_state = torch.load(source / "best" / "spatial_head.pt", map_location="cpu", weights_only=True)
    plan = {"source_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": fingerprint,
            "total_records": len(order), "record_uids": order,
            "residual_gate": head_state["residual_gate"].item(),
            "conditions": list(CONDITIONS), "permutation_seed": spec["permutation_seed"]}
    del head_state
    plan_path = output / "plan.json"
    if plan_path.exists():
        validate_plan(json.loads(plan_path.read_text()), plan)
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    progress_path = output / "paired_predictions.jsonl"
    completed = read_progress(progress_path)
    if not completed <= set(order):
        raise ValueError("Progress contains records outside the cached evaluation")
    todo = [uid for uid in order if uid not in completed]
    if not todo or time.time() >= deadline:
        return
    config["device"] = spec["device"]
    labels, _ = definitions_from_config(config)
    samples = {sample.record_uid: sample for sample in read_discovery_samples(config, config["protocol"], spec["split"])}
    if set(samples) != set(rows):
        raise ValueError("Current evaluation IDs differ from cached predictions")
    for uid, sample in samples.items():
        target = {"presence": sample.presence, "category": sample.label,
                  "bbox_1000": list(sample.bbox_1000) if sample.bbox_1000 is not None else None}
        if rows[uid]["target"] != target:
            raise ValueError(f"Current target differs from cached target: {uid}")
    print("[load] frozen Perception-Spatial; no training or generation", flush=True)
    model, processor = build_model(config, checkpoint=source / "best", is_trainable=False)
    model.eval()
    capture = PairedSpatialReadout(model, spec["permutation_seed"])
    collator = Collator(processor, config, training=False)
    device = torch.device(spec["device"])
    batch_size = spec["batch_size"]
    progress = tqdm(total=len(order), initial=len(completed), desc="paired spatial interventions", unit="image")
    with progress_path.open("a", encoding="utf-8") as log, torch.inference_mode():
        for start in range(0, len(todo), batch_size):
            if time.time() >= deadline:
                break
            uids = todo[start:start + batch_size]
            outputs, active, completions = {}, [], []
            for uid in uids:
                prediction = rows[uid]["prediction"]
                tokens = cached_completion(processor.tokenizer, prediction.get("raw_output", ""))
                if prediction["valid"] and prediction["category"] in labels and model.vis_id in tokens:
                    active.append(uid)
                    completions.append(tokens)
                else:
                    # Keep no-event/invalid outputs unchanged, including their failures.
                    outputs[uid] = paired_record(rows[uid])
            if active:
                inputs, _, _ = collator([samples[uid] for uid in active])
                length = max(map(len, completions))
                tokens = torch.full((len(active), length), processor.tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros_like(tokens)
                for row, completion in enumerate(completions):
                    tokens[row, :len(completion)] = torch.tensor(completion)
                    mask[row, :len(completion)] = 1
                inputs["input_ids"] = torch.cat((inputs["input_ids"], tokens), dim=1)
                inputs["attention_mask"] = torch.cat((inputs["attention_mask"], mask), dim=1)
                if "mm_token_type_ids" in inputs:
                    inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(tokens)), dim=1)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                inputs["logits_to_keep"] = 1
                capture.uids, capture.boxes = active, None
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    model(inputs)
                for index, uid in enumerate(active):
                    outputs[uid] = paired_record(rows[uid], capture.boxes, index)
            for uid in uids:
                log.write(json.dumps(outputs[uid], ensure_ascii=False) + "\n")
            log.flush()  # Every saved record contains all four paired conditions.
            progress.update(len(uids))
            progress.set_postfix(hours_left=f"{max(0, deadline-time.time())/3600:.2f}")
    capture.handle.remove()
    progress.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--worker-deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec = read_yaml(args.config)
    if args.worker_deadline is not None:
        worker(spec, args.worker_deadline)
        return
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    budget = min(float(spec["max_hours"]), 12.0) * 3600
    if budget <= 180:
        raise ValueError("Allow more than three minutes for model loading and summary")
    if args.summary_only:
        subprocess.run([sys.executable, "scripts/summarize_spatial_evidence.py", "--config", args.config],
                       check=True, timeout=min(budget, 120))
        return  # Keep the original worker timing/status record unchanged.
    started = time.monotonic()
    status = {"training": False, "max_hours": budget / 3600, "worker": "not_started"}
    try:
        command = [sys.executable, "scripts/run_spatial_evidence.py", "--config", args.config,
                   "--worker-deadline", str(time.time() + budget - 180)]
        try:
            child = subprocess.run(command, timeout=budget - 120)
            status["worker"] = "returned" if child.returncode == 0 else "failed"
            status["returncode"] = child.returncode
        except subprocess.TimeoutExpired:
            status["worker"] = "deadline_stopped"
        status["elapsed_seconds"] = time.monotonic() - started
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if not (output / "paired_predictions.jsonl").is_file():
            if status["worker"] != "failed":
                print("No paired predictions completed within this run.", file=sys.stderr)
            raise SystemExit(status.get("returncode") or 1)
        remaining = budget - (time.monotonic() - started) - 5
        subprocess.run([sys.executable, "scripts/summarize_spatial_evidence.py", "--config", args.config],
                       check=True, timeout=max(1, remaining))
        if status["worker"] == "failed":
            raise SystemExit(status["returncode"])
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
