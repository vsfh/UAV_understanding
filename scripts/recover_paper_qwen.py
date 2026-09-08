"""Adopt a completed direct-Qwen run without loading a model or running a job.

Explicit recovery only: verify both outputs before writing either missing receipt.
Existing receipts are never replaced. Older/partial outputs without the required
configuration, data, or training evidence remain untouched for manual review.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid


def equivalent(actual, expected, label):
    """Strict structure, tolerant floating point; NaN never compares equal."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label}: fields differ")
        for key, value in expected.items():
            equivalent(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"{label}: list differs")
        for i, (left, right) in enumerate(zip(actual, expected)):
            equivalent(left, right, f"{label}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"{label}: numeric value differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label}: value differs")


def unique_rows(rows, label):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: empty/non-list records")
    mapped = {}
    for row in rows:
        uid = row.get("record_uid") if isinstance(row, dict) else None
        if not isinstance(uid, str) or not uid or uid in mapped:
            raise ValueError(f"{label}: missing/duplicate record UID")
        mapped[uid] = row
    return mapped


def expected_updates(train, samples):
    budget = train.get("samples_per_epoch", "positive_records")
    if budget == "positive_records":
        budget = sum(sample.presence for sample in samples)
    elif budget is None:
        budget = len(samples)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("Invalid recorded training sample budget")
    for key in ("batch_size", "gradient_accumulation", "epochs"):
        if isinstance(train[key], bool) or not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError(f"Invalid train.{key}")
    batches = math.ceil(budget / train["batch_size"])
    return math.ceil(batches / train["gradient_accumulation"]) * train["epochs"]


def check_training_log(directory, updates):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not list(directory.glob("events.out.tfevents.*")):
        raise ValueError("Training event log missing; cannot verify completion without rerunning")
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0}, purge_orphaned_data=False)
    events.Reload()
    if "train/loss" not in events.Tags()["scalars"]:
        raise ValueError("Training completion evidence has no train/loss steps")
    values = events.Scalars("train/loss")
    if ([item.step for item in values] != list(range(1, updates + 1))
            or not all(math.isfinite(item.value) for item in values)):
        raise ValueError(f"Training log does not contain exactly {updates} finite, consecutive updates")
    return {"optimizer_updates": updates, "loss_records": len(values)}


def verify_outputs(config, protocol, root, backend=None):
    """Read and recompute only; every model/training entry point is out of scope."""
    if backend is None:
        from clear_uav import table4 as backend
    from run_matched_ablations import fingerprint
    def resolve(value):
        path = Path(value.format(protocol=protocol, seed=43))
        return path if path.is_absolute() else root / path
    output = config["output"]
    run_dir = resolve(output["root"])
    adapter = resolve(output["adapter"])
    state = json.loads((run_dir / "training_data.json").read_text())
    equivalent(state, backend.training_data_state(config, protocol), "training_data")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    for stored, setting in (("r", "lora_r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        equivalent(adapter_config.get(stored), config["train"][setting], f"adapter.{stored}")
    for key, expected in (("target_modules", backend.LORA_PATTERNS[config["train"]["lora_scope"]]),
                          ("bias", "none"), ("task_type", "CAUSAL_LM")):
        equivalent(adapter_config.get(key), expected, f"adapter.{key}")
    tokenizer = json.loads((adapter / "tokenizer.json").read_text())
    token_ids = {item["content"]: item["id"] for item in tokenizer["added_tokens"]}
    location_ids = [token_ids[f"<loc_{i}>"] for i in range(config["generation"]["location_tokens"])]
    equivalent(adapter_config.get("trainable_token_indices"),
               {"model.language_model.embed_tokens": location_ids, "lm_head": location_ids},
               "adapter location-token indices")
    train_samples = backend.read_discovery_samples(config, protocol, "train")
    training = check_training_log(run_dir / "tensorboard", expected_updates(config["train"], train_samples))
    print(f"[recover] Training evidence verified: {training['optimizer_updates']} optimizer updates", flush=True)
    del train_samples
    calibration = json.loads(resolve(output["calibration"]).read_text())
    result = json.loads(resolve(output["results"]).read_text())
    if result.get("config_sha256") != fingerprint(config):
        raise ValueError("Test result does not bind the current resolved config; preserve for manual review")
    for key, value in (("experiment", config["experiment"]), ("protocol", protocol), ("seed", 43)):
        equivalent(result.get(key), value, f"result.{key}")
    labels = backend.labels_from_config(config)
    counts = {}
    for split, cache_key in (("val", "validation_predictions"), ("test", "test_predictions")):
        samples = backend.read_discovery_samples(config, protocol, split)
        expected = {sample.record_uid: sample for sample in samples}
        if not expected or len(expected) != len(samples):
            raise ValueError(f"{split}: empty/duplicate source IDs")
        cached = unique_rows(json.loads(resolve(output[cache_key]).read_text()), split)
        if set(cached) != set(expected):
            raise ValueError(f"{split}: cache does not cover exactly the frozen split")
        predictions = [cached[s.record_uid]["prediction"] for s in samples]
        for prediction in predictions:
            score = prediction.get("presence_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{split}: invalid presence score")
            if prediction.get("category") not in [None, *labels]:
                raise ValueError(f"{split}: category outside configured ontology")
            box = prediction.get("bbox_1000")
            if box is not None and (not isinstance(box, list) or len(box) != 4 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
                raise ValueError(f"{split}: invalid candidate box")
        counts[split] = len(samples)
        if split == "val":
            threshold, selection = backend.select_threshold(samples, predictions, config["validation"])
            equivalent(calibration.get("threshold"), threshold, "validation threshold")
            equivalent(calibration.get("selection"), {"policy": "max_recall_at_fpr",
                       "max_n_fpr": config["validation"]["max_n_fpr"], **selection}, "selection")
            saved_metrics = calibration["metrics"]
        else:
            saved_rows = unique_rows(result.get("rows"), "test result")
            if set(saved_rows) != set(expected):
                raise ValueError("Test result does not cover exactly the frozen split")
            for sample, prediction in zip(samples, predictions):
                row = saved_rows[sample.record_uid]
                equivalent(row.get("prediction"), prediction, "test prediction/cache")
                equivalent(row.get("target"), {"presence": sample.presence,
                           "bbox_1000": sample.bbox_1000, "category": sample.label}, "test target")
                equivalent(row.get("group_id"), sample.group_id, "test group")
                equivalent(row.get("negative_subtype"), sample.negative_subtype, "test negative subtype")
                equivalent(row.get("iou"), backend.box_iou(sample.bbox_1000, prediction["bbox_1000"]), "test IoU")
            saved_metrics = result["metrics"]
        metrics = backend.table4_metrics(samples, predictions, labels, threshold, True)
        equivalent(saved_metrics, metrics, f"{split} metrics")
        print(f"[recover] {split}: {len(samples)} records, predictions and metrics verified", flush=True)
    return {"training": training, "records": counts, "threshold": threshold,
            "checks": ["current training manifest and negative partition", "adapter LoRA settings",
                       "consecutive training optimizer steps", "test config fingerprint",
                       "complete unique validation/test IDs", "test targets and cached predictions",
                       "validation-only threshold selection", "validation and test metrics"],
            "historical_config_scope": "Test config is recorded; training compatibility is checked against data marker, LoRA settings and step budget. Full historical optimizer settings were not separately recorded."}


def source_files(config, protocol, root):
    """Source identities bound at adoption time, not claimed as historical receipts."""
    def resolve(value):
        p = Path(value.format(protocol=protocol, seed=43))
        return p if p.is_absolute() else root / p
    data = resolve(config["data"]["root"])
    paths = [data / protocol / n for n in ("train.csv", "val.csv", "test_inputs.csv", "test_labels_private.csv")]
    paths += [resolve(config["data"][k]) for k in ("bbox_annotations", "ontology", "labels")]
    paths += sorted((resolve(config["output"]["root"]) / "tensorboard").glob("events.out.tfevents.*"))
    return paths


def write_receipt_exclusive(path, payload):
    """Publish complete JSON atomically without replacing an existing receipt."""
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def recover_existing_qwen(protocol, catalog, runner, verifier=verify_outputs):
    names = [f"{protocol}_qwen_calibration", f"{protocol}_qwen_results"]
    entries = [catalog[name] for name in names]
    root = runner.ROOT
    print(f"[recover] {protocol}: verify existing Qwen outputs; no model execution", flush=True)
    with ExitStack() as stack:
        for _, _, _, marker in entries:
            stack.enter_context(runner.stage_lock(runner.lock_path(marker), marker.stem))
        missing = []
        for _, config, artifact, marker in entries:
            if marker.exists():
                runner.completed(config, artifact, marker)
            else:
                if not artifact.is_file():
                    raise ValueError(f"Recovery requires completed calibration AND test output: {artifact}")
                missing.append(marker)
        if not missing:
            print("[recover] Existing receipts verified; nothing changed.", flush=True)
            return
        config = entries[0][1]
        if any(runner.fingerprint(c) != runner.fingerprint(config) for _, c, _, _ in entries):
            raise ValueError("Qwen calibration/test configurations differ")
        def identities():
            values = {}
            for _, c, artifact, marker in entries:
                values[marker.stem] = {"artifact": runner.artifact_identity(artifact, root),
                                      "support": runner.supporting_artifacts(c, marker)}
            values["sources"] = {identity["path"]: identity for p in source_files(config, protocol, root)
                                 for identity in [runner.artifact_identity(p, root)]}
            return values
        before = identities()
        for _, c, artifact, marker in entries:
            runner.validate_artifact(c, artifact, marker)
        evidence = verifier(config, protocol, root)
        if identities() != before:
            raise ValueError("Source/adapter/cache/output changed during recovery; no receipts written")
        receipts = []
        for _, c, artifact, marker in entries:
            if marker not in missing:
                continue
            dependencies = {name: before[name]["artifact"] for name in runner.dependency_names(marker)}
            receipts.append((marker, {"config_sha256": runner.fingerprint(c), "resolved_config": c,
                "stage": marker.stem, "protocol": protocol, "seed": 43,
                "artifact_identity": before[marker.stem]["artifact"],
                "dependencies": dependencies, "supporting_artifacts": before[marker.stem]["support"],
                "recovery": {"kind": "validated_existing_outputs", "recovered_utc": datetime.now(timezone.utc).isoformat(),
                             "no_training_or_inference": True, "evidence": evidence,
                             "source_identities": before["sources"]}}))
        for marker, payload in receipts:
            if marker.exists():
                raise ValueError(f"Receipt appeared during recovery; refusing to replace {marker}")
            marker.parent.mkdir(parents=True, exist_ok=True)
            write_receipt_exclusive(marker, payload)
            print(f"[recover] Registered {marker.name}", flush=True)
        for _, c, artifact, marker in entries:
            if not runner.completed(c, artifact, marker):
                raise ValueError(f"Recovered receipt did not verify: {marker}")
        print("[recover] Ready. Existing weights/predictions preserved; no experiment started.", flush=True)
