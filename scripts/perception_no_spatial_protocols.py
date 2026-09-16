"""Protocol-specific configuration and provenance for the existing Perception recipe."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOLS = ("unseen_site", "forward_temporal")
STAGES = ("base", "continue")


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_path(protocol, stage):
    return ROOT / "configs/yaml" / f"perception_no_spatial_{protocol}_{stage}.yaml"


def reference_path(stage):
    return ROOT / "configs/yaml" / f"perception_no_spatial_session_{stage}.yaml"


def load_config(protocol, stage):
    if protocol not in PROTOCOLS or stage not in STAGES:
        raise ValueError((protocol, stage))
    config = read_config(config_path(protocol, stage))
    reference = read_config(reference_path(stage))
    expected = copy.deepcopy(reference)
    expected["protocol"] = protocol
    # This historical audit pointer is not a model or training setting.
    expected.pop("matched_spatial_config", None)
    if config != expected:
        changed = [k for k in config.keys() | expected.keys() if config.get(k) != expected.get(k)]
        raise ValueError(f"Settings differ from the saved session-disjoint recipe: {changed}")
    return config


def output_path(config):
    return ROOT / config["output"].format(**config)


def split_contract(config):
    """Read public manifests only; never open private test labels during training."""
    protocol = config["protocol"]
    folder = ROOT / config["data"]["root"] / protocol
    paths = {"train": folder / "train.csv", "val": folder / "val.csv", "test": folder / "test_inputs.csv"}
    rows = {}
    for split, path in paths.items():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows[split] = list(csv.DictReader(stream))
    keys = ["record_uid", "content_group_id", "site_id" if protocol == "unseen_site" else "session_id"]
    for key in keys:
        groups = {split: {row[key] for row in records} for split, records in rows.items()}
        if any(not records or "" in groups[split] for split, records in rows.items()):
            raise ValueError(f"Missing {key} in {protocol} manifests")
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if groups[first] & groups[second]:
                raise ValueError(f"{protocol}: {key} overlaps between {first} and {second}")
    dates = {split: [min(r["detected_at"] for r in records), max(r["detected_at"] for r in records)]
             for split, records in rows.items()}
    if protocol == "forward_temporal" and not (dates["train"][1] < dates["val"][0] <= dates["val"][1] < dates["test"][0]):
        raise ValueError("Forward-temporal train/validation/test dates are not ordered")
    return {
        "protocol": protocol, "disjoint_keys": keys,
        "manifests": {split: {"sha256": sha256(path), "records_before_ontology_filter": len(rows[split]),
                              "date_range": dates[split]} for split, path in paths.items()},
        "negative_scope": "existing protocol-seeded group-disjoint no-event pool; no site/time metadata",
    }


def training_complete(config):
    output = output_path(config)
    if not output.exists():
        return False
    needed = ["config.yaml", "history.json", "protocol_recipe.json", "training_data.json",
              "best/adapter_config.json", "best/adapter_model.safetensors", "best/box_head.pt"]
    if any(not (output / name).is_file() for name in needed):
        raise RuntimeError(f"Incomplete training at {output}; optimizer resume is not supported. Preserve it before restarting.")
    history = json.loads((output / "history.json").read_text())
    if read_config(output / "config.yaml") != config or [r["epoch"] for r in history] != list(range(1, config["train"]["epochs"] + 1)):
        raise RuntimeError(f"Different recipe or unfinished epochs at {output}")
    if (output / "best/spatial_head.pt").exists():
        raise ValueError(f"Spatial checkpoint is not allowed: {output}")
    contract = json.loads((output / "protocol_recipe.json").read_text())["split_contract"]
    if contract != split_contract(config):
        raise ValueError(f"Split manifests changed since training: {output}")
    return True


def check_protocol(protocol):
    from clear_uav.table4 import read_discovery_samples, no_event_split, training_data_state
    base, continuation = (load_config(protocol, stage) for stage in STAGES)
    source = (ROOT / continuation["initial_checkpoint"].format(**continuation)).resolve()
    if source != (output_path(base) / "best").resolve():
        raise ValueError("Continuation must use this protocol's own base/best")
    contract = split_contract(base)
    counts = {}
    for split in ("train", "val"):
        samples = read_discovery_samples(base, protocol, split)
        counts[split] = {"positive": sum(s.presence for s in samples), "negative": sum(not s.presence for s in samples)}
    return {"protocol": protocol, "settings_match_session_disjoint": True, "spatial_module": False,
            "epochs": {stage: load_config(protocol, stage)["train"]["epochs"] for stage in STAGES},
            "records": counts, "negative_counts": {k: len(v) for k, v in no_event_split(base, protocol).items()},
            "split_contract": contract, "training_data": training_data_state(base, protocol),
            "outputs": {stage: str(output_path(load_config(protocol, stage))) for stage in STAGES}}


def train_stage(protocol, stage):
    import train_perception_qwen as baseline
    from clear_uav.table4 import training_data_state
    config = load_config(protocol, stage)
    if training_complete(config):
        if json.loads((output_path(config) / "training_data.json").read_text()) != training_data_state(config, protocol):
            raise ValueError("Training data changed; do not reuse this checkpoint")
        print(f"[reuse] {protocol}/{stage}: complete three-epoch training", flush=True)
        return
    receipt = {"stage": stage, "reference_config_sha256": sha256(reference_path(stage)),
               "split_contract": split_contract(config), "spatial_module": False,
               "source_sha256": {name: sha256(ROOT / "scripts" / name) for name in
                                  ("train_perception_qwen.py", "train_perception_continue.py", "perception_extension_runtime.py")}}
    builder = baseline.build_model
    if stage == "continue":
        from train_perception_continue import build_model, checkpoint_fingerprint
        base = load_config(protocol, "base")
        if not training_complete(base):
            raise ValueError("Complete this protocol's base training first")
        if json.loads((output_path(base) / "training_data.json").read_text()) != training_data_state(base, protocol):
            raise ValueError("Protocol base training data changed")
        source = (ROOT / config["initial_checkpoint"].format(**config)).resolve()
        if source != (output_path(base) / "best").resolve():
            raise ValueError("Wrong-protocol initialization")
        receipt["initial_checkpoint"] = str(source)
        receipt["initial_checkpoint_sha256"] = checkpoint_fingerprint(source)
        builder = build_model
    output = output_path(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol_recipe.json").write_text(json.dumps(receipt, indent=2))
    (output / "training_data.json").write_text(json.dumps(training_data_state(config, protocol), indent=2))
    original = baseline.build_model
    baseline.build_model = builder
    try:
        baseline.train(config)
    finally:
        baseline.build_model = original
    if not training_complete(config):
        raise RuntimeError(f"Training did not finish: {output}")


def evaluate_protocol(protocol, split):
    from clear_uav.table4 import training_data_state
    from test_perception_continue import evaluate
    config = load_config(protocol, "continue")
    if not training_complete(config):
        raise ValueError("Complete this protocol's base and continuation training first")
    if json.loads((output_path(config) / "training_data.json").read_text()) != training_data_state(config, protocol):
        raise ValueError("Training data changed since this checkpoint was trained")
    evaluate(config, split)
