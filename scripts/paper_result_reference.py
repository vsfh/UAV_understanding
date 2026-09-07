"""Model-free expected evaluation rows from frozen benchmark source artifacts.

Mirrors table4.read_discovery_samples / no_event_split, without importing torch.
Never derives membership or targets from experiment predictions. An optional
offline inventory is a JSON list, or an object with ``filenames`` (alternatively
``negative_filenames``), containing all no-event PNG basenames before splitting.
"""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


PROTOCOLS = ("session_disjoint", "unseen_site", "forward_temporal")
NO_EVENT_TIMESTAMP = re.compile(r"photo - (\d{4}-\d{2}-\d{2}T\d{6}\.\d+)\.png$")


def project_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def load_config(root, value="configs/yaml/table4_qwen3vl_t4.yaml", active=()):
    path = project_path(root, value).resolve()
    if path in active:
        raise ValueError(f"Cyclic base_config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid source configuration: {path}")
    base = raw.pop("base_config", None)
    result = load_config(root, base, (*active, path)) if base else {}

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(result, raw)
    return result


def read_csv_unique(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("record_uid") for row in rows]
    if not rows or any(not uid for uid in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Empty/missing/duplicate source record IDs: {path}")
    return rows


def labels_from_config(root, config):
    labels_path = project_path(root, config["data"]["labels"])
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8-sig").splitlines()
              if line.strip() and not line.startswith("#")]
    ontology_path = project_path(root, config["data"]["ontology"])
    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    names = [event["name"] for event in ontology["events"]]
    if not labels or len(labels) != len(set(labels)) or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate source ontology labels")
    if set(labels) - set(names):
        raise ValueError("Label subset contains names outside source ontology")
    return labels


def bbox_targets(path):
    coco = json.loads(path.read_text(encoding="utf-8"))
    images = {}
    filenames = set()
    for row in coco["images"]:
        if row["id"] in images or row["file_name"] in filenames:
            raise ValueError(f"Duplicate COCO image identity: {path}")
        images[row["id"]] = row
        filenames.add(row["file_name"])
    boxes = {}
    for annotation in coco["annotations"]:
        image = images.get(annotation["image_id"])
        if image is None:
            raise ValueError(f"COCO annotation has no image: {path}")
        name = image["file_name"]
        if name in boxes:
            raise ValueError(f"Multiple contextual targets for one image: {name}")
        values = [image["width"], image["height"], *annotation["bbox"]]
        if len(values) != 6 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError(f"Invalid COCO target numbers: {name}")
        image_w, image_h, x, y, width, height = values
        if min(image_w, image_h, width, height) <= 0:
            raise ValueError(f"Invalid COCO target dimensions: {name}")
        boxes[name] = [1000*x/image_w, 1000*y/image_h,
                       1000*(x+width)/image_w, 1000*(y+height)/image_h]
    return boxes


def negative_filenames(root, settings, inventory=None):
    if inventory is None:
        directory = project_path(root, settings["root"])
        if not directory.is_dir():
            raise FileNotFoundError(f"No-event source directory missing: {directory}; "
                                    "use --negative-inventory only for an offline snapshot")
        names = sorted(path.name for path in directory.glob("*.png") if path.is_file())
        source = {"kind": "directory", "path": directory.as_posix()}
    else:
        path = project_path(root, inventory)
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = payload if isinstance(payload, list) else payload.get("filenames", payload.get("negative_filenames"))
        if not isinstance(names, list):
            raise ValueError("Negative inventory requires a list of PNG filenames")
        source = {"kind": "offline_filename_inventory", "path": path.as_posix(),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        expected_hash = payload.get("filenames_sha256") if isinstance(payload, dict) else None
        if expected_hash and all(isinstance(n, str) for n in names):
            actual_hash = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("Negative inventory filenames_sha256 mismatch")
    if (not names or any(not isinstance(n, str) or not n.endswith(".png")
                         or "/" in n or "\\" in n or "\0" in n or "\n" in n or "\r" in n
                         for n in names)):
        raise ValueError("Negative inventory must contain nonempty PNG basenames only")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate negative inventory filename")
    names = sorted(names)
    source.update(count=len(names), filenames_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest())
    return names, source


def negative_test_rows(names, settings, protocol):
    """Same timestamp grouping and size-first deficit assignment as table4.py."""
    gap = settings["group_gap_seconds"]
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0:
        raise ValueError("Invalid negative group time gap")
    timestamped, untimestamped = [], []
    for name in sorted(names):
        match = NO_EVENT_TIMESTAMP.fullmatch(name)
        if match:
            timestamped.append((datetime.strptime(match.group(1), "%Y-%m-%dT%H%M%S.%f"), name))
        else:
            untimestamped.append(name)
    groups, current, previous = [], [], None
    for captured_at, name in sorted(timestamped):
        if previous is not None and (captured_at - previous).total_seconds() > gap:
            groups.append(current)
            current = []
        current.append(name)
        previous = captured_at
    if current:
        groups.append(current)
    if untimestamped:
        groups.append(untimestamped)
    splits = ("train", "val", "test")
    ratios = settings["split_ratios"]
    if any(not isinstance(ratios.get(s), (int, float)) or not math.isfinite(ratios[s]) or ratios[s] <= 0 for s in splits):
        raise ValueError("Negative split ratios must be positive finite values")
    targets = {s: len(names)*ratios[s]/sum(ratios[n] for n in splits) for s in splits}
    counts = dict.fromkeys(splits, 0)
    seed = settings.get("seed", 43)

    def digest(group):
        return hashlib.sha256(f"{seed}:{protocol}:".encode()+"\n".join(group).encode()).digest()

    result = []
    for group in sorted(groups, key=lambda g: (-len(g), digest(g))):
        split = max(splits, key=lambda s: (targets[s]-counts[s])/targets[s])
        counts[split] += len(group)
        if split != "test":
            continue
        group_id = "neggrp_" + hashlib.sha256("\n".join(group).encode()).hexdigest()[:16]
        for name in group:
            result.append({"record_uid": "neg_"+hashlib.sha256(name.encode()).hexdigest()[:20],
                           "group_id": group_id,
                           "target": {"presence": False, "category": None, "bbox_1000": None}})
    return result


def build_expected_rows(root, protocol, negative_inventory=None, config=None):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    root = Path(root)
    config = config or load_config(root)
    data = config["data"]
    if data.get("max_samples") or data.get("max_test_samples"):
        raise ValueError("Paper audit requires the full frozen test population")
    labels = labels_from_config(root, config)
    data_root = project_path(root, data["root"])
    inputs_path = data_root/protocol/"test_inputs.csv"
    labels_path = data_root/protocol/"test_labels_private.csv"
    boxes_path = project_path(root, data["bbox_annotations"])
    inputs = read_csv_unique(inputs_path)
    private_rows = read_csv_unique(labels_path)
    private = {r["record_uid"]: r for r in private_rows}
    if {r["record_uid"] for r in inputs} != set(private):
        raise ValueError(f"Frozen test inputs/private labels do not have identical IDs: {protocol}")
    boxes = bbox_targets(boxes_path)
    result, excluded = [], 0
    for row in inputs:
        label_row = private[row["record_uid"]]
        label = label_row.get("source_class") or None
        joined = label_row | row  # Same precedence as read_discovery_samples.
        presence_value = joined.get("presence", "")
        presence = (presence_value.strip().lower() in {"1", "true", "yes", "positive"}
                    if presence_value != "" else bool(label and label != data.get("negative_label", "no_event")))
        if presence and label not in labels:
            excluded += 1
            continue
        image_name = row["context_path"].removeprefix("data/")
        if presence and image_name not in boxes:
            raise ValueError(f"Missing source COCO bbox: {row['record_uid']}")
        result.append({"record_uid": row["record_uid"], "group_id": row.get("content_group_id") or None,
                       "target": {"presence": presence, "category": label if presence else None,
                                  "bbox_1000": boxes[image_name] if presence else None}})
    inventory_source = None
    if data.get("no_event"):
        names, inventory_source = negative_filenames(root, data["no_event"], negative_inventory)
        result.extend(negative_test_rows(names, data["no_event"], protocol))
    ids = [r["record_uid"] for r in result]
    if not result or len(ids) != len(set(ids)):
        raise ValueError(f"Empty/duplicate independent reference IDs: {protocol}")
    paths = [inputs_path, labels_path, boxes_path,
             project_path(root, data["labels"]), project_path(root, data["ontology"])]
    return {"rows": result, "labels": labels,
            "provenance": {"records": len(result), "positive_records": sum(r["target"]["presence"] for r in result),
                           "negative_records": sum(not r["target"]["presence"] for r in result),
                           "excluded_outside_label_subset": excluded,
                           "uid_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                           "negative_inventory": inventory_source,
                           "sources": [{"path": p.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}}
