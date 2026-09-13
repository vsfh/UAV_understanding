"""Evaluate the shared What / Where / Verify checkpoint on independent GPUs.

Launch with torchrun. Each rank loads one complete checkpoint once, then predicts
an unpadded shard in batches. Only rank zero selects validation thresholds and
writes the original metrics and traces. Agent decisions keep the same semantics.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from test_perception_agents import (
    CALIBRATION_VERSION, MODES, agent_metrics, evaluation_configuration,
    file_sha256, inference_signature, validate_calibration, write_traces,
)


DEFAULT_CONFIG = "configs/yaml/perception_agents_dual_pro6000.yaml"


def shard_indices(count, rank, world_size):
    """Unlike DistributedSampler, never pad validation/test with repeated images."""
    if count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid sample count, rank or world size")
    return list(range(rank, count, world_size))


def merge_predictions(shards, samples):
    """Restore dataset order and reject missing, repeated or mismatched records."""
    ordered = [None] * len(samples)
    seen = set()
    for shard in shards:
        rows = shard["rows"]
        for index, record_uid, prediction in rows:
            if type(index) is not int or not 0 <= index < len(samples):
                raise ValueError(f"Out-of-range prediction index: {index}")
            if index in seen:
                raise ValueError(f"Repeated prediction index: {index}")
            if record_uid != samples[index].record_uid:
                raise ValueError(f"Prediction record_uid differs at index {index}")
            seen.add(index)
            ordered[index] = prediction
    if len(seen) != len(samples):
        raise ValueError(f"Missing predictions: received {len(seen)} of {len(samples)}")
    return ordered


def predict_shard(runtime, samples, rank, world_size, mode, part, batch_size):
    from tqdm import tqdm

    if batch_size < 1:
        raise ValueError("test.batch_size must be positive")
    indices = shard_indices(len(samples), rank, world_size)
    rows = []
    with tqdm(total=len(indices), desc=f"rank {rank} agents {mode} {part}",
              position=rank, dynamic_ncols=True) as progress:
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            predictions = runtime.predict_batch([samples[index].image_path for index in batch], mode=mode)
            if len(predictions) != len(batch):
                raise ValueError("Agent runtime returned a different number of predictions")
            rows.extend((index, samples[index].record_uid, prediction)
                        for index, prediction in zip(batch, predictions))
            progress.update(len(batch))
    return rows


def source_fingerprints():
    """Keep the original calibration binding and include this merge/eval driver."""
    import train_perception_spatial, perception_spatial_head, train_perception_qwen
    from clear_uav import generation_constraints, table4

    fingerprints = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in (
            "perception_agents_core.py", "perception_agents_runtime.py",
            "perception_agents_batched_runtime.py",
            "perception_agents_dual_common.py",
            "test_perception_agents.py", "test_perception_agents_dual.py",
        )
    }
    for dependency in (
        train_perception_spatial, perception_spatial_head, train_perception_qwen,
        generation_constraints, table4,
    ):
        fingerprints[dependency.__name__] = file_sha256(dependency.__file__)
    return fingerprints


def save_evaluation(config, checkpoint, fingerprint, signature, mode, part,
                    samples, predictions, timing, threshold=None):
    """Rank-zero writer: preserve the original evaluator's metrics and outputs."""
    from clear_uav.table4 import definitions_from_config, save_results, select_threshold, table4_metrics
    from perception_agents_core import bbox_iou
    from perception_spatial_head import final_output_fpr
    from perception_agents_runtime import output_path

    labels, _ = definitions_from_config(config)
    destination = output_path(config) / "evaluation" / mode
    destination.mkdir(parents=True, exist_ok=True)
    if part == "val":
        threshold, selection = select_threshold(samples, predictions, config["validation"])
        limit = config.get("max_val_samples")
        calibration = {
            "schema_version": CALIBRATION_VERSION,
            "split": "val", "mode": mode, "threshold": threshold,
            "selection": selection, "policy": "max_recall_at_fpr",
            "max_n_fpr": config["validation"]["max_n_fpr"],
            "num_samples": len(samples), "sample_limit": limit,
            "limited_run": limit is not None or bool(config["data"].get("max_samples")),
            "protocol": config["protocol"], "seed": config["seed"],
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint,
            "agent_schema_sha256": file_sha256(checkpoint / "agent_schema.json"),
            "inference_signature": signature,
        }
        (destination / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    elif threshold is None:
        raise ValueError("Test requires a validated validation threshold")

    metrics = table4_metrics(samples, predictions, labels, threshold, classification=True)
    metrics["final_output_fpr"] = final_output_fpr(samples, predictions, threshold, labels)
    metrics["final_output_n_fpr"] = metrics["final_output_fpr"]
    metrics.update(agent_metrics(predictions))
    positive_rows = [(sample, prediction) for sample, prediction in zip(samples, predictions) if sample.presence]
    initial_hits, final_hits = [], []
    for sample, prediction in positive_rows:
        initial = prediction["trace"]["initial_hypothesis"]
        initial_hits.append(initial["category"] == sample.label and bbox_iou(initial["bbox_1000"], sample.bbox_1000) >= .5)
        final_hits.append(prediction["category"] == sample.label and bbox_iou(prediction["bbox_1000"], sample.bbox_1000) >= .5)
    metrics["agent_initial_joint50"] = sum(initial_hits) / len(positive_rows) if positive_rows else None
    metrics["agent_final_joint50"] = sum(final_hits) / len(positive_rows) if positive_rows else None
    metrics["agent_joint50_recovered"] = sum(not before and after for before, after in zip(initial_hits, final_hits))
    metrics["agent_joint50_lost"] = sum(before and not after for before, after in zip(initial_hits, final_hits))
    metrics["agent_mode"] = mode
    metrics.update(timing)
    config.setdefault("train", {})["seeds"] = [config["seed"]]
    result_predictions = [
        {key: value for key, value in prediction.items() if key != "trace"}
        for prediction in predictions
    ]
    save_results(destination / f"{part}_results.json", config, config["protocol"], samples,
                 result_predictions, metrics, seed=config["seed"])
    write_traces(destination / f"{part}_traces.jsonl", samples, predictions)
    print(json.dumps({"split": part, "mode": mode, **metrics}, indent=2), flush=True)


def evaluation_context(requested, split, mode):
    """Read and validate one authoritative checkpoint before allocating GPUs."""
    from perception_agents_runtime import output_path, read_config, SCHEMA_VERSION
    from perception_spatial_head import checkpoint_fingerprint
    from clear_uav.table4 import definitions_from_config

    output = output_path(requested)
    config = evaluation_configuration(read_config(output / "config.yaml"), requested)
    checkpoint = output / "best"
    for name in ("agent_schema.json", "box_head.pt", "spatial_head.pt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Missing trained agent checkpoint file: {checkpoint / name}")
    if json.loads((checkpoint / "agent_schema.json").read_text(encoding="utf-8"))["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported agent checkpoint schema")
    fingerprint = checkpoint_fingerprint(checkpoint)
    labels, definitions = definitions_from_config(config)
    sources = source_fingerprints()
    modes = MODES if mode == "all" else (mode,)
    signatures = {
        current: inference_signature(config, current, labels, definitions, sources)
        for current in modes
    }
    for name in ("max_val_samples", "max_test_samples"):
        if config.get(name) is not None and config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if (split == "all" and config.get("max_val_samples") is not None
            and config.get("max_test_samples") is None and not config["data"].get("max_samples")):
        raise ValueError("Limit both splits for a smoke run, or use full validation before full test")
    if split == "test":
        for current in modes:
            path = output / "evaluation" / current / "calibration.json"
            validate_calibration(json.loads(path.read_text(encoding="utf-8")), config,
                                 checkpoint, fingerprint, signatures[current], current)
    return config, str(checkpoint), fingerprint, labels, modes, signatures


def evaluate(requested, split="all", mode="full"):
    import torch
    import torch.distributed as dist
    from perception_agents_runtime import output_path
    from perception_agents_dual_common import build_fast_model
    from perception_agents_batched_runtime import BatchedAgentRuntime
    from clear_uav.table4 import read_discovery_samples

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Models need no DDP wrapper in inference. Gloo gathers CPU prediction objects
    # without putting pickled traces in GPU memory or introducing NCCL padding.
    if world_size > 1:
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    try:
        context = [evaluation_context(requested, split, mode) if rank == 0 else None]
        if world_size > 1:
            dist.broadcast_object_list(context, src=0)
        config, checkpoint_string, fingerprint, labels, modes, signatures = context[0]
        checkpoint = Path(checkpoint_string)
        config["device"] = device
        load_started = time.perf_counter()
        model, processor = build_fast_model(config, checkpoint=checkpoint, trainable=False)
        runtime = BatchedAgentRuntime(model, processor, config, labels)
        batch_size = int(config.get("test", {}).get("batch_size", 4))
        print(json.dumps({"rank": rank, "world_size": world_size, "device": device,
                          "checkpoint_loaded_seconds": round(time.perf_counter() - load_started, 3)},
                         ensure_ascii=False), flush=True)

        for current in modes:
            for part in (("val", "test") if split == "all" else (split,)):
                threshold = None
                if part == "test" and rank == 0:
                    path = output_path(config) / "evaluation" / current / "calibration.json"
                    threshold = validate_calibration(
                        json.loads(path.read_text(encoding="utf-8")), config,
                        checkpoint, fingerprint, signatures[current], current,
                    )
                samples = read_discovery_samples(config, config["protocol"], part)
                limit = config.get(f"max_{part}_samples")
                samples = samples[:limit] if limit is not None else samples
                if not samples:
                    raise ValueError(f"Empty {part} split")
                if world_size > 1:
                    dist.barrier()
                started = time.perf_counter()
                rows = predict_shard(runtime, samples, rank, world_size, current, part, batch_size)
                elapsed = time.perf_counter() - started
                local = {"rank": rank, "rows": rows, "elapsed_seconds": elapsed}
                print(json.dumps({"rank": rank, "mode": current, "split": part,
                                  "num_samples": len(rows), "elapsed_seconds": round(elapsed, 3)},
                                 ensure_ascii=False), flush=True)
                shards = [None] * world_size if rank == 0 else None
                if world_size > 1:
                    dist.gather_object(local, object_gather_list=shards, dst=0)
                else:
                    shards = [local]
                if rank == 0:
                    predictions = merge_predictions(shards, samples)
                    split_elapsed = time.perf_counter() - started
                    timing = {
                        "parallel_world_size": world_size,
                        "parallel_batch_size_per_rank": batch_size,
                        "parallel_split_elapsed_s": split_elapsed,
                        "parallel_images_per_second": len(samples) / split_elapsed,
                        "parallel_rank_elapsed_s": [shard["elapsed_seconds"] for shard in shards],
                        "parallel_rank_num_samples": [len(shard["rows"]) for shard in shards],
                        "parallel_timing_scope": "split_prediction_and_gather_excluding_model_load_and_metrics",
                    }
                    save_evaluation(config, checkpoint, fingerprint, signatures[current], current, part,
                                    samples, predictions, timing, threshold)
                if world_size > 1:
                    # Complete rank-zero validation calibration before any test
                    # shard starts; also prevent interleaved result writers.
                    dist.barrier()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="full")
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    from perception_agents_runtime import read_config
    requested = read_config(args.config)
    for name in ("output", "max_val_samples", "max_test_samples"):
        if getattr(args, name) is not None:
            requested[name] = getattr(args, name)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        requested.setdefault('test', {})['batch_size'] = args.batch_size
    evaluate(requested, args.split, args.mode)


if __name__ == "__main__":
    main()
