#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("forward_temporal", "session_disjoint", "unseen_site")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class GpuProfile:
    index: int
    name: str
    total_mib: int
    zero_batch: int
    linear_batch: int
    full_batch: int
    full_accumulation: int
    minimum_free_mib: int


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    protocol: str
    seed: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribute independent OpenCLIP experiment shards over heterogeneous GPUs"
    )
    parser.add_argument("--gpus", type=int, nargs="+", help="Physical GPU indices; default: all")
    parser.add_argument("--data-root", type=Path, default=Path("um7"))
    parser.add_argument("--models-root", type=Path, default=Path("hf_cache"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/openclip_multi_gpu")
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/openclip_multi_gpu")
    )
    parser.add_argument("--labels-file", type=Path, default=Path("configs/core18_complete.txt"))
    parser.add_argument("--protocols", choices=PROTOCOLS, nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def memory_profile(index: int, name: str, total_mib: int) -> GpuProfile:
    if total_mib >= 20_000:
        settings = (16, 32, 4, 4, min(18_000, total_mib - 2_048))
    elif total_mib >= 14_000:
        settings = (12, 24, 2, 8, total_mib - 2_048)
    elif total_mib >= 10_000:
        settings = (8, 16, 1, 16, total_mib - 1_536)
    elif total_mib >= 8_000:
        settings = (4, 8, 1, 16, total_mib - 1_024)
    else:
        raise ValueError(
            f"GPU {index} ({name}) has only {total_mib} MiB; at least 8 GiB is required"
        )
    return GpuProfile(index, name, total_mib, *settings)


def detect_gpus(selected: list[int] | None) -> list[GpuProfile]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = {}
    for line in result.stdout.splitlines():
        index_text, name, memory_text = [part.strip() for part in line.split(",", 2)]
        inventory[int(index_text)] = (name, int(memory_text))
    indices = selected if selected is not None else sorted(inventory)
    if len(indices) != len(set(indices)):
        raise ValueError("--gpus contains duplicate indices")
    missing = [index for index in indices if index not in inventory]
    if missing:
        raise ValueError(f"Unknown GPU indices: {missing}; available={sorted(inventory)}")
    return [memory_profile(index, *inventory[index]) for index in indices]


def jobs_for(protocols: list[str], seeds: list[int]) -> list[Job]:
    finetuning = [
        Job(f"finetune.{protocol}.seed{seed}", "finetune", protocol, seed)
        for protocol in protocols
        for seed in seeds
    ]
    zero_shot = [Job(f"zero.{protocol}", "zero", protocol) for protocol in protocols]
    return [*finetuning, *zero_shot]


def command_for(
    job: Job,
    gpu: GpuProfile,
    *,
    args: argparse.Namespace,
    data_root: Path,
    models_root: Path,
    output_root: Path,
    results_root: Path,
    labels_file: Path,
) -> list[str]:
    shard_root = (
        results_root / "shards" / "zero" / job.protocol
        if job.kind == "zero"
        else results_root
        / "shards"
        / "finetune"
        / job.protocol
        / f"seed{job.seed}"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment_suite.py"),
        "--profile",
        "development",
        "--data-root",
        str(data_root),
        "--models-root",
        str(models_root),
        "--output-root",
        str(output_root),
        "--results-root",
        str(shard_root),
        "--paper-tables-dir",
        str(shard_root / "paper_tables"),
        "--labels-file",
        str(labels_file),
        "--protocols",
        job.protocol,
        "--zero-shot-only",
        "--zero-shot-models",
        "openclip",
        "--openclip-batch-size",
        str(gpu.zero_batch),
        "--openclip-linear-batch-size",
        str(gpu.linear_batch),
        "--openclip-full-batch-size",
        str(gpu.full_batch),
        "--openclip-full-gradient-accumulation",
        str(gpu.full_accumulation),
        "--openclip-num-workers",
        str(args.num_workers),
        "--cuda-devices",
        str(gpu.index),
        "--min-free-gpu-mib",
        str(gpu.minimum_free_mib),
        "--resume",
    ]
    if job.kind == "finetune":
        command.extend(
            [
                "--seeds",
                str(job.seed),
                "--skip-zero-shot",
                "--openclip-finetuning",
                "--openclip-linear-epochs",
                "20",
                "--openclip-linear-learning-rate",
                "1e-3",
                "--openclip-full-epochs",
                "3",
                "--openclip-full-learning-rate",
                "5e-4",
                "--openclip-backbone-learning-rate",
                "1e-5",
            ]
        )
    return command


def atomic_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def readable_status(text: str, maximum: int = 180) -> str:
    cleaned = ANSI_ESCAPE.sub("", text).replace("\x1b", "").strip()
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > maximum:
        return "…" + cleaned[-(maximum - 1) :]
    return cleaned


def stream_process(
    process: subprocess.Popen,
    log,
    *,
    on_status,
) -> int:
    """Copy complete child output to a log and expose CR-based tqdm updates."""
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        log.write(chunk)
        log.flush()
        pending += decoder.decode(chunk)
        records = re.split(r"[\r\n]+", pending)
        pending = records.pop()
        for record in records:
            status = readable_status(record)
            if status:
                on_status(status)
    pending += decoder.decode(b"", final=True)
    status = readable_status(pending)
    if status:
        on_status(status)
    return process.wait()


def main() -> None:
    args = parse_args()
    profiles = detect_gpus(args.gpus)
    if not profiles:
        raise RuntimeError("No GPUs selected")
    data_root = resolve(args.data_root)
    models_root = resolve(args.models_root)
    output_root = resolve(args.output_root)
    results_root = resolve(args.results_root)
    labels_file = resolve(args.labels_file)
    jobs = jobs_for(args.protocols, list(dict.fromkeys(args.seeds)))

    print("GPU profiles:")
    for profile in profiles:
        print(json.dumps(asdict(profile), ensure_ascii=False))
    print(
        f"Selected {len(profiles)} GPUs, total physical memory "
        f"{sum(profile.total_mib for profile in profiles)} MiB, jobs={len(jobs)}"
    )
    print(
        "Memory is not pooled: every job must fit one GPU; independent jobs run in parallel."
    )
    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = profiles[index % len(profiles)]
            print(f"[{job.job_id} -> GPU {gpu.index}] " + " ".join(command_for(
                job,
                gpu,
                args=args,
                data_root=data_root,
                models_root=models_root,
                output_root=output_root,
                results_root=results_root,
                labels_file=labels_file,
            )))
        return

    required = [data_root, models_root / "openclip" / "config.json", labels_file]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n- " + "\n- ".join(missing))

    work_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work_queue.put(job)
    statuses = {job.job_id: "pending" for job in jobs}
    gpu_runtime = {
        profile.index: {"job": "idle", "latest": "waiting", "started": None}
        for profile in profiles
    }
    failures: list[tuple[str, int]] = []
    lock = threading.Lock()
    stop = threading.Event()
    status_path = results_root / "multi_gpu_status.json"
    logs_root = results_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    bar = tqdm(
        total=len(jobs),
        desc="OpenCLIP jobs",
        unit="job",
        dynamic_ncols=True,
        position=0,
    )
    gpu_bars = {
        profile.index: tqdm(
            total=0,
            desc=f"GPU {profile.index} idle",
            bar_format="{desc} | {postfix}",
            dynamic_ncols=True,
            position=position,
            leave=True,
        )
        for position, profile in enumerate(profiles, 1)
    }

    try:

        def worker(gpu: GpuProfile) -> None:
            while not stop.is_set():
                try:
                    job = work_queue.get_nowait()
                except queue.Empty:
                    return
                command = command_for(
                    job,
                    gpu,
                    args=args,
                    data_root=data_root,
                    models_root=models_root,
                    output_root=output_root,
                    results_root=results_root,
                    labels_file=labels_file,
                )
                with lock:
                    statuses[job.job_id] = f"running_gpu_{gpu.index}"
                    gpu_runtime[gpu.index] = {
                        "job": job.job_id,
                        "latest": "starting child process",
                        "started": time.monotonic(),
                    }
                    atomic_status(status_path, statuses)
                    tqdm.write(f"[START GPU {gpu.index}] {job.job_id}")
                log_path = logs_root / f"{job.job_id}.log"
                env = os.environ.copy()
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=False,
                        bufsize=0,
                    )

                    def update_child_status(status: str) -> None:
                        with lock:
                            gpu_runtime[gpu.index]["latest"] = status

                    return_code = stream_process(
                        process, log, on_status=update_child_status
                    )
                with lock:
                    if return_code:
                        statuses[job.job_id] = f"failed_gpu_{gpu.index}"
                        failures.append((job.job_id, return_code))
                        stop.set()
                        tqdm.write(
                            f"[FAILED GPU {gpu.index}] {job.job_id}; log={log_path}"
                        )
                    else:
                        statuses[job.job_id] = f"completed_gpu_{gpu.index}"
                        tqdm.write(f"[DONE GPU {gpu.index}] {job.job_id}")
                    gpu_runtime[gpu.index] = {
                        "job": "idle",
                        "latest": "waiting for next shard",
                        "started": None,
                    }
                    atomic_status(status_path, statuses)
                    bar.update(1)
                work_queue.task_done()

        threads = [
            threading.Thread(target=worker, args=(profile,), daemon=False)
            for profile in profiles
        ]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            with lock:
                snapshot = {
                    index: dict(runtime) for index, runtime in gpu_runtime.items()
                }
            now = time.monotonic()
            for profile in profiles:
                runtime = snapshot[profile.index]
                started = runtime["started"]
                elapsed = int(now - started) if isinstance(started, float) else 0
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                gpu_bar = gpu_bars[profile.index]
                gpu_bar.set_description_str(
                    f"GPU {profile.index} {runtime['job']} {hours:02d}:{minutes:02d}:{seconds:02d}"
                )
                gpu_bar.set_postfix_str(runtime["latest"], refresh=True)
            for thread in threads:
                thread.join(timeout=0.2)
            bar.refresh()
    finally:
        bar.close()
        for gpu_bar in gpu_bars.values():
            gpu_bar.close()

    if failures:
        raise subprocess.CalledProcessError(failures[0][1], failures[0][0])

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "merge_openclip_shards.py"),
            "--shards-root",
            str(results_root / "shards"),
            "--output-dir",
            str(results_root),
            "--expected-seeds",
            *[str(seed) for seed in args.seeds],
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"Completed multi-GPU suite: {results_root / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
