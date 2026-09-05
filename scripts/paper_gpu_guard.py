#!/usr/bin/env python3
"""Fail closed if the explicitly selected GPU is occupied."""
import os
import subprocess


def check_gpu():
    gpu = os.environ.get("GPU_ID")
    if gpu is None or not gpu.isdigit():
        raise ValueError("Set GPU_ID to an explicit physical GPU index.")
    output = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip()
    memory, utilization = map(int, output.split(","))
    if memory > 512 or utilization > 5:
        raise ValueError(f"GPU {gpu} is occupied ({memory} MiB, {utilization}%). "
                         "Wait; do not terminate another user's process.")
    print(f"GPU {gpu} available: {memory} MiB, {utilization}%.")


if __name__ == "__main__":
    try:
        check_gpu()
    except (ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
