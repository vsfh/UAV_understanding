#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/um7" "${REPO_ROOT}/hf_cache" "${REPO_ROOT}/outputs" "${REPO_ROOT}/results"

sshfs feihong@10.119.46.67:/feihong/uav_understanding_data \
  "./um7" \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

sshfs feihong@10.119.46.67:/feihong/hf_cache \
    "./hf_cache" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

sshfs feihong@10.119.46.67:/feihong/outputs \
    "./outputs" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
    
sshfs feihong@10.119.46.67:/feihong/results \
    "./results" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3