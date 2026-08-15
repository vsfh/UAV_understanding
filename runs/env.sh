#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/um7" "${REPO_ROOT}/hf_cache" "${REPO_ROOT}/conda"

sshfs feihong@10.119.46.67:/feihong/uav_understanding_data \
  "${REPO_ROOT}/um7" \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

sshfs feihong@10.119.46.67:/feihong/hf_cache \
    "${REPO_ROOT}/hf_cache" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

sshfs feihong@10.119.46.67:/feihong/miniconda3 \
    "${REPO_ROOT}/conda" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

sshfs feihong@10.119.46.67:/feihong/output \
    "${REPO_ROOT}/output" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
    
sshfs feihong@10.119.46.67:/feihong/result \
    "${REPO_ROOT}/result" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3