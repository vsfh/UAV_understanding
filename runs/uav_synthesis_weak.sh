#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "${BASH_SOURCE[0]%/*}/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-$repo_root/.venvs/uav_synthesis/bin/python}"
stage="${1:-plan}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$stage" in
    plan|prepare|all|status|export)
        exec "$python_bin" scripts/uav_synthesis_weak.py \
            --config configs/yaml/uav_synthesis_weak.yaml \
            --stage "$stage" "$@"
        ;;
    smoke)
        exec "$python_bin" scripts/uav_synthesis_weak.py \
            --config configs/yaml/uav_synthesis_weak.yaml \
            --stage all --per-class 2 --chunk-size 10 \
            --selection-dir outputs/uav_synthesis_weak_smoke/seed43 \
            --synthetic-root um7/synthetic_weak_smoke "$@"
        ;;
    *)
        printf 'Usage: bash runs/uav_synthesis_weak.sh {plan|prepare|all|status|export|smoke} [options]\n' >&2
        exit 2
        ;;
esac
