#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
source runs/perception_runtime.sh
perception_resolve_python
"$python_bin" scripts/run_perception_no_spatial.py --protocol forward_temporal "$@"
