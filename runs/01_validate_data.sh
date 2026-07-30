#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/media/data1/feihong/uav_understanding_data}"
for protocol in forward_temporal session_disjoint unseen_site; do
  python scripts/validate_split.py --data-root "${DATA_ROOT}" --protocol "${protocol}"
done

