from __future__ import annotations

import json
from pathlib import Path


def read_targets(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    targets: dict[str, str] = {}
    counterfactuals: dict[str, str] = {}
    tiers: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = row["record_uid"]
            target = row["target"]
            counterfactual = row.get("counterfactual_target")
            tiers[uid] = row.get("supervision_tier", "unspecified")
            targets[uid] = json.dumps(
                json.loads(target) if isinstance(target, str) else target,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if counterfactual is not None:
                counterfactuals[uid] = json.dumps(
                    (
                        json.loads(counterfactual)
                        if isinstance(counterfactual, str)
                        else counterfactual
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
    return targets, counterfactuals, tiers
