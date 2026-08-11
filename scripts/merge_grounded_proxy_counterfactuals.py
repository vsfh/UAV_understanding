#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            uid = row["record_uid"]
            if uid in rows:
                raise ValueError(f"Duplicate record_uid in {path}:{line_number}: {uid}")
            rows[uid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use crop captions as grounded positive targets and attach proxy "
            "counterfactuals for development CLEAR runs"
        )
    )
    parser.add_argument("--grounded-targets", type=Path, required=True)
    parser.add_argument("--proxy-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grounded = read_jsonl(args.grounded_targets)
    proxy = read_jsonl(args.proxy_targets)
    if set(grounded) != set(proxy):
        grounded_only = sorted(set(grounded) - set(proxy))
        proxy_only = sorted(set(proxy) - set(grounded))
        raise ValueError(
            "Grounded/proxy target coverage differs: "
            f"grounded_only={grounded_only[:1]}, proxy_only={proxy_only[:1]}"
        )

    merged = []
    for uid in tqdm(
        grounded,
        desc="merge grounded + proxy CF",
        unit="target",
        dynamic_ncols=True,
    ):
        grounded_row = grounded[uid]
        proxy_row = proxy[uid]
        target = grounded_row.get("target")
        counterfactual = proxy_row.get("counterfactual_target")
        if not isinstance(target, dict) or not target.get("events"):
            raise ValueError(f"Invalid grounded target for {uid}")
        if not isinstance(counterfactual, dict) or not counterfactual.get("events"):
            raise ValueError(f"Missing proxy counterfactual for {uid}")
        if set(target["events"]) & set(counterfactual["events"]):
            raise ValueError(f"Counterfactual repeats a positive event for {uid}")
        merged.append(
            {
                "record_uid": uid,
                "source_caption": grounded_row.get("source_caption"),
                "caption_check": grounded_row.get("caption_check"),
                "target": target,
                "counterfactual_target": counterfactual,
                "supervision_tier": "teacher_crop_grounded_with_proxy_counterfactual",
                "target_provenance": {
                    "positive": str(args.grounded_targets),
                    "counterfactual": str(args.proxy_targets),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(merged)} grounded+proxy-counterfactual targets to {args.output}")


if __name__ == "__main__":
    main()
