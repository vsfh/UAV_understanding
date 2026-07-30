#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Transformers training run")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.output_dir / "trainer_state.json"
    if not state_path.is_file():
        checkpoints = sorted(
            args.output_dir.glob("checkpoint-*/trainer_state.json"),
            key=lambda path: int(path.parent.name.removeprefix("checkpoint-")),
        )
        state_path = checkpoints[-1]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    loss_rows = [row for row in state["log_history"] if "loss" in row]
    result = {
        "state_path": str(state_path),
        "global_step": state["global_step"],
        "max_steps": state["max_steps"],
        "epoch": state["epoch"],
        "first_logged_loss": loss_rows[0]["loss"],
        "last_logged_loss": loss_rows[-1]["loss"],
        "minimum_logged_loss": min(row["loss"] for row in loss_rows),
        "last_learning_rate": loss_rows[-1]["learning_rate"],
    }
    metrics_path = args.output_dir / "train_results.json"
    if metrics_path.is_file():
        result["train_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
