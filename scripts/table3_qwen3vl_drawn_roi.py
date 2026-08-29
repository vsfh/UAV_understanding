#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3 import ensure_hf_model, train_qwen


CONFIG = "configs/yaml/table3_qwen3vl_drawn_roi.yaml"


def main():
    config = load_yaml(CONFIG)
    ensure_hf_model(config["model"])
    train_qwen(config)


if __name__ == "__main__":
    main()
