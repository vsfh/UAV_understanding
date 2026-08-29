#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3 import ensure_hf_model, train_florence


CONFIG = "configs/yaml/table3_florence2_gt_crop.yaml"


def main():
    config = load_yaml(CONFIG)
    ensure_hf_model(config["model"])
    train_florence(config)


if __name__ == "__main__":
    main()
