#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3 import ensure_hf_model, train_encoder


CONFIG = "configs/yaml/table3_convnext_gt_crop.yaml"


def main():
    config = load_yaml(CONFIG)
    ensure_hf_model(config["model"])
    train_encoder(config)


if __name__ == "__main__":
    main()
