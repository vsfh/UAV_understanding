#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3 import test_encoder


CONFIG = "configs/yaml/table3_convnext_gt_crop.yaml"


if __name__ == "__main__":
    test_encoder(load_yaml(CONFIG))
