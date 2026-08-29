#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3 import test_qwen


CONFIG = "configs/yaml/table3_qwen3vl_gt_crop.yaml"


if __name__ == "__main__":
    test_qwen(load_yaml(CONFIG))
