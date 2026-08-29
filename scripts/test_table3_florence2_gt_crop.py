#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3 import test_florence


CONFIG = "configs/yaml/table3_florence2_gt_crop.yaml"


if __name__ == "__main__":
    test_florence(load_yaml(CONFIG))
