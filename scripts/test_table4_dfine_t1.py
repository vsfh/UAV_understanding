#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import test_dfine


CONFIG = "configs/yaml/table4_dfine_t1.yaml"


if __name__ == "__main__":
    test_dfine(load_yaml(CONFIG))
