#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import test_florence_discovery


CONFIG = "configs/yaml/table4_florence2_t4.yaml"


if __name__ == "__main__":
    test_florence_discovery(load_yaml(CONFIG))
