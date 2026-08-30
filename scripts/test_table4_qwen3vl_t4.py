#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import test_qwen_discovery


CONFIG = "configs/yaml/table4_qwen3vl_t4.yaml"


if __name__ == "__main__":
    test_qwen_discovery(load_yaml(CONFIG))
