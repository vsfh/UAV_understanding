#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import test_best_localizer_qwen


CONFIG = "configs/yaml/table4_best_localizer_qwen3vl_t5.yaml"


if __name__ == "__main__":
    test_best_localizer_qwen(load_yaml(CONFIG))
