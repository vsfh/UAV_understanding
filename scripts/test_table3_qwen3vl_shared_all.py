#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table3_qwen_shared import test_qwen_shared


CONFIG = "configs/yaml/table3_qwen3vl_shared.yaml"


if __name__ == "__main__":
    test_qwen_shared(load_yaml(CONFIG))
