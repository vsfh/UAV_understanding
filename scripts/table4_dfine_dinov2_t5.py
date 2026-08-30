#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import train_dfine_dinov2


CONFIG = "configs/yaml/table4_dfine_dinov2_t5.yaml"


if __name__ == "__main__":
    train_dfine_dinov2(load_yaml(CONFIG))
