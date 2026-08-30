#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import train_grounding_dino


CONFIG = "configs/yaml/table4_grounding_dino_t1.yaml"


if __name__ == "__main__":
    train_grounding_dino(load_yaml(CONFIG))
