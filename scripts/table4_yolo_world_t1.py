#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import train_yolo_world


CONFIG = "configs/yaml/table4_yolo_world_t1.yaml"


if __name__ == "__main__":
    train_yolo_world(load_yaml(CONFIG))
