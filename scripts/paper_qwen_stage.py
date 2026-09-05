#!/usr/bin/env python3
"""Explicit-config adapter to the existing Qwen training/evaluation functions."""
import argparse
from clear_uav.experiment_config import load_yaml_with_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=["train", "test", "tiling-val", "tiling-test"], required=True)
    args = parser.parse_args()
    from clear_uav.table4 import (train_qwen_discovery, test_qwen_discovery,
                                 train_qwen_agent, test_qwen_agent)
    actions = {"train": train_qwen_discovery, "test": test_qwen_discovery,
               "tiling-val": train_qwen_agent, "tiling-test": test_qwen_agent}
    actions[args.action](load_yaml_with_base(args.config))
