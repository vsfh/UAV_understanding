#!/usr/bin/env python3
from clear_uav.experiment_config import load_yaml
from clear_uav.table4 import test_qwen_agent


CONFIG = "configs/yaml/table4_qwen3vl_tool_agent.yaml"


if __name__ == "__main__":
    test_qwen_agent(load_yaml(CONFIG))
