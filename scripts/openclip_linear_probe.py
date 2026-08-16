#!/usr/bin/env python3
from __future__ import annotations

import argparse

from clear_uav.experiment_config import load_yaml
from clear_uav.openclip_training import train_openclip


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only an OpenCLIP classification head")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_openclip(load_yaml(args.config), "linear_probe")


if __name__ == "__main__":
    main()
