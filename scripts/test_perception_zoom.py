"""Validate/calibrate or test the saved active-review Perception-Qwen checkpoint."""
from train_perception_zoom import main

if __name__ == "__main__":
    main(default_mode="evaluate")
