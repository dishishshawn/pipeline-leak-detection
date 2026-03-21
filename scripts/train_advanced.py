#!/usr/bin/env python3
"""
Advanced Training Runner
Run the scalable training pipeline with configuration.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.train import advanced_train

if __name__ == "__main__":
    config_path = "config/training_config.yaml"
    advanced_train(config_path)