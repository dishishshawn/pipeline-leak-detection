#!/usr/bin/env python3
r"""
Train realtime-oriented SCADA models for live monitoring and simulator scoring.

Usage:
  venv\Scripts\python.exe scripts\train_realtime_models.py
  venv\Scripts\python.exe scripts\train_realtime_models.py --config config/realtime_training.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.realtime_train import train_realtime_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train realtime SCADA monitoring models")
    parser.add_argument(
        "--config",
        default="config/realtime_training.yaml",
        help="Path to the realtime training config file",
    )
    args = parser.parse_args()

    result = train_realtime_models(args.config)
    print(json.dumps(result["saved_paths"], indent=2))
    print(f"summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
