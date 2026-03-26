from __future__ import annotations

import argparse
import json

from src.models.realtime_train import train_realtime_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train robust live-safe leak models.")
    parser.add_argument(
        "--config",
        default="config/robust_training.yaml",
        help="Path to the robust training config.",
    )
    args = parser.parse_args()
    result = train_realtime_models(args.config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
