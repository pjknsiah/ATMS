"""
ATMS — Automated Traffic Management System
Entry point.

Usage:
    python main.py
    python main.py --config path/to/.env
"""

import argparse
from dotenv import load_dotenv

from src.utils.logger import get_logger, setup_logging
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATMS — Automated Traffic Management System")
    parser.add_argument("--config", default=".env", help="Path to .env config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.config)
    setup_logging()
    log = get_logger(__name__)
    config = load_config()
    log.info("atms_starting", lane_count=config.lane_count, window_seconds=config.window_seconds)
    # TODO Phase 5: import and start pipeline.manager.Manager here


if __name__ == "__main__":
    main()
