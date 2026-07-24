"""
ATMS — Automated Traffic Management System
Entry point.

Usage:
    python main.py
    python main.py --config path/to/.env
"""

import argparse
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATMS — Automated Traffic Management System")
    parser.add_argument("--config", default=".env", help="Path to .env config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.config)

    # TODO Phase 5: import and start pipeline.manager.Manager here
    print("ATMS starting — build out src/ modules first (see CLAUDE.md)")


if __name__ == "__main__":
    main()
