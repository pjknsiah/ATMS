"""
ATMS — Automated Traffic Management System
Entry point.

Usage:
    python main.py
    python main.py --config path/to/.env
    python main.py --config path/to/.env --dry-run

Startup sequence:
1. Parse CLI args
2. Load .env config
3. Set up structured logging
4. Validate video files exist
5. Start PipelineManager (blocks until Ctrl-C)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.manager import PipelineManager
from src.utils.config import Config, load_config
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace with fields:
            config   (str)  — path to the .env file to load
            dry_run  (bool) — if True, validate config and exit without running
    """
    parser = argparse.ArgumentParser(
        description="ATMS — Automated Traffic Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --config /etc/atms/.env
  python main.py --dry-run          # validate config + videos, then exit
        """,
    )
    parser.add_argument(
        "--config",
        default=".env",
        metavar="PATH",
        help="Path to .env config file (default: .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and video files, then exit without starting the pipeline",
    )
    return parser.parse_args()


def _validate_videos(config: Config, log) -> list[Path]:
    """
    Check that all expected lane video files exist under config.video_dir.

    Args:
        config: Loaded Config dataclass.
        log: Bound structlog logger.

    Returns:
        List of resolved Path objects (one per lane).

    Raises:
        SystemExit: If any expected video file is missing.
    """
    video_dir = Path(config.video_dir)
    missing: list[str] = []
    paths: list[Path] = []

    for lane_id in range(config.lane_count):
        p = video_dir / f"lane_{lane_id}.mp4"
        if p.exists():
            paths.append(p)
            log.debug("video_found", lane_id=lane_id, path=str(p))
        else:
            missing.append(str(p))
            log.error("video_missing", lane_id=lane_id, path=str(p))

    if missing:
        log.error(
            "startup_aborted",
            reason="missing_video_files",
            missing_count=len(missing),
            missing=missing,
        )
        sys.exit(1)

    return paths


def _validate_weights(config: Config, log) -> None:
    """
    Warn if the YOLO weights file is absent (Ultralytics auto-downloads on
    first use, so this is advisory only).

    Args:
        config: Loaded Config dataclass.
        log: Bound structlog logger.
    """
    weights = Path(config.model_weights)
    if not weights.exists():
        log.warning(
            "weights_not_found_locally",
            path=str(weights),
            hint="Ultralytics will auto-download the model on first inference",
        )
    else:
        log.debug("weights_found", path=str(weights))


def main() -> None:
    """
    Application entry point.

    Loads config, validates assets, and starts the multi-lane pipeline.
    Blocks until KeyboardInterrupt (Ctrl-C) or an unhandled exception.
    """
    args = parse_args()

    # ── Bootstrap ──────────────────────────────────────────────────────────────
    load_dotenv(args.config)
    setup_logging()
    log = get_logger(__name__)

    # ── Config ─────────────────────────────────────────────────────────────────
    config = load_config()
    log.info(
        "atms_starting",
        lane_count=config.lane_count,
        window_seconds=config.window_seconds,
        deadlock_threshold=config.deadlock_threshold,
        confidence_threshold=config.confidence_threshold,
        model_weights=config.model_weights,
        video_dir=config.video_dir,
        vehicle_classes=list(config.vehicle_classes),
    )

    # ── Asset validation ───────────────────────────────────────────────────────
    _validate_weights(config, log)
    _validate_videos(config, log)

    if args.dry_run:
        log.info("dry_run_complete", status="all_assets_ok")
        return

    # ── Pipeline ───────────────────────────────────────────────────────────────
    manager = PipelineManager(config)
    log.info("pipeline_manager_created", lane_count=config.lane_count)

    try:
        manager.start()          # blocks; KeyboardInterrupt handled inside
    except Exception:
        log.exception("pipeline_fatal_error")
        sys.exit(1)

    log.info("atms_stopped")


if __name__ == "__main__":
    main()
