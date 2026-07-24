"""
Structured logger — single setup point for the whole project.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("signal_granted", lane_id=2, count=14)
"""

import logging
import os
import structlog


def setup_logging() -> None:
    """Configure structlog. Call once at startup in main.py."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
