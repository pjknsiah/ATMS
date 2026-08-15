"""
Configuration loader — reads environment variables populated by python-dotenv.

All config access goes through load_config(); no bare os.getenv() calls elsewhere.
"""

from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Typed, immutable snapshot of all runtime settings."""

    window_seconds: int
    deadlock_threshold: int
    confidence_threshold: float
    model_weights: str
    lane_count: int
    video_dir: str
    log_level: str
    vehicle_classes: tuple[str, ...]
    every_n_frames: int = 5  # sample 1 in every N frames per window


def load_config() -> Config:
    """
    Build Config from environment variables. Call after dotenv is loaded.

    Returns:
        Immutable Config dataclass with all settings.
    """
    raw_classes = os.getenv("VEHICLE_CLASSES", "car,motorcycle,bus,truck")
    return Config(
        window_seconds=int(os.getenv("WINDOW_SECONDS", "5")),
        deadlock_threshold=int(os.getenv("DEADLOCK_THRESHOLD", "2")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.45")),
        model_weights=os.getenv("MODEL_WEIGHTS", "data/weights/yolo11n.pt"),
        lane_count=int(os.getenv("LANE_COUNT", "4")),
        video_dir=os.getenv("VIDEO_DIR", "data/samples"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        vehicle_classes=tuple(c.strip() for c in raw_classes.split(",")),
        every_n_frames=int(os.getenv("EVERY_N_FRAMES", "5")),
    )
