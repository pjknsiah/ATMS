"""
Video utilities — load video files and extract frames for processing.

Responsibilities:
- Open a video file with OpenCV
- Extract frames from a time-window subclip
- Yield frames as numpy arrays for the detector
"""

from __future__ import annotations
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np


def iter_subclip_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    every_n_frames: int = 3,
) -> Generator[np.ndarray, None, None]:
    """
    Yield frames from a video between start_sec and end_sec.

    Args:
        video_path: Path to the video file.
        start_sec: Start of the subclip in seconds.
        end_sec: End of the subclip in seconds.
        every_n_frames: Sample every N frames (reduces CPU load).

    Yields:
        BGR frames as numpy arrays.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If the video cannot be opened.
    """
    # TODO Phase 1: implement with cv2.VideoCapture
    raise NotImplementedError("Implement in Phase 1 — see CLAUDE.md")


def get_video_duration(video_path: str | Path) -> float:
    """
    Return the total duration of a video file in seconds.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds as float.
    """
    # TODO Phase 1: use cv2.VideoCapture + CAP_PROP_FRAME_COUNT / CAP_PROP_FPS
    raise NotImplementedError("Implement in Phase 1 — see CLAUDE.md")
