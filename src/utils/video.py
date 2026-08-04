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

from src.utils.logger import get_logger

log = get_logger(__name__)


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
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (start_frame + frame_idx) >= end_frame:
                break
            if frame_idx % every_n_frames == 0:
                yield frame
            frame_idx += 1
    except Exception:
        log.exception("frame_read_error", video=str(path))
        raise
    finally:
        cap.release()


def get_video_duration(video_path: str | Path) -> float:
    """
    Return the total duration of a video file in seconds.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds as float.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If the video cannot be opened or has invalid FPS.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(f"Invalid FPS ({fps}) for video: {path}")
        return frame_count / fps
    finally:
        cap.release()
