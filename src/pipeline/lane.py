"""
Per-lane pipeline — reads one video subclip, runs detection + tracking,
and reports the unique vehicle count for that window.

Responsibilities:
- Extract frames from the current WINDOW_SECONDS subclip
- Run detection and tracking on each frame
- Accumulate unique vehicle IDs via the tracker
- Post the window count to a shared result queue
- Loop continuously until stop_event is set
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import numpy as np

from src.detection.detector import Detection, VehicleDetector
from src.detection.tracker import VehicleTracker
from src.utils.logger import get_logger
from src.utils.video import get_video_duration, iter_subclip_frames

log = get_logger(__name__)


class LanePipeline:
    """
    Processes one lane: reads subclip frames, detects and tracks vehicles,
    and emits a per-window unique count.

    Args:
        lane_id: Zero-based lane index.
        video_path: Path to the lane's video file.
        detector: Initialised VehicleDetector.
        tracker: Initialised VehicleTracker.
        window_seconds: Duration of each processing window in seconds.
    """

    def __init__(
        self,
        lane_id: int,
        video_path: str | Path,
        detector: VehicleDetector,
        tracker: VehicleTracker,
        window_seconds: int,
    ) -> None:
        self.lane_id = lane_id
        self.video_path = Path(video_path)
        self.detector = detector
        self.tracker = tracker
        self.window_seconds = window_seconds

    def process_window(self, start_sec: float) -> int:
        """
        Process one WINDOW_SECONDS chunk of video starting at start_sec.

        Frames are looped if start_sec exceeds the video duration.
        Resets the tracker window after counting so the next call starts fresh.

        Args:
            start_sec: Absolute elapsed time in seconds (may exceed video length).

        Returns:
            Number of unique vehicle IDs observed in this window.
        """
        duration = get_video_duration(self.video_path)
        actual_start = start_sec % duration
        actual_end = min(actual_start + self.window_seconds, duration)

        try:
            for frame in iter_subclip_frames(self.video_path, actual_start, actual_end):
                detections: list[Detection] = self.detector.detect(frame)
                self.tracker.update(detections, frame)
        except Exception:
            log.exception("process_window_error", lane_id=self.lane_id, start=actual_start)

        count = self.tracker.unique_count()
        self.tracker.reset_window()
        return count

    def run(
        self,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ) -> None:
        """
        Continuously process windows and put (lane_id, count) onto result_queue
        until stop_event is set.

        Args:
            result_queue: Queue shared with PipelineManager for result delivery.
            stop_event: Set by PipelineManager to request a clean shutdown.
        """
        log.info("lane_started", lane_id=self.lane_id, video=str(self.video_path))
        window_start = 0.0

        while not stop_event.is_set():
            count = self.process_window(window_start)
            result_queue.put((self.lane_id, count))
            log.debug("window_done", lane_id=self.lane_id, count=count, start=window_start)
            window_start += self.window_seconds
