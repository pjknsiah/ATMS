"""
Vehicle tracker — assigns consistent IDs to detections across frames
using ByteTrack (built into Ultralytics).

Responsibilities:
- Accept a list of Detection objects for a frame
- Assign / maintain track_id across frames via ByteTrack
- Return the same list with track_id populated
- Expose a method to get unique IDs seen in the current window (for counting)
"""

from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.trackers import BYTETracker

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)

# ByteTrack hyperparameters — values match the ultralytics bytetrack.yaml defaults.
_BYTETRACK_ARGS = SimpleNamespace(
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Return IoU between two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _FakeResults:
    """
    Minimal duck-typed shim so BYTETracker.update() can receive
    pre-computed detections without re-running YOLO inference.

    BYTETracker reads `.boxes.xyxy`, `.boxes.conf`, and `.boxes.cls`
    as float32 tensors.
    """

    def __init__(self, detections: list[Detection]) -> None:
        if detections:
            bboxes = [list(d.bbox) for d in detections]
            confs = [d.confidence for d in detections]
        else:
            bboxes, confs = [], []

        self.boxes = SimpleNamespace(
            xyxy=torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            conf=torch.tensor(confs, dtype=torch.float32),
            cls=torch.zeros(len(detections), dtype=torch.float32),
        )


class VehicleTracker:
    """
    Wraps ByteTrack via Ultralytics for cross-frame vehicle tracking.

    Usage:
        tracker = VehicleTracker()
        for frame in subclip_frames:
            detections = detector.detect(frame)
            tracked = tracker.update(detections, frame)
        window_count = tracker.unique_count()
        tracker.reset_window()
    """

    def __init__(self, frame_rate: int = 30) -> None:
        self._tracker = BYTETracker(_BYTETRACK_ARGS, frame_rate=frame_rate)
        self._seen_ids: set[int] = set()

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        """
        Update tracker state with new detections and return detections
        with track_id populated.

        Args:
            detections: Output from VehicleDetector.detect().
            frame: The current BGR frame (passed to ByteTrack internally).

        Returns:
            Same detections with track_id set where a track was confirmed.
            Detections with no matching track keep track_id=None.
        """
        if not detections:
            return []

        try:
            fake = _FakeResults(detections)
            tracks = self._tracker.update(fake, frame)
        except Exception:
            log.exception("tracker_update_failed")
            return detections

        # Map each track's bbox back to a Detection via IoU.
        # STrack exposes .tlbr → [x1, y1, x2, y2] and .track_id.
        track_entries = [
            (int(t.tlbr[0]), int(t.tlbr[1]), int(t.tlbr[2]), int(t.tlbr[3]), t.track_id)
            for t in tracks
        ]

        _MIN_IOU = 0.4
        for det in detections:
            best_id: int | None = None
            best_iou = _MIN_IOU
            for tx1, ty1, tx2, ty2, tid in track_entries:
                score = _iou(det.bbox, (tx1, ty1, tx2, ty2))
                if score > best_iou:
                    best_iou = score
                    best_id = tid
            det.track_id = best_id
            if best_id is not None:
                self._seen_ids.add(best_id)

        return detections

    def unique_count(self) -> int:
        """Return the number of unique vehicle IDs seen in the current window."""
        return len(self._seen_ids)

    def reset_window(self) -> None:
        """Clear seen IDs at the start of a new time window."""
        self._seen_ids.clear()
