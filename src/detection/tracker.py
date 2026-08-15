"""
Vehicle tracker — assigns consistent IDs to detections across frames
using ByteTrack (built into Ultralytics).

Responsibilities:
- Accept a list of Detection objects for a frame
- Assign / maintain track_id across frames via ByteTrack
- Return the same list with track_id populated
- Expose a method to get unique IDs seen in the current window (for counting)

Ultralytics 8.4+ API notes:
- BYTETracker(args) — no frame_rate kwarg; frame_rate can be added to args if needed.
- update() expects a Results-like object with .conf / .xywh / .cls as float32
  numpy arrays and boolean-index support.
- update() returns np.ndarray shape (M, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
  where idx is the index into the original detection list passed this frame.
"""

from __future__ import annotations
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers import BYTETracker

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)

# ByteTrack hyperparameters — match ultralytics bytetrack.yaml defaults.
_BYTETRACK_ARGS = SimpleNamespace(
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """
    Return Intersection-over-Union between two (x1, y1, x2, y2) boxes.

    Args:
        a: First bounding box as (x1, y1, x2, y2).
        b: Second bounding box as (x1, y1, x2, y2).

    Returns:
        IoU score in [0.0, 1.0].
    """
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

    Ultralytics 8.4+ BYTETracker reads:
        .conf  — float32 ndarray (N,)   confidence scores
        .xywh  — float32 ndarray (N,4)  centre-x, centre-y, width, height
        .cls   — float32 ndarray (N,)   class indices
    It also calls results[bool_mask] to split high/low confidence subsets,
    and len(results) to check for empty batches.
    """

    def __init__(
        self,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xywh = xywh  # (N, 4) float32
        self.conf = conf  # (N,)   float32
        self.cls = cls  # (N,)   float32

    # ── Container protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_FakeResults":
        """Boolean-index all arrays in lock-step."""
        return _FakeResults(self.xywh[mask], self.conf[mask], self.cls[mask])

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> "_FakeResults":
        """
        Build a _FakeResults from a list of Detection objects.

        Converts (x1, y1, x2, y2) bboxes to (cx, cy, w, h) required by ByteTrack.

        Args:
            detections: Output from VehicleDetector.detect().

        Returns:
            A _FakeResults instance ready to pass to BYTETracker.update().
        """
        if not detections:
            return cls(
                np.zeros((0, 4), dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        xywh_rows: list[list[float]] = []
        confs: list[float] = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = float(x2 - x1)
            h = float(y2 - y1)
            xywh_rows.append([cx, cy, w, h])
            confs.append(d.confidence)

        return cls(
            np.array(xywh_rows, dtype=np.float32),
            np.array(confs, dtype=np.float32),
            np.zeros(len(detections), dtype=np.float32),
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

    Args:
        frame_rate: Video frame rate hint (unused by BYTETracker 8.4+,
                    kept for API compatibility).
    """

    def __init__(self, frame_rate: int = 30) -> None:  # noqa: ARG002
        self._tracker = BYTETracker(_BYTETRACK_ARGS)
        self._seen_ids: set[int] = set()

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        """
        Update tracker state with new detections and return detections
        with track_id populated.

        ByteTrack 8.4+ returns a numpy array of shape (M, 8):
            [x1, y1, x2, y2, track_id, score, cls, idx]
        where ``idx`` is the index into ``detections`` for this frame.
        We use idx to set ``detection.track_id`` directly without IoU matching.

        Args:
            detections: Output from VehicleDetector.detect().
            frame: Current BGR frame passed to ByteTrack internally.

        Returns:
            Same detections list with track_id set for confirmed tracks.
            Unmatched detections keep track_id=None.
        """
        if not detections:
            return []

        try:
            fake = _FakeResults.from_detections(detections)
            # tracks: ndarray (M, 8) — [x1,y1,x2,y2, track_id, score, cls, idx]
            tracks: np.ndarray = self._tracker.update(fake, frame)
        except Exception:
            log.exception("tracker_update_failed")
            return detections

        if tracks is None or len(tracks) == 0:
            return detections

        for row in tracks:
            track_id = int(row[4])
            det_idx = int(row[7])
            self._seen_ids.add(track_id)
            if 0 <= det_idx < len(detections):
                detections[det_idx].track_id = track_id

        return detections

    def unique_count(self) -> int:
        """Return the number of unique vehicle IDs seen in the current window."""
        return len(self._seen_ids)

    def reset_window(self) -> None:
        """Clear seen IDs at the start of a new time window."""
        self._seen_ids.clear()
