"""
Tests for detection/tracker.py.

BYTETracker is mocked — no GPU or real video required.

Ultralytics 8.4+ BYTETracker.update() returns a numpy array of shape (M, 8):
    [x1, y1, x2, y2, track_id, score, cls, idx]
where idx is the index into the original detection list passed this frame.

Run with: pytest tests/test_tracker.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.detector import Detection
from src.detection.tracker import VehicleTracker, _iou


def _det(bbox: tuple[int, int, int, int], conf: float = 0.9) -> Detection:
    return Detection(class_name="car", confidence=conf, bbox=bbox)


def _track_row(
    bbox: tuple[int, int, int, int],
    track_id: int,
    det_idx: int = 0,
    score: float = 0.9,
) -> np.ndarray:
    """
    Build one row of the numpy array returned by BYTETracker.update().

    Columns: [x1, y1, x2, y2, track_id, score, cls, idx]
    """
    x1, y1, x2, y2 = bbox
    return np.array([x1, y1, x2, y2, track_id, score, 0.0, det_idx], dtype=np.float32)


@pytest.fixture
def tracker(mocker) -> VehicleTracker:
    """VehicleTracker with BYTETracker mocked out."""
    mocker.patch("src.detection.tracker.BYTETracker")
    return VehicleTracker(frame_rate=30)


# ── IoU helper ────────────────────────────────────────────────────────────────

class TestIoU:
    def test_identical_boxes(self) -> None:
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_non_overlapping(self) -> None:
        assert _iou((0, 0, 5, 5), (10, 10, 20, 20)) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # boxes share a 5x5 corner; union = 75
        score = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < score < 1.0


# ── VehicleTracker ────────────────────────────────────────────────────────────

class TestVehicleTracker:
    def test_empty_detections_returns_empty_list(self, tracker: VehicleTracker) -> None:
        result = tracker.update([], np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_track_ids_assigned_via_idx(self, tracker: VehicleTracker) -> None:
        """
        ByteTrack returns a row whose idx column (7) points back to the
        original detection; that detection must get the matching track_id.
        """
        bbox = (10, 10, 100, 100)
        det = _det(bbox)
        # Return one track row: track_id=7, det_idx=0 → detections[0]
        tracker._tracker.update.return_value = np.array([_track_row(bbox, track_id=7, det_idx=0)])

        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))

        assert result[0].track_id == 7

    def test_no_tracks_returned_means_no_id(self, tracker: VehicleTracker) -> None:
        """
        If ByteTrack returns an empty array (no confirmed tracks),
        the detection keeps track_id=None.
        """
        det = _det((0, 0, 10, 10))
        tracker._tracker.update.return_value = np.zeros((0, 8), dtype=np.float32)

        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))

        assert result[0].track_id is None

    def test_unique_count_increments_per_new_id(self, tracker: VehicleTracker) -> None:
        bbox = (10, 10, 100, 100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker._tracker.update.return_value = np.array([_track_row(bbox, track_id=1, det_idx=0)])

        tracker.update([_det(bbox)], frame)
        assert tracker.unique_count() == 1

        # Same ID again — count stays at 1
        tracker.update([_det(bbox)], frame)
        assert tracker.unique_count() == 1

    def test_unique_count_counts_distinct_ids(self, tracker: VehicleTracker) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (10, 10, 100, 100)

        tracker._tracker.update.return_value = np.array([_track_row(bbox, track_id=1, det_idx=0)])
        tracker.update([_det(bbox)], frame)

        tracker._tracker.update.return_value = np.array([_track_row(bbox, track_id=2, det_idx=0)])
        tracker.update([_det(bbox)], frame)

        assert tracker.unique_count() == 2

    def test_reset_window_clears_seen_ids(self, tracker: VehicleTracker) -> None:
        bbox = (10, 10, 100, 100)
        tracker._tracker.update.return_value = np.array([_track_row(bbox, track_id=5, det_idx=0)])
        tracker.update([_det(bbox)], np.zeros((480, 640, 3), dtype=np.uint8))
        assert tracker.unique_count() == 1

        tracker.reset_window()
        assert tracker.unique_count() == 0

    def test_out_of_range_idx_does_not_crash(self, tracker: VehicleTracker) -> None:
        """Track row with out-of-range idx is counted but does not set any detection."""
        det = _det((10, 10, 50, 50))
        # idx=99 is way out of bounds — should be silently ignored
        tracker._tracker.update.return_value = np.array(
            [_track_row((10, 10, 50, 50), track_id=3, det_idx=99)]
        )

        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))

        assert tracker.unique_count() == 1   # track_id was recorded
        assert result[0].track_id is None    # but the detection didn't get updated

    def test_tracker_exception_returns_detections_unchanged(
        self, tracker: VehicleTracker
    ) -> None:
        """If ByteTrack raises, detections are returned with track_id=None."""
        tracker._tracker.update.side_effect = RuntimeError("tracker crash")
        det = _det((10, 10, 100, 100))
        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result) == 1
        assert result[0].track_id is None
