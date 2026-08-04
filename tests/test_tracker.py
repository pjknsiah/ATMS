"""
Tests for detection/tracker.py.

BYTETracker is mocked — no GPU or real video required.

Run with: pytest tests/test_tracker.py -v
"""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.detector import Detection
from src.detection.tracker import VehicleTracker, _iou


def _det(bbox: tuple[int, int, int, int], conf: float = 0.9) -> Detection:
    return Detection(class_name="car", confidence=conf, bbox=bbox)


def _fake_track(bbox: tuple[int, int, int, int], track_id: int) -> MagicMock:
    """Build a mock STrack object as returned by BYTETracker.update()."""
    t = MagicMock()
    t.tlbr = list(bbox)
    t.track_id = track_id
    return t


@pytest.fixture
def tracker(mocker) -> VehicleTracker:
    """VehicleTracker with BYTETracker mocked out."""
    mocker.patch("src.detection.tracker.BYTETracker")
    return VehicleTracker(frame_rate=30)


class TestIoU:
    def test_identical_boxes(self) -> None:
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_non_overlapping(self) -> None:
        assert _iou((0, 0, 5, 5), (10, 10, 20, 20)) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # boxes share a 5x5 corner; union = 75
        score = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < score < 1.0


class TestVehicleTracker:
    def test_empty_detections_returns_empty_list(self, tracker: VehicleTracker) -> None:
        result = tracker.update([], np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_track_ids_assigned_by_iou(self, tracker: VehicleTracker) -> None:
        """Detection overlapping a returned track must get its track_id."""
        bbox = (10, 10, 100, 100)
        det = _det(bbox)
        track = _fake_track(bbox, track_id=7)
        tracker._tracker.update.return_value = [track]

        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))

        assert result[0].track_id == 7

    def test_non_overlapping_track_gets_no_id(self, tracker: VehicleTracker) -> None:
        """Detection far from all tracks keeps track_id=None."""
        det = _det((0, 0, 10, 10))
        track = _fake_track((200, 200, 300, 300), track_id=1)
        tracker._tracker.update.return_value = [track]

        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))

        assert result[0].track_id is None

    def test_unique_count_increments_per_new_id(self, tracker: VehicleTracker) -> None:
        bbox = (10, 10, 100, 100)
        tracker._tracker.update.return_value = [_fake_track(bbox, track_id=1)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        tracker.update([_det(bbox)], frame)
        assert tracker.unique_count() == 1

        # Same ID again — count stays at 1
        tracker.update([_det(bbox)], frame)
        assert tracker.unique_count() == 1

    def test_unique_count_counts_distinct_ids(self, tracker: VehicleTracker) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (10, 10, 100, 100)

        tracker._tracker.update.return_value = [_fake_track(bbox, track_id=1)]
        tracker.update([_det(bbox)], frame)

        tracker._tracker.update.return_value = [_fake_track(bbox, track_id=2)]
        tracker.update([_det(bbox)], frame)

        assert tracker.unique_count() == 2

    def test_reset_window_clears_seen_ids(self, tracker: VehicleTracker) -> None:
        bbox = (10, 10, 100, 100)
        tracker._tracker.update.return_value = [_fake_track(bbox, track_id=5)]
        tracker.update([_det(bbox)], np.zeros((480, 640, 3), dtype=np.uint8))
        assert tracker.unique_count() == 1

        tracker.reset_window()
        assert tracker.unique_count() == 0

    def test_tracker_exception_returns_detections_unchanged(
        self, tracker: VehicleTracker
    ) -> None:
        """If ByteTrack raises, detections are returned with track_id=None."""
        tracker._tracker.update.side_effect = RuntimeError("tracker crash")
        det = _det((10, 10, 100, 100))
        result = tracker.update([det], np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result) == 1
        assert result[0].track_id is None
