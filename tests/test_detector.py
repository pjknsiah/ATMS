"""
Tests for detection/detector.py.

Uses mocked YOLO output — does not require GPU or real video.

Run with: pytest tests/test_detector.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.detection.detector import Detection, VehicleDetector


def blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestVehicleDetector:

    def test_returns_only_vehicle_classes(self, mocker):
        """Detections for non-vehicle classes must be filtered out."""
        # TODO Phase 2: mock ultralytics YOLO output with mixed classes
        # assert only car/motorcycle/bus/truck in results
        pytest.skip("Implement after Phase 2 — mock YOLO output here")

    def test_filters_by_confidence(self, mocker):
        """Detections below confidence threshold must be excluded."""
        pytest.skip("Implement after Phase 2")

    def test_returns_list_of_detection_objects(self, mocker):
        """Return type must be List[Detection]."""
        pytest.skip("Implement after Phase 2")

    def test_detection_dataclass_fields(self):
        """Detection dataclass must have required fields."""
        d = Detection(
            class_name="car",
            confidence=0.87,
            bbox=(10, 20, 100, 200),
            track_id=None,
        )
        assert d.class_name == "car"
        assert d.confidence == 0.87
        assert d.bbox == (10, 20, 100, 200)
        assert d.track_id is None

    def test_detection_track_id_can_be_set(self):
        """track_id field must be settable after construction."""
        d = Detection(class_name="truck", confidence=0.9, bbox=(0, 0, 50, 50))
        d.track_id = 42
        assert d.track_id == 42
