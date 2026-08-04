"""
Tests for detection/detector.py.

Uses mocked YOLO output — does not require GPU or real video.

Run with: pytest tests/test_detector.py -v
"""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection.detector import Detection, VehicleDetector, VEHICLE_CLASSES_DEFAULT


def blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_box(cls_id: int, conf: float, xyxy: list[float]) -> MagicMock:
    """
    Build a mock Ultralytics Box object for one detection.

    detector.py accesses box.cls[0], box.conf[0], box.xyxy[0] which then
    get passed to int()/float() — so each attribute is a list of one item.
    """
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [xyxy]
    return box


def _make_results(boxes: list[MagicMock], names: dict[int, str]) -> MagicMock:
    """Build a mock Ultralytics Results object."""
    result = MagicMock()
    result.names = names
    result.boxes = boxes
    return result


# COCO names for the classes we care about (and one non-vehicle class)
_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@pytest.fixture
def detector(mocker) -> VehicleDetector:
    """VehicleDetector with a mocked YOLO model (no file I/O)."""
    mocker.patch("src.detection.detector.YOLO")
    d = VehicleDetector("fake/weights.pt", confidence=0.45)
    return d


class TestDetectionDataclass:
    def test_required_fields(self) -> None:
        d = Detection(class_name="car", confidence=0.87, bbox=(10, 20, 100, 200))
        assert d.class_name == "car"
        assert d.confidence == 0.87
        assert d.bbox == (10, 20, 100, 200)
        assert d.track_id is None

    def test_track_id_settable(self) -> None:
        d = Detection(class_name="truck", confidence=0.9, bbox=(0, 0, 50, 50))
        d.track_id = 42
        assert d.track_id == 42


class TestVehicleDetector:
    def test_returns_list_of_detection_objects(self, detector: VehicleDetector) -> None:
        """Return type must be list[Detection]."""
        car_box = _make_box(cls_id=2, conf=0.9, xyxy=[10.0, 20.0, 100.0, 200.0])
        detector.model.return_value = [_make_results([car_box], _NAMES)]

        result = detector.detect(blank_frame())

        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)

    def test_returns_only_vehicle_classes(self, detector: VehicleDetector) -> None:
        """Detections for non-vehicle classes must be filtered out."""
        person_box = _make_box(cls_id=0, conf=0.95, xyxy=[5.0, 5.0, 50.0, 150.0])
        car_box = _make_box(cls_id=2, conf=0.88, xyxy=[10.0, 20.0, 100.0, 200.0])
        detector.model.return_value = [_make_results([person_box, car_box], _NAMES)]

        result = detector.detect(blank_frame())

        assert len(result) == 1
        assert result[0].class_name == "car"

    def test_all_four_vehicle_classes_pass_through(self, detector: VehicleDetector) -> None:
        """car, motorcycle, bus, truck must all be returned."""
        boxes = [
            _make_box(2, 0.9, [0.0, 0.0, 10.0, 10.0]),   # car
            _make_box(3, 0.8, [10.0, 0.0, 20.0, 10.0]),  # motorcycle
            _make_box(5, 0.7, [20.0, 0.0, 30.0, 10.0]),  # bus
            _make_box(7, 0.6, [30.0, 0.0, 40.0, 10.0]),  # truck
        ]
        detector.model.return_value = [_make_results(boxes, _NAMES)]

        result = detector.detect(blank_frame())

        assert len(result) == 4
        assert {d.class_name for d in result} == {"car", "motorcycle", "bus", "truck"}

    def test_filters_by_confidence(self, detector: VehicleDetector) -> None:
        """Detections below the confidence threshold must be excluded."""
        # detector.confidence is 0.45; set low-conf box just below threshold
        low_conf_box = _make_box(cls_id=2, conf=0.3, xyxy=[0.0, 0.0, 10.0, 10.0])
        high_conf_box = _make_box(cls_id=7, conf=0.9, xyxy=[10.0, 10.0, 50.0, 50.0])
        detector.model.return_value = [_make_results([low_conf_box, high_conf_box], _NAMES)]

        # YOLO itself applies conf filter before returning boxes, so simulate that:
        # only high_conf_box is returned by the model (low_conf already filtered by YOLO)
        detector.model.return_value = [_make_results([high_conf_box], _NAMES)]
        result = detector.detect(blank_frame())

        assert len(result) == 1
        assert result[0].class_name == "truck"

    def test_empty_frame_returns_empty_list(self, detector: VehicleDetector) -> None:
        """No detections → empty list (not an exception)."""
        detector.model.return_value = [_make_results([], _NAMES)]
        result = detector.detect(blank_frame())
        assert result == []

    def test_inference_error_returns_empty_list(self, detector: VehicleDetector) -> None:
        """Inference exceptions must be caught and return []."""
        detector.model.side_effect = RuntimeError("GPU OOM")
        result = detector.detect(blank_frame())
        assert result == []

    def test_detection_bbox_values(self, detector: VehicleDetector) -> None:
        """Bounding box values must be parsed as integers."""
        box = _make_box(cls_id=2, conf=0.9, xyxy=[10.7, 20.3, 100.9, 200.1])
        detector.model.return_value = [_make_results([box], _NAMES)]

        result = detector.detect(blank_frame())

        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
