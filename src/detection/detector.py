"""
Vehicle detector — wraps Ultralytics YOLO11.

Responsibilities:
- Load model weights once at startup
- Run inference on a single frame
- Return only vehicle-class detections above the confidence threshold
- Do NOT track — tracking is in tracker.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.utils.logger import get_logger

log = get_logger(__name__)

VEHICLE_CLASSES_DEFAULT: frozenset[str] = frozenset({"car", "motorcycle", "bus", "truck"})


@dataclass
class Detection:
    """A single detected object in one frame."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    track_id: int | None = field(default=None)  # populated by tracker


class VehicleDetector:
    """
    Wraps Ultralytics YOLO11 for vehicle detection.

    Args:
        weights_path: Path to YOLO11 .pt model file.
        confidence: Minimum confidence threshold (0–1).
        vehicle_classes: Class names to keep. Defaults to car/motorcycle/bus/truck.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.45,
        vehicle_classes: frozenset[str] | set[str] = VEHICLE_CLASSES_DEFAULT,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.vehicle_classes = frozenset(vehicle_classes)
        try:
            self.model: YOLO = YOLO(str(self.weights_path))
            log.info("detector_loaded", weights=str(self.weights_path))
        except Exception:
            log.exception("detector_load_failed", weights=str(self.weights_path))
            raise

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: OpenCV BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects filtered to vehicle classes above
            the confidence threshold. Returns an empty list on inference error.
        """
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)[0]
        except Exception:
            log.exception("inference_failed")
            return []

        detections: list[Detection] = []
        names: dict[int, str] = results.names

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "")
            if class_name not in self.vehicle_classes:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(
                Detection(class_name=class_name, confidence=conf, bbox=(x1, y1, x2, y2))
            )

        return detections
