"""
Vehicle detector — wraps Ultralytics YOLO11.

Responsibilities:
- Load model weights once at startup
- Run inference on a single frame
- Return only vehicle-class detections above the confidence threshold
- Do NOT track — tracking is in tracker.py
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


VEHICLE_CLASSES_DEFAULT = {"car", "motorcycle", "bus", "truck"}


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
        vehicle_classes: Set of class names to keep. Defaults to car/motorcycle/bus/truck.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.45,
        vehicle_classes: set[str] = VEHICLE_CLASSES_DEFAULT,
    ) -> None:
        # TODO Phase 2: from ultralytics import YOLO; self.model = YOLO(weights_path)
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.vehicle_classes = vehicle_classes
        self.model = None  # replace in Phase 2

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: OpenCV BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects filtered to vehicle classes.
        """
        # TODO Phase 2: run self.model(frame, conf=self.confidence, verbose=False)
        # and parse results into Detection objects
        raise NotImplementedError("Implement in Phase 2 — see CLAUDE.md")
