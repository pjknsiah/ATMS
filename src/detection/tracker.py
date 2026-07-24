"""
Vehicle tracker — assigns consistent IDs to detections across frames
using ByteTrack (built into Ultralytics).

Responsibilities:
- Accept a list of Detection objects for a frame
- Assign / maintain track_id across frames
- Return the same list with track_id populated
- Expose a method to get unique IDs seen in the current window (for counting)
"""

from src.detection.detector import Detection


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

    def __init__(self) -> None:
        self._seen_ids: set[int] = set()
        # TODO Phase 2: initialise ByteTrack via Ultralytics tracker config

    def update(self, detections: list[Detection], frame) -> list[Detection]:
        """
        Update tracker state with new detections and return detections
        with track_id populated.

        Args:
            detections: Output from VehicleDetector.detect().
            frame: The current BGR frame (needed by ByteTrack internally).

        Returns:
            Same detections with track_id set. Detections without a confirmed
            track are returned with track_id=None.
        """
        # TODO Phase 2: call ByteTrack update, map IDs back to Detection objects
        # add newly seen IDs to self._seen_ids
        raise NotImplementedError("Implement in Phase 2 — see CLAUDE.md")

    def unique_count(self) -> int:
        """Return the number of unique vehicle IDs seen in the current window."""
        return len(self._seen_ids)

    def reset_window(self) -> None:
        """Clear seen IDs at the start of a new time window."""
        self._seen_ids.clear()
