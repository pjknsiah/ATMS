"""
Tests for pipeline/lane.py and pipeline/manager.py.

Lane inference is mocked — no GPU or real video files required.

Run with: pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import multiprocessing
import queue
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.pipeline.lane import LanePipeline
from src.pipeline.manager import PipelineManager
from src.signal.controller import LaneState
from src.utils.config import Config


# ── Shared helpers ────────────────────────────────────────────────────────────

def _blank_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _make_config(**overrides) -> Config:
    defaults = dict(
        window_seconds=5,
        deadlock_threshold=2,
        confidence_threshold=0.45,
        model_weights="data/weights/yolo11n.pt",
        lane_count=4,
        video_dir="data/samples",
        log_level="INFO",
        vehicle_classes=("car", "motorcycle", "bus", "truck"),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_lane(detector=None, tracker=None, window_seconds=5) -> LanePipeline:
    """Build a LanePipeline with mocked detector and tracker."""
    return LanePipeline(
        lane_id=0,
        video_path="fake/lane_0.mp4",
        detector=detector or MagicMock(),
        tracker=tracker or MagicMock(),
        window_seconds=window_seconds,
    )


# ── LanePipeline tests ────────────────────────────────────────────────────────

class TestLanePipeline:

    @pytest.fixture
    def two_frames(self):
        """Patch iter_subclip_frames to yield two blank frames."""
        frames = [_blank_frame(), _blank_frame()]
        with patch("src.pipeline.lane.iter_subclip_frames", return_value=iter(frames)):
            with patch("src.pipeline.lane.get_video_duration", return_value=10.0):
                yield frames

    def test_process_window_calls_detect_per_frame(self, two_frames) -> None:
        detector = MagicMock()
        detector.detect.return_value = []
        tracker = MagicMock()
        tracker.unique_count.return_value = 0

        lane = _make_lane(detector=detector, tracker=tracker)
        lane.process_window(0.0)

        assert detector.detect.call_count == 2

    def test_process_window_passes_detections_to_tracker(self, two_frames) -> None:
        fake_detections = [MagicMock()]
        detector = MagicMock()
        detector.detect.return_value = fake_detections
        tracker = MagicMock()
        tracker.unique_count.return_value = 1

        lane = _make_lane(detector=detector, tracker=tracker)
        lane.process_window(0.0)

        assert tracker.update.call_count == 2
        # Each call receives the fake detections and the frame
        first_call_dets = tracker.update.call_args_list[0][0][0]
        assert first_call_dets is fake_detections

    def test_process_window_returns_unique_count(self, two_frames) -> None:
        tracker = MagicMock()
        tracker.unique_count.return_value = 7

        lane = _make_lane(tracker=tracker)
        lane.detector.detect.return_value = []

        count = lane.process_window(0.0)

        assert count == 7

    def test_process_window_resets_tracker_after_counting(self, two_frames) -> None:
        tracker = MagicMock()
        tracker.unique_count.return_value = 3

        lane = _make_lane(tracker=tracker)
        lane.detector.detect.return_value = []
        lane.process_window(0.0)

        tracker.reset_window.assert_called_once()

    def test_process_window_loops_video_at_boundary(self) -> None:
        """start_sec beyond duration should wrap via modulo."""
        with patch("src.pipeline.lane.get_video_duration", return_value=10.0) as mock_dur:
            with patch("src.pipeline.lane.iter_subclip_frames", return_value=iter([])) as mock_iter:
                tracker = MagicMock()
                tracker.unique_count.return_value = 0
                lane = _make_lane(tracker=tracker)
                lane.detector.detect.return_value = []

                lane.process_window(23.0)  # 23 % 10 = 3.0

                _, call_kwargs = mock_iter.call_args
                # positional: (path, start, end, every_n)
                args = mock_iter.call_args[0]
                assert args[1] == pytest.approx(3.0)

    def test_process_window_returns_zero_on_video_error(self) -> None:
        with patch("src.pipeline.lane.get_video_duration", return_value=10.0):
            with patch("src.pipeline.lane.iter_subclip_frames", side_effect=RuntimeError("bad file")):
                tracker = MagicMock()
                tracker.unique_count.return_value = 0
                lane = _make_lane(tracker=tracker)
                lane.detector.detect.return_value = []

                count = lane.process_window(0.0)

                assert count == 0
                tracker.reset_window.assert_called_once()

    def test_run_puts_results_on_queue(self) -> None:
        """run() should emit (lane_id, count) to the queue each window."""
        import queue as stdlib_queue

        stop = multiprocessing.Event()
        q: stdlib_queue.Queue = stdlib_queue.Queue()

        lane = LanePipeline(
            lane_id=2,
            video_path="fake/lane_2.mp4",
            detector=MagicMock(),
            tracker=MagicMock(),
            window_seconds=5,
        )

        # Override process_window: return 5 and stop the loop immediately.
        def _one_shot(start_sec: float) -> int:
            stop.set()
            return 5

        lane.process_window = _one_shot  # type: ignore[method-assign]
        lane.run(q, stop)

        lane_id, count = q.get_nowait()
        assert lane_id == 2
        assert count == 5


# ── PipelineManager tests ─────────────────────────────────────────────────────

class TestPipelineManager:

    def _manager_with_queue(self, config: Config, items: list[tuple[int, int]]):
        """
        Build a PipelineManager whose queue is pre-loaded with (lane_id, count) items.
        Also replaces _stop_event so _collect_and_decide doesn't hang.
        """
        mgr = PipelineManager(config)
        for item in items:
            mgr._queue.put(item)
        return mgr

    def test_counts_accumulate_across_lanes(self) -> None:
        config = _make_config(lane_count=4, window_seconds=1)
        mgr = self._manager_with_queue(config, [(0, 3), (1, 7), (2, 2), (3, 5)])

        mgr._collect_and_decide()

        counts = {l.lane_id: l.cumulative_count for l in mgr._lanes}
        # Lane 1 wins (7) and gets reset; others keep their counts.
        assert counts[0] == 3
        assert counts[1] == 0   # winner — reset by controller
        assert counts[2] == 2
        assert counts[3] == 5

    def test_collect_and_decide_returns_winner_lane_id(self) -> None:
        config = _make_config(lane_count=2, window_seconds=1)
        mgr = self._manager_with_queue(config, [(0, 1), (1, 10)])

        winner = mgr._collect_and_decide()

        assert winner == 1

    def test_missing_lane_report_treated_as_zero(self) -> None:
        """If a lane doesn't report, its cumulative count should increase by 0."""
        config = _make_config(lane_count=2, window_seconds=1)
        mgr = self._manager_with_queue(config, [(0, 5)])   # lane 1 never reports
        # Use a tiny timeout so the test doesn't hang.
        mgr._config = _make_config(lane_count=2, window_seconds=0)

        mgr._collect_and_decide()

        assert mgr._lanes[1].cumulative_count == 0
        assert mgr._lanes[0].cumulative_count == 0  # won and was reset

    def test_counts_accumulate_over_multiple_windows(self) -> None:
        """Counts from successive windows should add up until a green is issued."""
        config = _make_config(lane_count=2, window_seconds=1, deadlock_threshold=99)
        mgr = self._manager_with_queue(config, [(0, 5), (1, 1)])
        mgr._collect_and_decide()   # lane 0 wins (5 > 1), count reset to 0

        # Second window: lane 0 gets 3 more, lane 1 gets 4 (now cumulative 1+4=5)
        for item in [(0, 3), (1, 4)]:
            mgr._queue.put(item)
        mgr._collect_and_decide()   # lane 1 cumulative=5, lane 0 cumulative=3 → lane 1 wins

        assert mgr._lanes[1].cumulative_count == 0   # lane 1 just won, reset

    def test_winner_signal_set_to_green(self) -> None:
        config = _make_config(lane_count=3, window_seconds=1)
        mgr = self._manager_with_queue(config, [(0, 1), (1, 1), (2, 10)])

        winner = mgr._collect_and_decide()

        assert winner == 2
        winning_lane = next(l for l in mgr._lanes if l.lane_id == winner)
        assert winning_lane.current_signal == "green"

    def test_non_winners_signal_set_to_red(self) -> None:
        config = _make_config(lane_count=3, window_seconds=1)
        mgr = self._manager_with_queue(config, [(0, 1), (1, 1), (2, 10)])

        winner = mgr._collect_and_decide()

        for lane in mgr._lanes:
            if lane.lane_id != winner:
                assert lane.current_signal == "red"

    def test_start_spawns_one_process_per_lane(self) -> None:
        config = _make_config(lane_count=3, window_seconds=1)
        mgr = PipelineManager(config)

        with patch("src.pipeline.manager._run_lane"):
            with patch("src.pipeline.manager.multiprocessing.Process") as MockProc:
                mock_proc_instance = MagicMock()
                MockProc.return_value = mock_proc_instance
                # Immediately stop the run loop
                mgr._stop_event.set()
                mgr.start()

        assert MockProc.call_count == 3
