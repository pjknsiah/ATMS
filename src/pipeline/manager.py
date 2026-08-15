"""
Pipeline manager — spawns one process per lane and runs the signal controller loop.

Responsibilities:
- Build VehicleDetector + VehicleTracker + LanePipeline per lane inside each process
- Collect per-window counts from all lanes via a shared Queue
- Accumulate counts into LaneState objects
- Call SignalController.decide() every WINDOW_SECONDS to issue green/red signals
- Handle clean shutdown on KeyboardInterrupt or explicit stop()

Concurrency model:
  One multiprocessing.Process per lane satisfies the "multiprocessing for
  CPU-bound YOLO inference" requirement from CLAUDE.md. The main process
  orchestrates the signal controller on a WINDOW_SECONDS cadence.
"""

from __future__ import annotations

import multiprocessing
import queue
import time
from pathlib import Path

from src.detection.detector import VehicleDetector
from src.detection.tracker import VehicleTracker
from src.pipeline.lane import LanePipeline
from src.signal.controller import LaneState, SignalController
from src.utils.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _run_lane(
    lane_id: int,
    video_path: str,
    weights_path: str,
    confidence: float,
    vehicle_classes: set[str],
    window_seconds: int,
    every_n_frames: int,
    result_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
) -> None:
    """
    Process-target function for a single lane.

    Creates the detector and tracker inside the subprocess so the heavy
    YOLO model is loaded in the worker, not the main process.

    Args:
        lane_id: Zero-based lane index.
        video_path: Path string to the lane video file.
        weights_path: Path string to YOLO11 weights.
        confidence: Minimum detection confidence threshold.
        vehicle_classes: Set of class names to count.
        window_seconds: Window duration in seconds.
        every_n_frames: Sample 1 in every N video frames per window.
        result_queue: Shared queue for posting (lane_id, count) results.
        stop_event: Set by the manager to request shutdown.
    """
    try:
        # Limit intra-op parallelism so N lane processes don't collectively
        # oversubscribe the available CPU cores (each would otherwise spawn
        # as many threads as there are cores, totalling N × cores threads).
        import os
        import torch

        n_cores = os.cpu_count() or 4
        threads_per_lane = max(1, n_cores // 4)  # always leave headroom for 4 lanes
        torch.set_num_threads(threads_per_lane)
        torch.set_num_interop_threads(1)

        detector = VehicleDetector(weights_path, confidence, vehicle_classes)
        tracker = VehicleTracker()
        pipeline = LanePipeline(
            lane_id, video_path, detector, tracker, window_seconds, every_n_frames
        )
        pipeline.run(result_queue, stop_event)
    except Exception:
        log.exception("lane_process_error", lane_id=lane_id)


class PipelineManager:
    """
    Orchestrates all lane pipelines and the signal controller.

    Args:
        config: Loaded Config dataclass (from load_config()).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._controller = SignalController(config.deadlock_threshold)
        self._lanes: list[LaneState] = [
            LaneState(lane_id=i) for i in range(config.lane_count)
        ]
        self._processes: list[multiprocessing.Process] = []
        self._queue: multiprocessing.Queue = multiprocessing.Queue()
        self._stop_event: multiprocessing.Event = multiprocessing.Event()

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Spawn one subprocess per lane and begin the signal controller loop.

        Blocks until stop() is called or a KeyboardInterrupt is received.
        """
        vehicle_classes = set(self._config.vehicle_classes)

        for lane in self._lanes:
            video_path = str(Path(self._config.video_dir) / f"lane_{lane.lane_id}.mp4")
            proc = multiprocessing.Process(
                target=_run_lane,
                args=(
                    lane.lane_id,
                    video_path,
                    self._config.model_weights,
                    self._config.confidence_threshold,
                    vehicle_classes,
                    self._config.window_seconds,
                    self._config.every_n_frames,
                    self._queue,
                    self._stop_event,
                ),
                daemon=True,
                name=f"lane-{lane.lane_id}",
            )
            proc.start()
            self._processes.append(proc)
            log.info("lane_process_started", lane_id=lane.lane_id, pid=proc.pid)

        try:
            self._run_loop()
        except KeyboardInterrupt:
            log.info("keyboard_interrupt_received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Signal all lane processes to stop and wait for them to exit."""
        self._stop_event.set()
        for proc in self._processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                log.warning("lane_process_force_killed", pid=proc.pid)
        log.info("pipeline_stopped")

    # ── Internal loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Main controller loop: collect one window of counts from every lane,
        update LaneState, and call the signal controller.
        """
        while not self._stop_event.is_set():
            winner = self._collect_and_decide()

    def _collect_and_decide(self) -> int:
        """
        Block until every lane has posted a count for the current window
        (or the 2× timeout expires), then run the signal controller.

        Args:
            None

        Returns:
            lane_id of the lane granted the green signal.
        """
        received: dict[int, int] = {}
        # collect_timeout gives all lanes a chance to post their window result
        # before we fall back to 0 for missing lanes.  On CPU-only machines N
        # parallel YOLO processes compete for cores, so the effective wall time
        # can exceed window_seconds × lane_count; set COLLECT_TIMEOUT in .env
        # to tune (default 60 s).
        deadline = time.monotonic() + self._config.collect_timeout

        while len(received) < len(self._lanes) and time.monotonic() < deadline:
            try:
                lane_id, count = self._queue.get(timeout=0.5)
                if lane_id not in received:
                    received[lane_id] = count
            except queue.Empty:
                continue

        for lane in self._lanes:
            lane.cumulative_count += received.get(lane.lane_id, 0)

        # Snapshot cumulative counts NOW — before decide() resets the winner.
        pre_decision = {l.lane_id: l.cumulative_count for l in self._lanes}
        winner_id = self._controller.decide(self._lanes)

        log.info(
            "signal_granted",
            winner=winner_id,
            cumulative_counts=pre_decision,
            window_received=received,
            signals={l.lane_id: l.current_signal for l in self._lanes},
        )
        return winner_id
