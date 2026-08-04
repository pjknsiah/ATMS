"""Tests for src.utils.video"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.video import get_video_duration, iter_subclip_frames


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a synthetic 3-second, 10-FPS, 320x240 video."""
    video_path = tmp_path / "test_lane.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    assert writer.isOpened(), "cv2.VideoWriter failed — check codec support"
    for i in range(30):  # 30 frames @ 10 FPS = 3 s
        frame = np.full((240, 320, 3), (i * 8) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestGetVideoDuration:
    def test_returns_correct_duration(self, sample_video: Path) -> None:
        duration = get_video_duration(sample_video)
        assert pytest.approx(duration, abs=0.5) == 3.0

    def test_accepts_string_path(self, sample_video: Path) -> None:
        duration = get_video_duration(str(sample_video))
        assert duration > 0

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Video not found"):
            get_video_duration(tmp_path / "nonexistent.mp4")


class TestIterSubclipFrames:
    def test_yields_numpy_arrays(self, sample_video: Path) -> None:
        frames = list(iter_subclip_frames(sample_video, 0.0, 1.0, every_n_frames=1))
        assert len(frames) > 0
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_frames_have_correct_shape(self, sample_video: Path) -> None:
        frames = list(iter_subclip_frames(sample_video, 0.0, 1.0, every_n_frames=1))
        assert frames[0].shape == (240, 320, 3)

    def test_every_n_frames_reduces_count(self, sample_video: Path) -> None:
        all_frames = list(iter_subclip_frames(sample_video, 0.0, 3.0, every_n_frames=1))
        sampled = list(iter_subclip_frames(sample_video, 0.0, 3.0, every_n_frames=3))
        assert len(sampled) < len(all_frames)
        assert len(sampled) == pytest.approx(len(all_frames) // 3, abs=2)

    def test_subclip_window_limits_frames(self, sample_video: Path) -> None:
        # A 1-second window at 10 FPS should give ~10 frames
        frames = list(iter_subclip_frames(sample_video, 1.0, 2.0, every_n_frames=1))
        assert 5 <= len(frames) <= 15

    def test_full_video_returns_all_frames(self, sample_video: Path) -> None:
        frames = list(iter_subclip_frames(sample_video, 0.0, 3.0, every_n_frames=1))
        # 30-frame video; allow small codec rounding
        assert 25 <= len(frames) <= 30

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Video not found"):
            list(iter_subclip_frames(tmp_path / "missing.mp4", 0.0, 1.0))

    def test_accepts_string_path(self, sample_video: Path) -> None:
        frames = list(iter_subclip_frames(str(sample_video), 0.0, 1.0, every_n_frames=1))
        assert len(frames) > 0
