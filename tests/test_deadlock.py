"""
Tests for signal/deadlock.py — DeadlockGuard in isolation.

Run with: pytest tests/test_deadlock.py -v
"""

import pytest
from src.signal.deadlock import DeadlockGuard


def _cg(*counts: int) -> dict[int, int]:
    """Build a consecutive-greens dict from positional lane counts."""
    return {i: c for i, c in enumerate(counts)}


class TestCheckOverride:
    def test_returns_none_below_threshold(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        assert guard.check_override(0, _cg(1, 0, 0, 0)) is None

    def test_returns_none_at_zero_consecutive(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        assert guard.check_override(0, _cg(0, 0, 0, 0)) is None

    def test_override_triggers_at_threshold(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        result = guard.check_override(0, _cg(2, 0, 0, 0))
        assert result is not None
        assert result != 0

    def test_override_triggers_above_threshold(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        result = guard.check_override(0, _cg(5, 0, 0, 0))
        assert result is not None
        assert result != 0

    def test_override_skips_already_greened_lanes(self) -> None:
        """If lane 1 already had green this round, override should skip it."""
        guard = DeadlockGuard(threshold=2, lane_count=4)
        guard.record_green(0)
        guard.record_green(1)
        result = guard.check_override(0, _cg(2, 0, 0, 0))
        assert result not in (0, 1)

    def test_override_picks_lowest_unserved_lane(self) -> None:
        """Override should pick the lowest lane_id not yet served this round."""
        guard = DeadlockGuard(threshold=2, lane_count=4)
        guard.record_green(0)
        result = guard.check_override(0, _cg(2, 0, 0, 0))
        assert result == 1  # lane 1 is the first unserved non-winner

    def test_round_resets_when_all_lanes_served(self) -> None:
        """When every other lane has been served, reset and restart the cycle."""
        guard = DeadlockGuard(threshold=2, lane_count=4)
        guard.record_green(0)
        guard.record_green(1)
        guard.record_green(2)
        guard.record_green(3)
        # All lanes served → reset happens internally; override returns lane 1
        result = guard.check_override(0, _cg(2, 0, 0, 0))
        assert result != 0
        assert result is not None

    def test_threshold_of_one(self) -> None:
        """threshold=1 means override triggers after a single consecutive green."""
        guard = DeadlockGuard(threshold=1, lane_count=2)
        result = guard.check_override(0, _cg(1, 0))
        assert result == 1


class TestRecordGreenAndReset:
    def test_record_green_tracked(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        guard.record_green(2)
        assert 2 in guard._round_greens

    def test_reset_round_clears_all(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        guard.record_green(0)
        guard.record_green(3)
        guard.reset_round()
        assert len(guard._round_greens) == 0

    def test_record_same_lane_twice_is_idempotent(self) -> None:
        guard = DeadlockGuard(threshold=2, lane_count=4)
        guard.record_green(1)
        guard.record_green(1)
        assert len(guard._round_greens) == 1
