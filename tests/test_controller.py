"""
Tests for signal/controller.py and signal/deadlock.py.

Run with: pytest tests/test_controller.py -v
"""

import pytest
from src.signal.controller import LaneState, SignalController


def make_lanes(counts: list[int]) -> list[LaneState]:
    """Helper — create LaneState list from a list of counts."""
    return [LaneState(lane_id=i, cumulative_count=c) for i, c in enumerate(counts)]


class TestSignalController:

    def test_winner_is_max_count_lane(self):
        """Lane with highest count should receive green."""
        lanes = make_lanes([5, 12, 3, 8])
        ctrl = SignalController(deadlock_threshold=2)
        winner = ctrl.decide(lanes)
        assert winner == 1  # lane_id 1 has count 12

    def test_winner_count_resets_to_zero(self):
        """Winner's cumulative_count must be 0 after decision."""
        lanes = make_lanes([2, 9, 1, 4])
        ctrl = SignalController(deadlock_threshold=2)
        winner = ctrl.decide(lanes)
        winning_lane = next(l for l in lanes if l.lane_id == winner)
        assert winning_lane.cumulative_count == 0

    def test_loser_counts_are_preserved(self):
        """Non-winning lanes must retain their cumulative counts."""
        lanes = make_lanes([3, 10, 5, 7])
        ctrl = SignalController(deadlock_threshold=2)
        winner = ctrl.decide(lanes)
        for lane in lanes:
            if lane.lane_id != winner:
                assert lane.cumulative_count > 0

    def test_winner_gets_green_signal(self):
        """Winner lane.current_signal must be 'green'."""
        lanes = make_lanes([1, 1, 15, 2])
        ctrl = SignalController(deadlock_threshold=2)
        winner = ctrl.decide(lanes)
        winning_lane = next(l for l in lanes if l.lane_id == winner)
        assert winning_lane.current_signal == "green"

    def test_non_winners_get_red_signal(self):
        """All non-winning lanes must have current_signal == 'red'."""
        lanes = make_lanes([1, 1, 15, 2])
        ctrl = SignalController(deadlock_threshold=2)
        winner = ctrl.decide(lanes)
        for lane in lanes:
            if lane.lane_id != winner:
                assert lane.current_signal == "red"


class TestDeadlockPrevention:

    def test_deadlock_override_triggers_at_threshold(self):
        """
        If the natural winner has consecutive_greens >= threshold,
        a different lane should win.
        """
        lanes = make_lanes([10, 1, 1, 1])
        lanes[0].consecutive_greens = 2  # already at threshold
        ctrl = SignalController(deadlock_threshold=2)
        winner = ctrl.decide(lanes)
        assert winner != 0  # lane 0 should NOT win again

    def test_all_lanes_eventually_receive_green(self):
        """
        Over enough rounds with equal counts, every lane must
        receive at least one green signal.
        """
        lanes = make_lanes([5, 5, 5, 5])
        ctrl = SignalController(deadlock_threshold=2)
        winners = set()
        for _ in range(12):
            # give all lanes equal count each round
            for lane in lanes:
                if lane.current_signal == "red":
                    lane.cumulative_count += 5
            winners.add(ctrl.decide(lanes))
        assert winners == {0, 1, 2, 3}

    def test_no_starvation_under_dominant_lane(self):
        """
        Even when lane 0 always has the most vehicles, other lanes
        must still receive green signals over time.
        """
        ctrl = SignalController(deadlock_threshold=2)
        lanes = make_lanes([0, 0, 0, 0])
        winners = []
        for _ in range(20):
            lanes[0].cumulative_count += 20   # lane 0 always dominant
            for i in range(1, 4):
                lanes[i].cumulative_count += 2
            winners.append(ctrl.decide(lanes))
        # lanes 1, 2, 3 must have won at least once
        assert 1 in winners
        assert 2 in winners
        assert 3 in winners
