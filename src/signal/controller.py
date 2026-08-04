"""
Signal controller — decides which lane gets the green signal each round.

Responsibilities:
- Accept current LaneState list
- Determine winner by highest cumulative_count
- Delegate to DeadlockGuard for override if threshold hit
- Update LaneState objects (reset winner count, update signals)
- Return the winning lane_id
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time

from src.signal.deadlock import DeadlockGuard


@dataclass
class LaneState:
    """Shared state for one lane, updated each processing window."""

    lane_id: int
    cumulative_count: int = 0
    consecutive_greens: int = 0
    current_signal: str = "red"        # "green" | "red"
    last_updated: float = field(default_factory=time.time)


class SignalController:
    """
    Determines which lane receives a green signal each round.

    Args:
        deadlock_threshold: Max consecutive greens before deadlock override kicks in.
    """

    def __init__(self, deadlock_threshold: int = 2) -> None:
        self.deadlock_threshold = deadlock_threshold
        self._guard: DeadlockGuard | None = None  # lazy-init on first decide()

    def decide(self, lanes: list[LaneState]) -> int:
        """
        Choose the winning lane for this round and update all LaneState objects.

        Args:
            lanes: Current state of all lanes.

        Returns:
            lane_id of the lane granted the green signal.
        """
        if self._guard is None:
            self._guard = DeadlockGuard(
                threshold=self.deadlock_threshold,
                lane_count=len(lanes),
            )

        # 1. Natural winner — lane with the highest cumulative vehicle count.
        natural_winner = max(lanes, key=lambda l: l.cumulative_count)

        # 2. Check deadlock guard; it returns an override lane_id or None.
        consecutive_greens = {l.lane_id: l.consecutive_greens for l in lanes}
        override_id = self._guard.check_override(natural_winner.lane_id, consecutive_greens)
        winner_id = override_id if override_id is not None else natural_winner.lane_id

        # 3–5. Apply the decision to each lane.
        for lane in lanes:
            if lane.lane_id == winner_id:
                lane.current_signal = "green"
                lane.cumulative_count = 0
                lane.consecutive_greens += 1
                lane.last_updated = time.time()
            else:
                lane.current_signal = "red"
                lane.consecutive_greens = 0

        self._guard.record_green(winner_id)
        return winner_id
