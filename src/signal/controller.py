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
        deadlock_threshold: Max consecutive greens before deadlock override.
    """

    def __init__(self, deadlock_threshold: int = 2) -> None:
        self.deadlock_threshold = deadlock_threshold
        # TODO Phase 3: initialise DeadlockGuard here

    def decide(self, lanes: list[LaneState]) -> int:
        """
        Choose the winning lane for this round and update all LaneState objects.

        Args:
            lanes: Current state of all lanes.

        Returns:
            lane_id of the lane granted the green signal.
        """
        # TODO Phase 3: implement decision logic per CLAUDE.md spec
        # 1. find lane with max cumulative_count
        # 2. check deadlock guard — get override lane_id if threshold hit
        # 3. grant green to winner, red to all others
        # 4. reset winner.cumulative_count = 0
        # 5. update consecutive_greens
        raise NotImplementedError("Implement in Phase 3 — see CLAUDE.md")
