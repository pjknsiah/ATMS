"""
Deadlock prevention — ensures no lane is indefinitely denied a green signal.

Algorithm:
    Track consecutive green counts per lane.
    If the natural winner has had >= DEADLOCK_THRESHOLD consecutive greens,
    return the next lane that has not received a green in the current round
    as an override winner instead.
"""

from __future__ import annotations


class DeadlockGuard:
    """
    Prevents signal starvation by overriding the natural winner when
    it has held the green for too many consecutive rounds.

    Args:
        threshold: Number of consecutive greens before override triggers.
        lane_count: Total number of lanes.
    """

    def __init__(self, threshold: int = 2, lane_count: int = 4) -> None:
        self.threshold = threshold
        self.lane_count = lane_count
        self._round_greens: set[int] = set()   # lane_ids that got green this round

    def check_override(
        self,
        natural_winner_id: int,
        consecutive_greens: dict[int, int],
    ) -> int | None:
        """
        Determine whether the natural winner should be overridden.

        Args:
            natural_winner_id: Lane id selected by max-count logic.
            consecutive_greens: Map of lane_id → consecutive green count.

        Returns:
            Override lane_id if deadlock detected, else None.
        """
        # TODO Phase 3: implement override logic
        # if consecutive_greens[natural_winner_id] >= self.threshold:
        #     return next lane_id not in self._round_greens
        # else:
        #     return None
        raise NotImplementedError("Implement in Phase 3 — see CLAUDE.md")

    def record_green(self, lane_id: int) -> None:
        """Record that lane_id received a green signal this round."""
        self._round_greens.add(lane_id)

    def reset_round(self) -> None:
        """
        Call at the start of each new full round (all lanes have had a turn).
        """
        self._round_greens.clear()
