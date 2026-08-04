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
        lane_count: Total number of lanes (informational; logic derives IDs
                    from the consecutive_greens dict passed to check_override).
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
        if consecutive_greens.get(natural_winner_id, 0) < self.threshold:
            return None

        lane_ids = sorted(consecutive_greens)

        # Pick the first lane that hasn't had green this round.
        for lane_id in lane_ids:
            if lane_id not in self._round_greens and lane_id != natural_winner_id:
                return lane_id

        # Every other lane has already been served this round — start a fresh round.
        self.reset_round()
        for lane_id in lane_ids:
            if lane_id != natural_winner_id:
                return lane_id

        return None  # edge case: only one lane exists

    def record_green(self, lane_id: int) -> None:
        """Record that lane_id received a green signal this round."""
        self._round_greens.add(lane_id)

    def reset_round(self) -> None:
        """Clear the per-round green log at the start of a new full rotation."""
        self._round_greens.clear()
