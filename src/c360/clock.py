"""
The simulated clock.

WHY THIS FILE EXISTS
--------------------
There are two different clocks in this system and confusing them is the classic
way a replay harness ends up cheating:

  * SIMULATED time -- the customer's timeline, Feb 1 to Apr 15 2026. Every
    decision, every memory query, every checkpoint is stamped in this time.
  * WALL-CLOCK time -- how long the demo takes to watch. Purely cosmetic.

SimClock owns simulated time and treats wall-clock pacing as an optional side
effect. That separation is what lets the test suite run the whole scenario in
milliseconds and still assert that the result is byte-identical to a paced run.
If correctness depended on real sleeping, the tests could not prove anything.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable

SECONDS_PER_DAY = 86_400.0


class ClockRegressionError(RuntimeError):
    """
    Raised when something tries to move simulated time backwards.

    This is the load-bearing guard of the replay engine. Simulated 'now' moving
    backwards would mean an agent could observe a state, then be handed an
    earlier state -- which is exactly the future-leakage failure mode in reverse,
    and would corrupt every checkpoint after it. We refuse loudly rather than
    tolerate it.
    """


class SimClock:
    """
    A monotonic clock over simulated time, with optional wall-clock pacing.

    Args:
        start: simulated time at which the clock begins.
        seconds_per_sim_day: wall-clock seconds to spend per simulated day.
            Taken from replay_config.json (30 for these scenarios). Set to 0 to
            run as fast as the CPU allows -- the simulated timeline is unchanged.
        sleep_fn: injected so tests can substitute a recorder instead of really
            sleeping. Dependency injection here is not over-engineering; it is
            the only way to assert that pacing was REQUESTED correctly without
            making the test suite take 37 minutes.
    """

    def __init__(
        self,
        start: datetime,
        seconds_per_sim_day: float = 0.0,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if start.tzinfo is None:
            raise ValueError("SimClock requires a timezone-aware start time")
        self._now = start
        self._start = start
        self.seconds_per_sim_day = float(seconds_per_sim_day)
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self.total_slept_seconds = 0.0
        self.advance_count = 0

    @property
    def now(self) -> datetime:
        """Current simulated time. The only definition of 'now' in the system."""
        return self._now

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def elapsed(self) -> timedelta:
        return self._now - self._start

    def advance_to(self, target: datetime) -> None:
        """
        Move simulated time forward to `target`, pacing in wall-clock if asked.

        Advancing to the CURRENT time is allowed and is a no-op: several events
        in this dataset share a timestamp to the second (EVT_000453/454 for
        example), and each of them legitimately advances the clock to the same
        instant. Going backwards is not allowed and raises.
        """
        if target.tzinfo is None:
            raise ValueError("advance_to requires a timezone-aware datetime")
        if target < self._now:
            raise ClockRegressionError(
                f"refusing to move simulated time backwards: "
                f"now={self._now.isoformat()} target={target.isoformat()}"
            )

        delta = (target - self._now).total_seconds()
        if delta > 0 and self.seconds_per_sim_day > 0:
            sleep_for = (delta / SECONDS_PER_DAY) * self.seconds_per_sim_day
            self.total_slept_seconds += sleep_for
            self._sleep(sleep_for)

        self._now = target
        self.advance_count += 1

    def __repr__(self) -> str:
        return f"SimClock(now={self._now.isoformat()}, speed={self.seconds_per_sim_day}s/day)"


def daily_boundaries(start: datetime, end: datetime, step: timedelta = timedelta(days=1)):
    """
    Yield every simulated midnight in [start, end], inclusive of both ends.

    WHY THIS MATTERS -- two independent reasons, either one sufficient:

    1. GRADED CHECKPOINTS EXIST IN EMPTY TIME. In scenario_03 the last live event
       is EVT_000471 on Apr 3, but replay_config.simulated_end is Apr 15 and the
       ground truth expects a checkpoint at 2026-04-10T00:00:00Z. A clock driven
       only by events never reaches Apr 10, so that checkpoint would simply never
       be emitted -- a scored row lost to an off-by-one in the loop structure,
       with nothing in the logs to suggest anything went wrong.

    2. ABSENCE IS A SIGNAL. "Card usage drops to zero" and "digital engagement
       falls away" are detectable only by noticing that nothing arrived. An
       event-driven-only system is structurally incapable of observing silence,
       because silence produces no event to react to. The daily tick is what
       gives the perception layer something to wake up on.

    `step` is a parameter rather than a hard-coded day so the cadence can be
    tightened later without touching the merge logic.
    """
    if start > end:
        return
    current = start
    while current <= end:
        yield current
        current += step
