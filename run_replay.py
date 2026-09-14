"""
Step-1 smoke test: replay a scenario and print what the system sees.

    python run_replay.py data/scenario_03 --speed 0
    python run_replay.py data/scenario_03 --speed 0 --show-events
    python run_replay.py data/scenario_03                    # real 30s/day pacing

Nothing here is part of the agent system -- it exists so you can watch the
replay engine behave before anything depends on it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from c360.replay import ClockTick, EventTick, ReplayEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Customer 360 scenario.")
    parser.add_argument("scenario", type=Path)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="wall-clock seconds per simulated day; 0 = instant. Default: replay_config value.",
    )
    parser.add_argument("--show-events", action="store_true", help="print every event released")
    parser.add_argument("--show-ticks", action="store_true", help="print every daily clock tick")
    parser.add_argument(
        "--assert-safe",
        action="store_true",
        help="verify the no-future-leakage rule on every tick (slow, but it is the proof)",
    )
    args = parser.parse_args()

    engine = ReplayEngine(args.scenario, speed=args.speed).load()
    print(engine.describe())
    print("-" * 72)

    released = []
    quiet_days = 0
    checkpoints = 0

    for tick in engine.stream():
        now = engine.clock.now

        if isinstance(tick, EventTick):
            released.append(tick.event)
            if args.show_events:
                event = tick.event
                flag = "  <-- LATE ARRIVAL" if event.is_late_arrival else ""
                print(
                    f"  {now:%Y-%m-%d %H:%M}  {event.event_id}  "
                    f"{event.source_system:<22}{event.event_type:<22}"
                    f"-> {event.perception_agent or 'UNROUTED'}{flag}"
                )
        elif isinstance(tick, ClockTick):
            checkpoints += 1
            if tick.events_since_last_tick == 0:
                quiet_days += 1
            if args.show_ticks:
                print(
                    f"  {now:%Y-%m-%d}  [checkpoint]  "
                    f"{tick.events_since_last_tick} event(s) since last tick"
                )

        if args.assert_safe:
            for event in released:
                assert event.event_time <= now, f"FUTURE LEAK: {event.event_id} at {now}"
                assert event.release_time <= now, f"EARLY RELEASE: {event.event_id} at {now}"

    print("-" * 72)
    print(f"events released  : {len(released)}")
    print(f"checkpoints       : {checkpoints}")
    print(f"quiet days        : {quiet_days} (no events arrived -- only a clock tick saw them)")
    print(f"final sim time    : {engine.clock.now.isoformat()}")
    print(f"wall-clock pacing : {engine.clock.total_slept_seconds / 60:.1f} min requested")
    if args.assert_safe:
        print("no-future-leakage : VERIFIED on every tick")

    # Spot-check against the graded checkpoint times for this scenario.
    for label in ("2026-02-15T00:00:00Z", "2026-03-08T00:00:00Z", "2026-04-10T00:00:00Z"):
        as_of = datetime.fromisoformat(label.replace("Z", "+00:00")).astimezone(timezone.utc)
        if engine.config.simulated_start <= as_of <= engine.clock.now:
            visible = engine.visible_events(as_of)
            print(f"visible at {label}: {len(visible)} events")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
