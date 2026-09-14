"""
End-to-end run: replay engine -> decision -> inferred-events file.

    python run_pipeline.py data/scenario_03 --speed 0
    python run_pipeline.py data/scenario_03 --speed 0 --out out/

Right now the "decision" is a DUMMY rule that always says no_significant_event.
That is deliberate and is the whole point of step 2: it proves the plumbing --
replay, PII redaction, checkpoint validation, file writing, graded-timestamp
coverage -- works before any agent exists to blame.

In step 5 the dummy_decision() function below gets replaced by the Synthesis
Agent. Nothing else in this file has to change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from c360.output import Checkpoint, InferredEventsWriter  # noqa: E402
from c360.pii import Redactor  # noqa: E402
from c360.replay import ClockTick, EventTick, ReplayEngine  # noqa: E402
from c360.schema import Action, ConfidenceBand, Event, HitlStatus, InferredState  # noqa: E402


def dummy_decision(as_of, events_today: list[Event], redactor: Redactor) -> Checkpoint:
    """
    Placeholder for the Synthesis + Action + Critique cascade.

    It reports only what it can see without reasoning: how many events arrived,
    which source systems they came from, and their ids. Always no_action -- an
    honest "I have no opinion" rather than a fake one.

    Note the notes field already cites event_ids. That is not decoration: the
    writer REFUSES any row that proposes an action without citations, so getting
    into the habit here means the real agents inherit a format that already
    satisfies the graded explainability requirement.
    """
    if not events_today:
        notes = "No events observed since the previous checkpoint."
    else:
        systems = sorted({e.source_system for e in events_today})
        ids = ", ".join(e.event_id for e in events_today[:5])
        more = f" (+{len(events_today) - 5} more)" if len(events_today) > 5 else ""
        notes = f"Observed {len(events_today)} event(s) across {', '.join(systems)}: {ids}{more}."

    return Checkpoint(
        as_of_time=as_of,
        inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
        confidence_band=ConfidenceBand.LOW,
        action=Action.NO_ACTION,
        action_subtype=None,
        hitl_status=HitlStatus.AUTO_APPROVED,
        notes=redactor.scrub(notes),
        citations=tuple(e.event_id for e in events_today),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a scenario end to end.")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--speed", type=float, default=0)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    engine = ReplayEngine(args.scenario, speed=args.speed).load()
    redactor = Redactor.from_entities(engine.entities)
    writer = InferredEventsWriter(redactor=redactor, scenario_id=engine.config.scenario_id)

    print(engine.describe())
    print("-" * 72)

    # History is backstory: loaded up front, all of it before simulated_start.
    # In step 3 this is where episodic memory gets seeded.
    print(f"seeded {len(engine.history)} history events as backstory")

    events_today: list[Event] = []
    late_seen = 0
    for tick in engine.stream():
        if isinstance(tick, EventTick):
            events_today.append(tick.event)
            if tick.event.is_late_arrival:
                late_seen += 1
                print(
                    f"  LATE ARRIVAL {tick.event.event_id}: occurred "
                    f"{tick.event.event_time:%Y-%m-%d %H:%M}, received "
                    f"{tick.event.ingestion_time:%Y-%m-%d %H:%M} "
                    f"({tick.event.arrival_lag_seconds / 3600:.0f}h late)"
                )
        elif isinstance(tick, ClockTick):
            writer.add(dummy_decision(tick.as_of, events_today, redactor))
            events_today = []

    path = writer.write(args.out / f"{args.scenario.name}_inferred_events.json")

    print("-" * 72)
    print(writer.summary())
    print(f"late arrivals observed: {late_seen}")
    print(f"written: {path}")

    # The check that matters: does every graded timestamp have a row?
    gt_path = args.scenario / "ground_truth.json"
    if gt_path.exists():
        from datetime import datetime, timezone

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        required = [
            datetime.fromisoformat(c["as_of_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
            for c in gt["checkpoints"]
        ]
        missing = writer.covers(required)
        if missing:
            print(f"MISSING graded checkpoints: {[m.isoformat() for m in missing]}")
            return 1
        print(f"graded checkpoint coverage: {len(required)}/{len(required)} present")
        print("   (coverage only -- the dummy rule gets every ANSWER wrong, as expected)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
