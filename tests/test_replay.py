"""
Tests for the replay engine.

The important one is test_no_future_leakage_ever. It does not spot-check a few
timestamps -- it asserts the invariant on EVERY tick of the stream, so if any
future path through the code ever releases an event early, this fails.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c360.clock import ClockRegressionError, SimClock, daily_boundaries  # noqa: E402
from c360.replay import ClockTick, EventTick, ReplayEngine  # noqa: E402
from c360.schema import EventParseError, event_from_dict, load_events  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SCENARIO_01 = DATA / "scenario_01"
SCENARIO_02 = DATA / "scenario_02"
SCENARIO_03 = DATA / "scenario_03"
ALL_SCENARIOS = [SCENARIO_01, SCENARIO_02, SCENARIO_03]

UTC = timezone.utc

needs_data = pytest.mark.skipif(
    not all(s.exists() for s in ALL_SCENARIOS), reason="scenario data not present"
)
every_scenario = pytest.mark.parametrize(
    "scenario", ALL_SCENARIOS, ids=lambda p: p.name
)


def graded_checkpoints(scenario: Path) -> list[datetime]:
    """The as_of_times this scenario's ground_truth.json will be scored at."""
    gt = json.loads((scenario / "ground_truth.json").read_text())
    return [ts(c["as_of_time"]) for c in gt["checkpoints"]]


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def make_row(event_id: str, event_time: str, ingestion_time: str | None = None, **over):
    row = {
        "event_id": event_id,
        "event_time": event_time,
        "ingestion_time": ingestion_time or event_time,
        "customer_id": "CUST_TEST",
        "account_id": "ACC_TEST",
        "source_system": "card_payments",
        "event_type": "purchase",
        "schema_version": "1.0",
        "payload": {"amount": 10},
    }
    row.update(over)
    return row


def write_scenario(
    tmp_path: Path,
    live_rows: list[dict],
    history_rows: list[dict] | None = None,
    start: str = "2026-02-01T00:00:00Z",
    end: str = "2026-02-10T00:00:00Z",
    speed: float = 30,
) -> Path:
    """Build a throwaway scenario directory on disk."""
    scenario = tmp_path / "scenario_x"
    scenario.mkdir(parents=True)
    (scenario / "replay_config.json").write_text(
        json.dumps(
            {
                "scenario_id": "scenario_x",
                "simulated_start": start,
                "simulated_end": end,
                "replay_speed_seconds_per_simulated_day": speed,
            }
        )
    )
    (scenario / "entities.json").write_text(json.dumps({"customer_id": "CUST_TEST"}))
    for name, rows in (("live_stream.jsonl", live_rows), ("history_seed.jsonl", history_rows or [])):
        (scenario / name).write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return scenario


# ===========================================================================
# Timestamp parsing
# ===========================================================================

def test_parses_z_suffix_and_offset_identically():
    a = event_from_dict(make_row("A", "2026-02-01T12:00:00Z"))
    b = event_from_dict(make_row("B", "2026-02-01T12:00:00+00:00"))
    assert a.event_time == b.event_time


def test_non_utc_offset_is_normalised():
    """A +05:30 timestamp must compare correctly against a UTC one."""
    ist = event_from_dict(make_row("A", "2026-02-01T17:30:00+05:30"))
    utc = event_from_dict(make_row("B", "2026-02-01T12:00:00Z"))
    assert ist.event_time == utc.event_time
    # And this is exactly what naive string comparison would get WRONG:
    assert "2026-02-01T17:30:00+05:30" > "2026-02-01T12:00:00Z"


def test_missing_required_field_is_rejected():
    row = make_row("A", "2026-02-01T12:00:00Z")
    del row["source_system"]
    with pytest.raises(EventParseError, match="source_system"):
        event_from_dict(row)


def test_absent_ingestion_time_defaults_to_event_time():
    row = make_row("A", "2026-02-01T12:00:00Z")
    del row["ingestion_time"]
    event = event_from_dict(row)
    assert event.ingestion_time == event.event_time
    assert not event.is_late_arrival


def test_malformed_rows_are_quarantined_not_fatal(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text(
        json.dumps(make_row("GOOD_1", "2026-02-01T10:00:00Z"))
        + "\n{ this is not json\n"
        + json.dumps({"event_id": "BAD_2"})
        + "\n"
        + json.dumps(make_row("GOOD_2", "2026-02-02T10:00:00Z"))
        + "\n"
    )
    report = load_events(path)
    assert [e.event_id for e in report.events] == ["GOOD_1", "GOOD_2"]
    assert len(report.rejected) == 2
    assert not report.ok


def test_strict_mode_raises_on_bad_row(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text("{ not json\n")
    with pytest.raises(EventParseError):
        load_events(path, strict=True)


# ===========================================================================
# Clock
# ===========================================================================

def test_clock_refuses_to_go_backwards():
    clock = SimClock(ts("2026-02-01T00:00:00Z"))
    clock.advance_to(ts("2026-02-05T00:00:00Z"))
    with pytest.raises(ClockRegressionError):
        clock.advance_to(ts("2026-02-04T00:00:00Z"))


def test_clock_allows_advancing_to_the_same_instant():
    """Several real events share a timestamp to the second; this must not raise."""
    clock = SimClock(ts("2026-02-01T00:00:00Z"))
    clock.advance_to(ts("2026-02-01T09:00:00Z"))
    clock.advance_to(ts("2026-02-01T09:00:00Z"))
    assert clock.now == ts("2026-02-01T09:00:00Z")


def test_pacing_is_requested_proportionally():
    slept: list[float] = []
    clock = SimClock(ts("2026-02-01T00:00:00Z"), seconds_per_sim_day=30, sleep_fn=slept.append)
    clock.advance_to(ts("2026-02-03T00:00:00Z"))  # 2 simulated days
    clock.advance_to(ts("2026-02-03T12:00:00Z"))  # half a day
    assert slept == [60.0, 15.0]


def test_zero_speed_never_sleeps():
    slept: list[float] = []
    clock = SimClock(ts("2026-02-01T00:00:00Z"), seconds_per_sim_day=0, sleep_fn=slept.append)
    clock.advance_to(ts("2026-03-01T00:00:00Z"))
    assert slept == []


def test_daily_boundaries_are_inclusive_of_both_ends():
    days = list(daily_boundaries(ts("2026-02-01T00:00:00Z"), ts("2026-02-04T00:00:00Z")))
    assert days == [
        ts("2026-02-01T00:00:00Z"),
        ts("2026-02-02T00:00:00Z"),
        ts("2026-02-03T00:00:00Z"),
        ts("2026-02-04T00:00:00Z"),
    ]


# ===========================================================================
# THE HARD RULE
# ===========================================================================

@needs_data
@every_scenario
def test_no_future_leakage_ever(scenario):
    """
    At every single tick, nothing released so far may have an event_time later
    than simulated now.

    Asserted on every iteration rather than sampled, because the failure this
    guards against is exactly the kind that hides in an edge case. Run against
    all three scenarios, not just one.
    """
    engine = ReplayEngine(scenario, speed=0).load()
    released = []
    for tick in engine.stream():
        now = engine.clock.now
        if isinstance(tick, EventTick):
            released.append(tick.event)
        for event in released:
            assert event.event_time <= now, (
                f"{event.event_id} has event_time {event.event_time.isoformat()} "
                f"but simulated now is only {now.isoformat()}"
            )
            assert event.release_time <= now


def test_late_arrival_is_invisible_until_it_arrives(tmp_path):
    """
    An event that happened on Feb 2 but only arrived on Feb 7 must be invisible
    on Feb 3 -- even though its event_time has passed.

    This is availability leakage, and it is the case the provided sample data
    never exercises: every row in scenario_03 has ingestion_time == event_time,
    so this code path is only reachable via synthetic injection. The hidden
    evaluation set is where it would actually bite.
    """
    scenario = write_scenario(
        tmp_path,
        [
            make_row("ON_TIME", "2026-02-01T10:00:00Z"),
            make_row("LATE", "2026-02-02T10:00:00Z", ingestion_time="2026-02-07T10:00:00Z"),
            make_row("AFTER", "2026-02-03T10:00:00Z"),
        ],
    )
    engine = ReplayEngine(scenario, speed=0).load()

    # Release order follows arrival, not occurrence.
    order = [t.event_id for t in engine.stream() if isinstance(t, EventTick)]
    assert order == ["ON_TIME", "AFTER", "LATE"]

    # On Feb 3 the late event has already happened but has not arrived.
    visible = {e.event_id for e in engine.visible_events(ts("2026-02-03T12:00:00Z"))}
    assert visible == {"ON_TIME", "AFTER"}
    assert "LATE" not in visible

    # On Feb 8 it is visible, and it is correctly flagged as late.
    later = {e.event_id for e in engine.visible_events(ts("2026-02-08T00:00:00Z"))}
    assert later == {"ON_TIME", "AFTER", "LATE"}
    assert engine.live_report.late_arrivals == 1


def test_impossible_timestamp_is_floored_at_event_time(tmp_path):
    """
    ingestion_time BEFORE event_time is physically impossible. If it appears, the
    event must still not be released before it happened.
    """
    scenario = write_scenario(
        tmp_path,
        [make_row("CORRUPT", "2026-02-05T10:00:00Z", ingestion_time="2026-02-01T10:00:00Z")],
    )
    engine = ReplayEngine(scenario, speed=0).load()
    assert any("ingestion_time BEFORE event_time" in w for w in engine.warnings)

    ticks = [t for t in engine.stream() if isinstance(t, EventTick)]
    assert ticks[0].as_of == ts("2026-02-05T10:00:00Z")  # held to its event_time
    assert engine.visible_events(ts("2026-02-03T00:00:00Z")) == []


def test_shuffled_file_order_does_not_change_the_stream(tmp_path):
    """Robustness to an out-of-order file: ordering comes from timestamps, not lines."""
    rows = [
        make_row("E1", "2026-02-01T10:00:00Z"),
        make_row("E2", "2026-02-02T10:00:00Z"),
        make_row("E3", "2026-02-03T10:00:00Z"),
        make_row("E4", "2026-02-04T10:00:00Z"),
    ]
    ordered = ReplayEngine(write_scenario(tmp_path / "a", rows), speed=0)
    shuffled = ReplayEngine(
        write_scenario(tmp_path / "b", [rows[2], rows[0], rows[3], rows[1]]), speed=0
    )
    assert [t.event_id for t in ordered.stream() if isinstance(t, EventTick)] == [
        t.event_id for t in shuffled.stream() if isinstance(t, EventTick)
    ]


def test_history_after_simulated_start_is_held_back(tmp_path):
    """
    A history_seed row that strays into the live window must not be pre-loaded --
    that would hand an agent a future event before the stream even starts.
    """
    scenario = write_scenario(
        tmp_path,
        live_rows=[make_row("LIVE", "2026-02-02T10:00:00Z")],
        history_rows=[
            make_row("OLD", "2026-01-15T10:00:00Z"),
            make_row("STRAY", "2026-02-05T10:00:00Z"),
        ],
    )
    engine = ReplayEngine(scenario, speed=0).load()
    assert [e.event_id for e in engine.history] == ["OLD"]
    assert "STRAY" in [e.event_id for e in engine.live]
    assert any("history_seed event(s) at or after simulated_start" in w for w in engine.warnings)
    assert engine.visible_events(ts("2026-02-01T00:00:00Z")) == [engine.history[0]]


# ===========================================================================
# Clock ticks / checkpoint coverage
# ===========================================================================

@needs_data
@every_scenario
def test_every_day_in_the_window_gets_a_clock_tick(scenario):
    engine = ReplayEngine(scenario, speed=0).load()
    ticks = {t.as_of for t in engine.stream() if isinstance(t, ClockTick)}
    expected = set(
        daily_boundaries(engine.config.simulated_start, engine.config.simulated_end)
    )
    assert expected <= ticks


@needs_data
@every_scenario
def test_every_graded_checkpoint_is_reachable(scenario):
    """
    Read the required as_of_times straight out of each ground_truth.json rather
    than hard-coding them, so this keeps working if the data changes.

    scenario_03's Apr 10 checkpoint is the one at risk: its last live event is
    Apr 3, so an event-driven-only clock would stop twelve days short of it and
    silently never emit that row.
    """
    engine = ReplayEngine(scenario, speed=0).load()
    ticks = {t.as_of for t in engine.stream() if isinstance(t, ClockTick)}
    for when in graded_checkpoints(scenario):
        assert when in ticks, f"{scenario.name}: no checkpoint emitted at {when.isoformat()}"


@needs_data
def test_signals_are_visible_by_their_checkpoint_and_not_before():
    """
    Ground truth says the Mar 8 checkpoint must reach HIGH confidence on the back
    of the standing-instruction cancellation (EVT_000457, Mar 4) and the savings
    transfer (EVT_000461, Mar 6). Both must be visible at Mar 8 -- and neither at
    the Feb 15 checkpoint, where the expected confidence is LOW.
    """
    engine = ReplayEngine(SCENARIO_03, speed=0).load()

    feb15 = {e.event_id for e in engine.visible_events(ts("2026-02-15T00:00:00Z"))}
    assert "EVT_000409" in feb15  # complaint raised Feb 5
    assert "EVT_000412" in feb15  # complaint denied Feb 6
    assert "EVT_000457" not in feb15  # SI cancellation is still 17 days away
    assert "EVT_000461" not in feb15

    mar8 = {e.event_id for e in engine.visible_events(ts("2026-03-08T00:00:00Z"))}
    assert {"EVT_000457", "EVT_000461"} <= mar8
    assert "EVT_000469" not in mar8  # Mar 20 -- must not leak backwards


@needs_data
def test_boundary_event_is_inclusive_at_its_own_checkpoint():
    """
    An event at exactly 00:00:00 belongs to the checkpoint taken at that instant,
    because the filter is event_time <= as_of. This pins the merge tie-break.
    """
    engine = ReplayEngine(SCENARIO_03, speed=0).load()
    seen: list[str] = []
    for tick in engine.stream():
        if isinstance(tick, EventTick):
            seen.append(tick.event_id)
        elif tick.as_of == ts("2026-03-06T00:00:00Z"):
            # Everything released up to this midnight must match visible_events.
            expected = {e.event_id for e in engine.visible_events(tick.as_of)}
            assert set(seen) | {e.event_id for e in engine.history} == expected
            break


# ===========================================================================
# Pacing must not change semantics
# ===========================================================================

@needs_data
@every_scenario
def test_speed_does_not_affect_output(scenario):
    """
    A paced run and an instant run must produce an identical tick sequence.
    If this ever fails, correctness has become dependent on wall-clock sleeping.
    """
    def fingerprint(speed):
        slept: list[float] = []
        engine = ReplayEngine(scenario, speed=speed, sleep_fn=slept.append).load()
        seq = [
            (t.as_of.isoformat(), getattr(t, "event_id", "CLOCK")) for t in engine.stream()
        ]
        return seq, sum(slept)

    fast, fast_slept = fingerprint(0)
    paced, paced_slept = fingerprint(30)
    assert fast == paced
    assert fast_slept == 0
    # 73-day window at 30s/day would be a ~37 minute demo.
    assert paced_slept == pytest.approx(73 * 30, abs=120)


@needs_data
@every_scenario
def test_stream_is_deterministic_across_runs(scenario):
    """The scoring harness depends on identical output from identical input."""
    def run():
        return [
            (t.as_of.isoformat(), getattr(t, "event_id", "CLOCK"))
            for t in ReplayEngine(scenario, speed=0).load().stream()
        ]

    assert run() == run()


@needs_data
@every_scenario
def test_all_live_events_are_released_exactly_once(scenario):
    engine = ReplayEngine(scenario, speed=0).load()
    released = [t.event_id for t in engine.stream() if isinstance(t, EventTick)]
    assert len(released) == len(engine.live)
    assert len(set(released)) == len(released)


@needs_data
@every_scenario
def test_history_is_entirely_before_the_live_window(scenario):
    """If this ever fails, backstory is leaking into the graded window."""
    engine = ReplayEngine(scenario, speed=0).load()
    assert engine.history, f"{scenario.name} has no history_seed events"
    assert max(e.event_time for e in engine.history) < engine.config.simulated_start


@needs_data
@every_scenario
def test_every_event_routes_to_a_perception_agent(scenario):
    """
    A source_system with no route would be silently invisible to the whole
    perception layer. Catch it here rather than wondering later why a signal
    never fired.
    """
    engine = ReplayEngine(scenario, speed=0).load()
    unrouted = {
        e.source_system for e in engine.history + engine.live if e.perception_agent is None
    }
    assert not unrouted, f"{scenario.name}: unrouted source_system(s) {sorted(unrouted)}"


# ===========================================================================
# The real late arrivals -- planted signal events in scenarios 01 and 02
# ===========================================================================

@needs_data
@pytest.mark.parametrize(
    "scenario, event_id, as_of_hidden, as_of_visible",
    [
        # scenario_01: a $450 Quest Diagnostics healthcare charge -- a
        # medical_hardship SIGNAL event -- occurs Mar 1 but arrives Mar 3.
        (SCENARIO_01, "EVT_000420", "2026-03-02T00:00:00Z", "2026-03-04T00:00:00Z"),
        # scenario_02: a Mothercare purchase -- a new_child_life_event SIGNAL
        # event -- occurs Mar 2 but arrives Mar 3.
        (SCENARIO_02, "EVT_000341", "2026-03-03T00:00:00Z", "2026-03-04T00:00:00Z"),
    ],
    ids=["scenario_01_quest_diagnostics", "scenario_02_mothercare"],
)
def test_real_late_arriving_signal_events(scenario, event_id, as_of_hidden, as_of_visible):
    """
    The dataset plants its late-arriving event ON a ground-truth signal event,
    not on background noise. Both of these appear in their scenario's
    `signal_events` list.

    That makes ingestion-time handling load-bearing rather than cosmetic: a
    system that sorts purely by event_time would claim to have known about a
    $450 diagnostics charge on Mar 1 when the bank did not receive it until
    Mar 3. The inference might still land on the right day, but any lead-time
    measurement built on it is wrong -- and scenario_03's ground truth grades
    lead time explicitly.
    """
    gt = json.loads((scenario / "ground_truth.json").read_text())
    assert event_id in gt["signal_events"], "precondition: this is a graded signal event"

    engine = ReplayEngine(scenario, speed=0).load()
    event = next(e for e in engine.live if e.event_id == event_id)
    assert event.is_late_arrival
    assert event.arrival_lag_seconds > 0

    # Already happened, not yet received -> must be invisible.
    hidden = {e.event_id for e in engine.visible_events(ts(as_of_hidden))}
    assert event.event_time <= ts(as_of_hidden), "precondition: it has already occurred"
    assert event_id not in hidden, "late-arriving event leaked before it arrived"

    # After it arrives -> visible.
    visible = {e.event_id for e in engine.visible_events(ts(as_of_visible))}
    assert event_id in visible

    # And it is released at arrival time, not occurrence time.
    tick = next(
        t for t in engine.stream() if isinstance(t, EventTick) and t.event_id == event_id
    )
    assert tick.as_of == event.ingestion_time
    assert tick.as_of > event.event_time
