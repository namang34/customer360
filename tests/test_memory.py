"""
Tests for episodic memory and the state board.

The headline test is test_memory_agrees_with_replay_engine_on_every_checkpoint.
EpisodicMemory (a SQL WHERE clause) and ReplayEngine.visible_events (a Python
list comprehension) are INDEPENDENT implementations of the same predicate. If
they agree on every checkpoint of every scenario, that is evidence rather than
tautology -- a bug would have to exist identically in both to go unnoticed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c360.findings import Finding, SignalStrength  # noqa: E402
from c360.memory import EpisodicMemory, TemporalLeakError  # noqa: E402
from c360.output import Checkpoint  # noqa: E402
from c360.replay import ReplayEngine  # noqa: E402
from c360.schema import Action, ConfidenceBand, Event, HitlStatus, InferredState  # noqa: E402
from c360.state_board import StateBoard  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = [DATA / f"scenario_0{n}" for n in (1, 2, 3)]

needs_data = pytest.mark.skipif(
    not all(s.exists() for s in ALL_SCENARIOS), reason="scenario data not present"
)
every_scenario = pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def make_event(event_id, event_time, ingestion_time=None, source_system="card_payments", **over):
    kwargs = dict(
        event_id=event_id,
        event_time=ts(event_time),
        ingestion_time=ts(ingestion_time or event_time),
        customer_id="CUST_TEST",
        account_id="ACC_TEST",
        source_system=source_system,
        event_type="purchase",
        schema_version="1.0",
        payload={"amount": 10},
    )
    kwargs.update(over)
    return Event(**kwargs)


def make_finding(agent, as_of, signal="x", systems=("card_payments",), strength=SignalStrength.MODERATE):
    return Finding(
        agent=agent,
        as_of=ts(as_of),
        signal=signal,
        strength=strength,
        event_ids=("EVT_000001",),
        source_systems=systems,
        detail="test finding",
    )


@pytest.fixture
def memory():
    with EpisodicMemory(customer_id="CUST_TEST") as store:
        yield store


# ===========================================================================
# The mandatory as_of argument
# ===========================================================================

def test_as_of_is_required_and_has_no_default(memory):
    """
    The whole safety design rests on a caller being unable to forget `as_of`.
    If someone ever gives it a default, this fails.
    """
    with pytest.raises(TypeError):
        memory.events_as_of()  # type: ignore[call-arg]


def test_naive_datetime_is_refused(memory):
    with pytest.raises(ValueError, match="timezone-aware"):
        memory.events_as_of(datetime(2026, 3, 1))


def test_non_datetime_as_of_is_refused(memory):
    with pytest.raises(TypeError):
        memory.events_as_of("2026-03-01T00:00:00Z")


def test_naive_datetime_is_refused_on_write(memory):
    with pytest.raises(ValueError, match="naive"):
        memory.record_event(
            Event(
                event_id="E",
                event_time=datetime(2026, 3, 1),
                ingestion_time=datetime(2026, 3, 1),
                customer_id="CUST_TEST",
                account_id=None,
                source_system="card_payments",
                event_type="purchase",
                schema_version="1.0",
                payload={},
            )
        )


# ===========================================================================
# The visibility predicate
# ===========================================================================

def test_future_events_are_invisible(memory):
    memory.record_events([
        make_event("PAST", "2026-02-01T10:00:00Z"),
        make_event("FUTURE", "2026-03-01T10:00:00Z"),
    ])
    visible = memory.events_as_of(ts("2026-02-15T00:00:00Z"))
    assert [e.event_id for e in visible] == ["PAST"]


def test_late_arrival_is_invisible_until_received(memory):
    """
    The availability-leakage case, modelled on scenario_01's EVT_000420: a $450
    diagnostics charge that occurred 1 March but only arrived 3 March.
    """
    memory.record_event(
        make_event("LATE", "2026-03-01T09:00:00Z", ingestion_time="2026-03-03T09:00:00Z")
    )
    assert memory.events_as_of(ts("2026-03-02T00:00:00Z")) == []
    assert memory.events_as_of(ts("2026-03-03T12:00:00Z"))[0].event_id == "LATE"


def test_boundary_is_inclusive(memory):
    memory.record_event(make_event("EXACT", "2026-03-08T00:00:00Z"))
    assert len(memory.events_as_of(ts("2026-03-08T00:00:00Z"))) == 1
    assert memory.events_as_of(ts("2026-03-07T23:59:59Z")) == []


def test_since_days_can_only_narrow_never_extend(memory):
    memory.record_events([
        make_event("OLD", "2026-01-01T10:00:00Z"),
        make_event("RECENT", "2026-03-01T10:00:00Z"),
        make_event("FUTURE", "2026-04-01T10:00:00Z"),
    ])
    ids = {e.event_id for e in memory.events_as_of(ts("2026-03-05T00:00:00Z"), since_days=30)}
    assert ids == {"RECENT"}  # OLD is outside the window, FUTURE is invisible


def test_source_system_and_event_type_filters(memory):
    memory.record_events([
        make_event("CARD", "2026-02-01T10:00:00Z", source_system="card_payments"),
        make_event("WEB", "2026-02-02T10:00:00Z", source_system="web_app_events", event_type="login"),
    ])
    now = ts("2026-03-01T00:00:00Z")
    assert [e.event_id for e in memory.events_as_of(now, source_systems=["web_app_events"])] == ["WEB"]
    assert [e.event_id for e in memory.events_as_of(now, event_types=["login"])] == ["WEB"]


def test_duplicate_event_ids_do_not_overwrite(memory):
    """The first copy wins; a later duplicate must not rewrite history."""
    memory.record_event(make_event("DUP", "2026-02-01T10:00:00Z"))
    memory.record_event(make_event("DUP", "2026-03-01T10:00:00Z"))
    stored = memory.events_as_of(ts("2026-04-01T00:00:00Z"))
    assert len(stored) == 1
    assert stored[0].event_time == ts("2026-02-01T10:00:00Z")


def test_non_utc_input_is_normalised_on_store_and_query(memory):
    """
    SQLite compares stored timestamps as text. One row written with a different
    offset would sort into the wrong place and corrupt every temporal filter.
    """
    memory.record_event(make_event("IST", "2026-03-01T17:30:00+05:30"))  # == 12:00Z
    assert memory.events_as_of(ts("2026-03-01T12:00:00Z"))[0].event_id == "IST"
    assert memory.events_as_of(ts("2026-03-01T11:59:00Z")) == []
    # And the same instant expressed in another zone must give the same answer.
    assert len(memory.events_as_of(ts("2026-03-01T17:30:00+05:30"))) == 1


# ===========================================================================
# Cross-validation against the replay engine
# ===========================================================================

@needs_data
@every_scenario
def test_memory_agrees_with_replay_engine_on_every_checkpoint(scenario):
    """
    Two independent implementations of the same predicate, compared at all 74
    checkpoints. A SQL WHERE clause and a Python list comprehension would have to
    be wrong in exactly the same way for this to pass by accident.
    """
    engine = ReplayEngine(scenario, speed=0).load()
    with EpisodicMemory(customer_id=engine.entities["customer_id"]) as memory:
        memory.record_events(engine.history)
        memory.record_events(engine.live)  # store everything; the FILTER does the work

        for as_of in engine.checkpoint_times:
            from_memory = {e.event_id for e in memory.events_as_of(as_of)}
            from_engine = {e.event_id for e in engine.visible_events(as_of)}
            assert from_memory == from_engine, f"disagreement at {as_of.isoformat()}"


@needs_data
@every_scenario
def test_storing_the_whole_file_up_front_still_leaks_nothing(scenario):
    """
    Deliberately the WORST case: every event, including April's, written to the
    database before the run starts. If the filter is right, the early checkpoints
    are unaffected. This is the test that proves safety comes from the query and
    not from withholding data.
    """
    engine = ReplayEngine(scenario, speed=0).load()
    with EpisodicMemory(customer_id=engine.entities["customer_id"]) as memory:
        memory.record_events(engine.history + engine.live)
        first = engine.checkpoint_times[0]
        visible = memory.events_as_of(first)
        assert all(e.event_time <= first and e.release_time <= first for e in visible)
        assert len(visible) < memory.stats()["events"]


@needs_data
def test_baseline_excludes_the_recent_window():
    """
    The baseline must not be dragged down by the very collapse it is measuring.
    scenario_03's David Chen logs in regularly through the history seed, then
    stops -- the baseline should still reflect the regular period.
    """
    engine = ReplayEngine(DATA / "scenario_03", speed=0).load()
    with EpisodicMemory(customer_id="CUST_00184") as memory:
        memory.record_events(engine.history + engine.live)
        as_of = ts("2026-03-08T00:00:00Z")
        recent = memory.daily_rate(as_of, source_systems=["web_app_events"], window_days=14)
        baseline = memory.baseline_daily_rate(
            as_of, source_systems=["web_app_events"], baseline_days=90, exclude_recent_days=14
        )
        assert baseline > recent, "engagement collapse should show as recent < baseline"


# ===========================================================================
# State board
# ===========================================================================

def test_board_rejects_an_unknown_agent_name(memory):
    board = StateBoard(memory)
    with pytest.raises(ValueError, match="unknown agent"):
        board.publish(make_finding("transction", "2026-03-01T00:00:00Z"))  # typo


def test_synthesis_trigger_needs_two_distinct_agents(memory):
    board = StateBoard(memory)
    now = ts("2026-03-08T00:00:00Z")

    board.publish(make_finding("transaction", "2026-03-06T00:00:00Z"))
    assert board.should_synthesise(now) is False, "one agent is not a correlation"

    board.publish(make_finding("transaction", "2026-03-07T00:00:00Z", signal="y"))
    assert board.should_synthesise(now) is False, "same agent twice is still one opinion"

    board.publish(make_finding("usage", "2026-03-07T00:00:00Z"))
    assert board.should_synthesise(now) is True


def test_trigger_respects_the_window(memory):
    board = StateBoard(memory, window_days=14)
    board.publish(make_finding("transaction", "2026-01-01T00:00:00Z"))
    board.publish(make_finding("usage", "2026-03-07T00:00:00Z"))
    assert board.should_synthesise(ts("2026-03-08T00:00:00Z")) is False


def test_guardrail_counts_source_systems_not_events(memory):
    """
    The red-herring defence. Three events from one system prove nothing; two
    systems corroborate. Modelled on scenario_03's tax refund (EVT_000447):
    one deposit, one source system, no support.
    """
    board = StateBoard(memory)
    now = ts("2026-03-08T00:00:00Z")

    board.publish(
        Finding(
            agent="transaction",
            as_of=ts("2026-03-07T00:00:00Z"),
            signal="large_inbound_deposit",
            strength=SignalStrength.STRONG,
            event_ids=("EVT_000447", "EVT_000446", "EVT_000448"),
            source_systems=("core_banking_ledger",),
            detail="tax refund",
        )
    )
    corroboration = board.corroboration(now)
    assert corroboration.independent_source_count == 1
    assert corroboration.is_corroborated is False
    assert "UNCORROBORATED" in corroboration.explain()

    board.publish(make_finding("usage", "2026-03-07T00:00:00Z", systems=("web_app_events",)))
    assert board.corroboration(now).is_corroborated is True


def test_duplicate_source_systems_cannot_inflate_the_count(memory):
    board = StateBoard(memory)
    board.publish(
        make_finding(
            "transaction", "2026-03-07T00:00:00Z", systems=("card_payments", "card_payments")
        )
    )
    assert board.corroboration(ts("2026-03-08T00:00:00Z")).independent_source_count == 1


def test_one_agent_can_corroborate_across_its_own_source_systems(memory):
    """
    The Transaction Agent covers four source systems. A salary stop plus a savings
    drawdown is two independent streams of evidence even though one agent saw
    both -- which is exactly why the guardrail counts systems, not agents.
    """
    board = StateBoard(memory)
    board.publish(
        make_finding(
            "transaction",
            "2026-03-07T00:00:00Z",
            systems=("core_banking_ledger", "instant_payments"),
        )
    )
    corroboration = board.corroboration(ts("2026-03-08T00:00:00Z"))
    assert corroboration.is_corroborated is True
    assert corroboration.agent_count == 1  # but the trigger would NOT fire


def test_findings_from_the_future_are_invisible(memory):
    board = StateBoard(memory)
    board.publish(make_finding("transaction", "2026-04-01T00:00:00Z"))
    assert board.findings(ts("2026-03-08T00:00:00Z")) == []


# ===========================================================================
# Persisted state -- the "don't recompute from scratch" requirement
# ===========================================================================

def test_state_persists_across_checkpoints(memory):
    board = StateBoard(memory)
    memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.RELATIONSHIP_MANAGER_ESCALATION,
            action_subtype="premium_retention_offer_and_fee_waiver",
            hitl_status=HitlStatus.ESCALATED,
            notes="Driven by EVT_000457 and EVT_000461.",
            guardrail_checked=True,
        )
    )
    # A month later, with nothing new, the belief still stands.
    later = board.current_state(ts("2026-04-10T00:00:00Z"))
    assert later.inferred_state is InferredState.CHURN_RISK
    assert later.confidence_band is ConfidenceBand.HIGH
    assert later.age_days == pytest.approx(33, abs=1)


def test_cold_start_is_no_significant_event(memory):
    state = StateBoard(memory).current_state(ts("2026-02-01T00:00:00Z"))
    assert state.inferred_state is InferredState.NO_SIGNIFICANT_EVENT
    assert state.is_default is True


def test_a_decision_is_invisible_before_it_was_made(memory):
    board = StateBoard(memory)
    memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.NO_ACTION,
            notes="EVT_000461",
        )
    )
    earlier = board.current_state(ts("2026-02-15T00:00:00Z"))
    assert earlier.is_default is True, "a future decision leaked into a past checkpoint"
