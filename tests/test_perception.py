"""
Tests for the perception swarm.

Two kinds of test here:

  - UNIT tests for each detector, using tiny hand-built fixtures.
  - OUTCOME tests that run the real swarm over the real scenarios and check it
    actually notices the events the answer key calls signals, and does NOT build
    a corroborated case out of the planted red herrings.

The second kind is the one that matters. A detector that passes its unit test but
never fires on the real data is worth nothing.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force the deterministic path. A machine with real keys in its environment must
# still run the fallback during pytest rather than making live API calls.
os.environ["C360_OFFLINE"] = "1"

from c360.agents import (  # noqa: E402
    LifeSignalAgent,
    PerceptionContext,
    SupportAgent,
    TransactionAgent,
    UsageAgent,
    default_swarm,
    run_swarm_on_event,
    run_swarm_on_tick,
)
from c360.agents.base import SIGNALS  # noqa: E402
from c360.findings import SignalStrength  # noqa: E402
from c360.llm import NullLLM  # noqa: E402
from c360.memory import EpisodicMemory  # noqa: E402
from c360.pii import Redactor  # noqa: E402
from c360.replay import ClockTick, EventTick, ReplayEngine  # noqa: E402
from c360.schema import Event  # noqa: E402
from c360.state_board import StateBoard  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = [DATA / f"scenario_0{n}" for n in (1, 2, 3)]
needs_data = pytest.mark.skipif(
    not all(s.exists() for s in ALL_SCENARIOS), reason="scenario data not present"
)
every_scenario = pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ev(event_id, when, source_system, event_type, payload, account_id="ACC_CHK_001"):
    return Event(
        event_id=event_id,
        event_time=ts(when),
        ingestion_time=ts(when),
        customer_id="CUST_TEST",
        account_id=account_id,
        source_system=source_system,
        event_type=event_type,
        schema_version="1.0",
        payload=payload,
    )


@pytest.fixture
def ctx():
    memory = EpisodicMemory(customer_id="CUST_TEST")
    yield PerceptionContext(
        memory=memory,
        redactor=Redactor(customer_id="CUST_TEST", name="David Chen"),
        entities={"customer_id": "CUST_TEST", "accounts": [{"account_id": "ACC_SAV_003", "type": "savings"}]},
        llm=NullLLM(),
    )
    memory.close()


def signals(findings) -> set[str]:
    return {f.signal for f in findings}


# ===========================================================================
# Swarm structure
# ===========================================================================

def test_agents_cover_disjoint_source_systems():
    """Overlap would let one event reach two agents and be counted as two witnesses."""
    seen: set[str] = set()
    for agent in default_swarm():
        overlap = seen & set(agent.source_systems)
        assert not overlap, f"{agent.name} overlaps on {overlap}"
        seen |= set(agent.source_systems)


def test_every_dataset_source_system_has_an_owner():
    from c360.schema import KNOWN_SOURCE_SYSTEMS

    owned = {s for agent in default_swarm() for s in agent.source_systems}
    assert KNOWN_SOURCE_SYSTEMS <= owned, f"unowned: {KNOWN_SOURCE_SYSTEMS - owned}"


@needs_data
def test_swarm_result_is_independent_of_agent_order(ctx):
    """The defining property of a swarm. If order mattered, it is a pipeline."""
    engine = ReplayEngine(DATA / "scenario_03", speed=0).load()
    ctx.memory.record_events(engine.history)
    as_of = ts("2026-03-08T00:00:00Z")
    ctx.memory.record_events([e for e in engine.live if e.release_time <= as_of])

    forward = run_swarm_on_tick(default_swarm(), as_of, ctx)
    backward = run_swarm_on_tick(list(reversed(default_swarm())), as_of, ctx)
    assert signals(forward) == signals(backward)


def test_all_emitted_signals_are_in_the_vocabulary(ctx):
    """A typo'd signal would silently never match anything in synthesis."""
    agent = TransactionAgent()
    found = agent.on_event(
        ev("E1", "2026-02-10T10:00:00Z", "card_payments", "purchase",
           {"merchant_name": "City General Hospital ER", "mcc_category": "healthcare", "amount": 500}),
        ts("2026-02-10T12:00:00Z"),
        ctx,
    )
    assert found and all(f.signal in SIGNALS for f in found)


# ===========================================================================
# Transaction Agent
# ===========================================================================

def test_major_medical_expense_beats_routine_healthcare(ctx):
    agent = TransactionAgent()
    small = agent.on_event(
        ev("S", "2026-02-10T10:00:00Z", "card_payments", "purchase",
           {"mcc_category": "healthcare", "amount": 500, "merchant_name": "ER"}),
        ts("2026-02-10T12:00:00Z"), ctx)
    large = agent.on_event(
        ev("L", "2026-03-10T10:00:00Z", "card_payments", "purchase",
           {"mcc_category": "healthcare", "amount": 8500, "merchant_name": "Hospital Billing"}),
        ts("2026-03-10T12:00:00Z"), ctx)
    assert signals(small) == {"healthcare_spend"}
    assert signals(large) == {"major_medical_expense"}
    assert large[0].strength is SignalStrength.STRONG


def test_baby_retailer_detected_even_when_mcc_says_clothing(ctx):
    """
    scenario_02's Mothercare purchase is filed under mcc_category 'clothing'.
    Category alone would miss a graded signal event.
    """
    found = TransactionAgent().on_event(
        ev("M", "2026-03-02T12:00:00Z", "card_payments", "purchase",
           {"merchant_name": "Mothercare", "mcc_category": "clothing", "amount": 85}),
        ts("2026-03-02T12:00:00Z"), ctx)
    assert "baby_retail_spend" in signals(found)


def test_self_transfer_is_distinguished_from_a_third_party_transfer(ctx):
    """The churn tell. Both are large outbound transfers; only one means leaving."""
    agent = TransactionAgent()
    mine = agent.on_event(
        ev("SELF", "2026-03-06T11:20:00Z", "instant_payments", "outbound_transfer",
           {"amount": 22500, "counterparty_name": "David Chen - Chase Bank", "direction": "outbound"}),
        ts("2026-03-06T12:00:00Z"), ctx)
    theirs = agent.on_event(
        ev("TUITION", "2026-02-05T11:20:00Z", "ach_wire", "outbound_transfer",
           {"amount": 12000, "counterparty_name": "State University", "direction": "outbound"}),
        ts("2026-02-05T12:00:00Z"), ctx)
    assert "self_transfer_external" in signals(mine)
    assert "self_transfer_external" not in signals(theirs)
    assert "large_outbound_transfer" in signals(theirs)


def test_salary_swept_out_needs_a_recent_salary(ctx):
    agent = TransactionAgent()
    ctx.memory.record_event(
        ev("SAL", "2026-03-20T08:00:00Z", "core_banking_ledger", "deposit",
           {"amount": 5000, "transaction_type": "salary_credit", "balance_after": 62500}))
    found = agent.on_event(
        ev("OUT", "2026-03-20T09:15:00Z", "instant_payments", "outbound_transfer",
           {"amount": 4900, "counterparty_name": "David Chen - Ext Bank", "direction": "outbound"}),
        ts("2026-03-20T09:15:00Z"), ctx)
    assert "salary_swept_out" in signals(found)
    swept = next(f for f in found if f.signal == "salary_swept_out")
    assert set(swept.source_systems) == {"core_banking_ledger", "instant_payments"}


def test_income_disruption_uses_the_customers_own_median(ctx):
    """scenario_02's 4500 -> 2700 maternity dip."""
    for i, amount in enumerate([4500, 4500, 4500, 4500, 2700]):
        ctx.memory.record_event(
            ev(f"D{i}", f"2026-01-{(i * 5) + 1:02d}T08:00:00Z", "core_banking_ledger", "deposit",
               {"amount": amount, "transaction_type": "salary_credit", "balance_after": 10000}))
    found = TransactionAgent().on_tick(ts("2026-02-01T00:00:00Z"), ctx)
    assert "income_disruption" in signals(found)
    disruption = next(f for f in found if f.signal == "income_disruption")
    assert disruption.metrics["drop_ratio"] == pytest.approx(0.4, abs=0.01)


def test_income_replacement_is_flagged_separately_from_a_dip(ctx):
    found = TransactionAgent().on_event(
        ev("BEN", "2026-02-20T08:00:00Z", "core_banking_ledger", "deposit",
           {"amount": 1400, "transaction_type": "benefits_credit", "balance_after": 44100}),
        ts("2026-02-20T08:00:00Z"), ctx)
    assert "income_replacement" in signals(found)
    assert found[0].strength is SignalStrength.STRONG


def test_standing_instruction_stopped_needs_two_missed_cycles(ctx):
    """
    The clearest case for the time-based tick -- there is no event for 'the rent
    did not go out this month' -- but it is deliberately CONSERVATIVE.

    All three practice scenarios are missing their February standing instructions
    entirely (a data-generation artifact: every ledger runs Nov, Dec, Jan, nothing,
    Mar). A one-missed-cycle threshold fired in all three, including the two
    customers who are not churning. Two missed cycles is the price of silencing
    that; see the long note in transaction.py.
    """
    for month in (11, 12, 1, 2, 3):
        year = 2025 if month >= 11 else 2026
        ctx.memory.record_event(
            ev(f"SI{year}{month}", f"{year}-{month:02d}-01T09:00:00Z",
               "core_banking_ledger", "standing_instruction",
               {"amount": 2500, "transaction_type": "rent_payment", "balance_after": 50000}))

    # One missed cycle (1 April) -- not yet enough.
    assert "standing_instruction_stopped" not in signals(
        TransactionAgent().on_tick(ts("2026-04-12T00:00:00Z"), ctx))

    # Two missed cycles (1 April and 1 May) -- now it fires.
    found = TransactionAgent().on_tick(ts("2026-05-10T00:00:00Z"), ctx)
    assert "standing_instruction_stopped" in signals(found)


@needs_data
@every_scenario
def test_standing_instruction_detector_is_silent_on_the_february_artifact(scenario):
    """
    Pins the honest finding above: the missing February must NOT be reported as a
    stopped instruction in ANY scenario -- least of all scenario_01 and
    scenario_02, whose customers resume paying normally on 1 April.
    """
    engine, memory, board = _run_swarm(scenario, ts("2026-03-01T00:00:00Z"))
    stopped = board.findings(ts("2026-03-01T00:00:00Z"), window_days=400,
                             signals=["standing_instruction_stopped"])
    assert stopped == [], f"{scenario.name}: fired on the February data gap"
    memory.close()


def test_standing_instruction_not_flagged_while_still_on_schedule(ctx):
    for month in (1, 2, 3):
        ctx.memory.record_event(
            ev(f"SI{month}", f"2026-{month:02d}-01T09:00:00Z",
               "core_banking_ledger", "standing_instruction",
               {"amount": 2500, "transaction_type": "rent_payment", "balance_after": 50000}))
    found = TransactionAgent().on_tick(ts("2026-03-10T00:00:00Z"), ctx)
    assert "standing_instruction_stopped" not in signals(found)


def test_windfall_deposit_is_weak_and_single_sourced(ctx):
    """
    scenario_03's tax refund. Recorded honestly, but weak and from one system --
    so it can never corroborate itself into an offer.
    """
    found = TransactionAgent().on_event(
        ev("TAX", "2026-02-28T09:00:00Z", "core_banking_ledger", "deposit",
           {"amount": 5200, "transaction_type": "tax_refund", "balance_after": 62900}),
        ts("2026-02-28T09:00:00Z"), ctx)
    windfall = next(f for f in found if f.signal == "large_inbound_deposit")
    assert windfall.strength is SignalStrength.WEAK
    assert windfall.independent_source_count == 1


# ===========================================================================
# Usage Agent
# ===========================================================================

def test_cancellation_page_fires_immediately(ctx):
    found = UsageAgent().on_event(
        ev("C", "2026-03-04T19:00:00Z", "web_app_events", "feature_used",
           {"feature_or_page": "manage_standing_instructions_cancel", "device_type": "desktop"}),
        ts("2026-03-04T19:00:00Z"), ctx)
    assert signals(found) == {"cancellation_feature_used"}
    assert found[0].strength is SignalStrength.STRONG


def test_ordinary_page_views_produce_nothing(ctx):
    found = UsageAgent().on_event(
        ev("H", "2026-03-04T19:00:00Z", "web_app_events", "feature_used",
           {"feature_or_page": "home_dashboard"}),
        ts("2026-03-04T19:00:00Z"), ctx)
    assert found == []


@pytest.mark.parametrize(
    "query, expected",
    [
        # scenario_01 EVT_000453. Matches BOTH "hardship" (financial_hardship) and
        # "medical hardship" (medical); the longer, more specific phrase must win.
        # Picking by dict order sent this to financial_distress_general and cost
        # the medical_hardship inference at the 26 March checkpoint.
        ("medical hardship plan", "medical"),
        ("i need a payment plan", "financial_hardship"),
        # scenario_02 EVT_000344.
        ("child education savings plan", "child_planning"),
        ("transaction history", None),
    ],
)
def test_search_intent_keyword_fallback(ctx, query, expected):
    """With NullLLM configured, the deterministic path must still work."""
    found = UsageAgent().on_event(
        ev("S", "2026-03-20T10:00:00Z", "web_app_events", "search_query",
           {"feature_or_page": "search", "search_text": query}),
        ts("2026-03-20T10:00:00Z"), ctx)
    if expected is None:
        assert found == []
    else:
        assert found[0].signal == "search_intent"
        assert found[0].metrics["intent"] == expected
        assert found[0].metrics["classified_by"] == "keywords"


def test_search_text_is_scrubbed_before_classification(ctx):
    found = UsageAgent().on_event(
        ev("S", "2026-03-20T10:00:00Z", "web_app_events", "search_query",
           {"search_text": "David Chen hardship payment plan"}),
        ts("2026-03-20T10:00:00Z"), ctx)
    assert "David" not in found[0].detail
    assert "<CUSTOMER_NAME>" in found[0].metrics["query"]


# ===========================================================================
# Support Agent
# ===========================================================================

def test_denied_complaint_is_structural_not_textual(ctx):
    """
    scenario_03 EVT_000412. The text is the BANK refusing; the signal comes from
    resolution_status, so it fires with or without a language model.
    """
    found = SupportAgent().on_event(
        ev("R", "2026-02-06T10:15:00Z", "support_logs", "ticket_resolved",
           {"channel": "chat", "category": "dispute", "resolution_status": "human_rejected",
            "raw_text": "As per our schedule of fees, the fee is valid and cannot be waived."},
           account_id=None),
        ts("2026-02-06T10:15:00Z"), ctx)
    assert "complaint_denied" in signals(found)
    denied = next(f for f in found if f.signal == "complaint_denied")
    assert denied.strength is SignalStrength.STRONG


def test_hardship_category_fires_regardless_of_wording(ctx):
    found = SupportAgent().on_event(
        ev("H", "2026-03-25T10:00:00Z", "support_logs", "ticket_created",
           {"category": "payment_arrangements", "resolution_status": "open",
            "raw_text": "I've been in the hospital and my income dropped. Can I set up a payment plan?"},
           account_id=None),
        ts("2026-03-25T10:00:00Z"), ctx)
    assert "hardship_contact" in signals(found)


def test_one_ticket_counts_once_per_signal(ctx):
    """
    The category check and the text check can both land on hardship_contact.
    Counting it twice would inflate the confidence score off a single ticket.
    """
    found = SupportAgent().on_event(
        ev("H", "2026-03-25T10:00:00Z", "support_logs", "ticket_created",
           {"category": "payment_arrangements", "resolution_status": "open",
            "raw_text": "hardship - cannot pay, I was in hospital"},
           account_id=None),
        ts("2026-03-25T10:00:00Z"), ctx)
    assert len([f for f in found if f.signal == "hardship_contact"]) == 1


def test_reassurance_ticket_is_not_treated_as_distress(ctx):
    """scenario_02's baby-monitor confirmation is calm, and child-related."""
    found = SupportAgent().on_event(
        ev("B", "2026-02-26T10:00:00Z", "support_logs", "ticket_resolved",
           {"category": "transaction_query", "resolution_status": "resolved",
            "raw_text": "Just confirming it's me, I bought a new video baby monitor online."},
           account_id=None),
        ts("2026-02-26T10:00:00Z"), ctx)
    assert "complaint_denied" not in signals(found)
    assert "reassurance_contact" in signals(found)


# ===========================================================================
# Life-Signal Agent
# ===========================================================================

def test_dependents_increase_is_strong(ctx):
    found = LifeSignalAgent().on_event(
        ev("K", "2026-03-25T10:00:00Z", "loan_kyc", "dependents_change",
           {"event_subtype": "dependents_change", "old_value": 1, "new_value": 2}),
        ts("2026-03-25T10:00:00Z"), ctx)
    assert signals(found) == {"dependents_increase"}
    assert found[0].strength is SignalStrength.STRONG


def test_dependents_decrease_is_not_a_new_child_signal(ctx):
    found = LifeSignalAgent().on_event(
        ev("K", "2026-03-25T10:00:00Z", "loan_kyc", "dependents_change",
           {"event_subtype": "dependents_change", "old_value": 2, "new_value": 1}),
        ts("2026-03-25T10:00:00Z"), ctx)
    assert found == []


def test_social_signal_without_consent_is_ignored(ctx):
    found = LifeSignalAgent().on_event(
        ev("S", "2026-03-01T10:00:00Z", "social_signal_consented", "life_event_mention",
           {"platform": "x", "raw_text": "our baby arrived!", "consent_flag": False}),
        ts("2026-03-01T10:00:00Z"), ctx)
    assert found == []


def test_social_signal_with_consent_is_read_but_stays_weak(ctx):
    found = LifeSignalAgent().on_event(
        ev("S", "2026-03-01T10:00:00Z", "social_signal_consented", "life_event_mention",
           {"platform": "x", "raw_text": "our baby arrived!", "consent_flag": True}),
        ts("2026-03-01T10:00:00Z"), ctx)
    assert signals(found) == {"social_life_event"}
    assert found[0].strength is not SignalStrength.STRONG  # self-reported, unverified


# ===========================================================================
# OUTCOME TESTS -- the swarm on the real scenarios
# ===========================================================================

def _run_swarm(scenario: Path, until: datetime):
    """Replay up to `until`, running the swarm, and return the state board."""
    engine = ReplayEngine(scenario, speed=0).load()
    memory = EpisodicMemory(customer_id=engine.entities["customer_id"])
    memory.record_events(engine.history)
    ctx = PerceptionContext(
        memory=memory,
        redactor=Redactor.from_entities(engine.entities),
        entities=engine.entities,
        llm=NullLLM(),
    )
    board = StateBoard(memory)
    agents = default_swarm()

    for tick in engine.stream():
        if tick.as_of > until:
            break
        if isinstance(tick, EventTick):
            memory.record_event(tick.event)
            board.publish_all(run_swarm_on_event(agents, tick.event, tick.as_of, ctx))
        else:
            board.publish_all(run_swarm_on_tick(agents, tick.as_of, ctx))
    return engine, memory, board


@needs_data
@every_scenario
def test_swarm_cites_the_graded_signal_events(scenario):
    """
    The swarm must actually notice the events the answer key calls signals.

    Not all of them -- some are corroborating detail rather than something a
    detector should fire on -- but a substantial majority, and specifically the
    decisive ones near the final graded checkpoint.
    """
    gt = json.loads((scenario / "ground_truth.json").read_text())
    last_checkpoint = max(ts(c["as_of_time"]) for c in gt["checkpoints"])
    engine, memory, board = _run_swarm(scenario, last_checkpoint)

    cited_ids = {eid for f in board.findings(last_checkpoint, window_days=400) for eid in f.event_ids}
    signal_ids = set(gt["signal_events"])
    hit = signal_ids & cited_ids
    assert len(hit) >= len(signal_ids) * 0.5, (
        f"{scenario.name}: only cited {sorted(hit)} of {sorted(signal_ids)}"
    )
    memory.close()


@needs_data
@every_scenario
def test_red_herrings_never_stand_alone_as_corroborated_evidence(scenario):
    """
    THE RED-HERRING TEST.

    For each planted red herring, look at the findings that cite ONLY that event.
    They must never reach 2 independent source systems on their own -- which is
    what stops the guardrail from ever letting one trigger an action.
    """
    gt = json.loads((scenario / "ground_truth.json").read_text())
    last_checkpoint = max(ts(c["as_of_time"]) for c in gt["checkpoints"])
    engine, memory, board = _run_swarm(scenario, last_checkpoint)

    for herring in gt["red_herring_events"]:
        solo = [
            f
            for f in board.findings(last_checkpoint, window_days=400)
            if set(f.event_ids) == {herring}
        ]
        systems = {s for f in solo for s in f.source_systems}
        assert len(systems) < 2, (
            f"{scenario.name}: red herring {herring} alone reached {systems}"
        )
        for finding in solo:
            assert finding.strength is not SignalStrength.STRONG, (
                f"{scenario.name}: red herring {herring} produced a STRONG finding"
            )
    memory.close()


@needs_data
@every_scenario
def test_swarm_produces_multi_source_corroboration_by_the_final_checkpoint(scenario):
    """
    The mirror of the red-herring test: the GENUINE narrative must reach >= 2
    independent source systems, or no action could ever fire and every scenario
    would score no_action.
    """
    gt = json.loads((scenario / "ground_truth.json").read_text())
    last_checkpoint = max(ts(c["as_of_time"]) for c in gt["checkpoints"])
    engine, memory, board = _run_swarm(scenario, last_checkpoint)

    corroboration = board.corroboration(last_checkpoint, window_days=30)
    assert corroboration.is_corroborated, corroboration.explain()
    assert corroboration.agent_count >= 2, "synthesis would never trigger"
    memory.close()


@needs_data
def test_scenario_03_sees_the_cancellation_before_the_money_moves():
    """
    The lead-time claim, checked directly. scenario_03's ground truth wants an
    escalation with 3 days of lead time; the cancellation click on 4 March is two
    days before the transfer on 6 March, and it is what makes that possible.
    """
    engine, memory, board = _run_swarm(DATA / "scenario_03", ts("2026-03-05T00:00:00Z"))
    found = board.findings(ts("2026-03-05T00:00:00Z"), window_days=30)
    assert "cancellation_feature_used" in {f.signal for f in found}
    assert "EVT_000457" in {e for f in found for e in f.event_ids}
    # And the transfer has NOT happened yet.
    assert "EVT_000461" not in {e for f in found for e in f.event_ids}
    memory.close()
