"""
Tests for the Synthesis Agent.

The headline test is test_all_graded_checkpoints_match_ground_truth, which runs
the real pipeline over all three scenarios and compares inferred_state and
confidence_band against every checkpoint in every ground_truth.json.

Be clear about what that test does and does not prove. The thresholds in
synthesis.py were CALIBRATED against these eight checkpoints, so passing is not
independent evidence of generalisation -- it is confirmation that the calibration
was applied correctly. The value is as a REGRESSION guard: any later change that
breaks one of the eight shows up immediately.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["C360_OFFLINE"] = "1"

from c360.agents import PerceptionContext, default_swarm, run_swarm_on_event, run_swarm_on_tick  # noqa: E402
from c360.findings import Finding, SignalStrength  # noqa: E402
from c360.llm import NullLLM  # noqa: E402
from c360.memory import EpisodicMemory  # noqa: E402
from c360.output import Checkpoint  # noqa: E402
from c360.pii import Redactor  # noqa: E402
from c360.replay import EventTick, ReplayEngine  # noqa: E402
from c360.schema import Action, ConfidenceBand, HitlStatus, InferredState  # noqa: E402
from c360.state_board import StateBoard  # noqa: E402
from c360.synthesis import (  # noqa: E402
    SynthesisAgent,
    confidence_for,
    score_states,
)

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = [DATA / f"scenario_0{n}" for n in (1, 2, 3)]
needs_data = pytest.mark.skipif(
    not all(s.exists() for s in ALL_SCENARIOS), reason="scenario data not present"
)
every_scenario = pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def finding(agent, signal, strength, systems, when="2026-03-07T00:00:00Z", **metrics):
    return Finding(
        agent=agent,
        as_of=ts(when),
        signal=signal,
        strength=strength,
        event_ids=(f"EVT_{abs(hash(signal)) % 900000 + 1000:06d}",),
        source_systems=systems,
        detail=f"{signal} detail",
        metrics=metrics,
    )


@pytest.fixture
def board():
    memory = EpisodicMemory(customer_id="CUST_TEST")
    yield StateBoard(memory)
    memory.close()


# ===========================================================================
# Scoring
# ===========================================================================

def test_a_signal_counts_once_however_many_findings_carry_it():
    """
    Volume must not beat meaning. Five pharmacy purchases are one kind of
    evidence seen five times -- exactly the failure mode a red herring exploits.
    """
    one = score_states([finding("transaction", "healthcare_spend", SignalStrength.MODERATE, ("card_payments",))])
    five = score_states([
        finding("transaction", "healthcare_spend", SignalStrength.MODERATE, ("card_payments",))
        for _ in range(5)
    ])
    assert one == five


def test_stronger_evidence_of_the_same_signal_wins():
    findings = [
        finding("transaction", "healthcare_spend", SignalStrength.WEAK, ("card_payments",)),
        finding("transaction", "healthcare_spend", SignalStrength.MODERATE, ("card_payments",)),
    ]
    assert score_states(findings)[InferredState.MEDICAL_HARDSHIP] == pytest.approx(6.0)


def test_metric_carried_signals_route_by_their_metric():
    """
    search_intent means different things depending on what was searched for.
    The signal name alone cannot say which state it supports.
    """
    child = score_states([
        finding("usage", "search_intent", SignalStrength.MODERATE, ("web_app_events",), intent="child_planning")
    ])
    leaving = score_states([
        finding("usage", "search_intent", SignalStrength.MODERATE, ("web_app_events",), intent="leaving")
    ])
    assert child.get(InferredState.NEW_CHILD_LIFE_EVENT, 0) > 0
    assert leaving.get(InferredState.CHURN_RISK, 0) > 0
    assert child.get(InferredState.CHURN_RISK, 0) == 0


# ===========================================================================
# Confidence bands
# ===========================================================================

@pytest.mark.parametrize(
    "strong_signals, sources, expected",
    [
        (0, 1, ConfidenceBand.LOW),
        (0, 3, ConfidenceBand.LOW),     # breadth alone is not enough
        (1, 3, ConfidenceBand.LOW),     # one strong signal is not enough
        (2, 2, ConfidenceBand.MEDIUM),  # two strong, two systems
        (2, 3, ConfidenceBand.HIGH),    # two strong, three systems
        (3, 4, ConfidenceBand.HIGH),
    ],
)
def test_confidence_needs_both_depth_and_breadth(board, strong_signals, sources, expected):
    """
    The rule in one line: two independent STRONG signals across three independent
    SOURCE SYSTEMS is high. Requiring strength as well as breadth is what keeps
    15 February low in all three scenarios while the evidence is circumstantial.
    """
    systems = tuple(f"system_{i}" for i in range(sources))
    findings = [
        finding("transaction", f"sig_{i}", SignalStrength.STRONG, systems) for i in range(strong_signals)
    ] or [finding("transaction", "sig_weak", SignalStrength.MODERATE, systems)]

    for f in findings:
        board.memory.record_finding(f)
    corroboration = board.corroboration(ts("2026-03-08T00:00:00Z"))
    assert confidence_for(findings, corroboration) is expected


def test_unrelated_evidence_cannot_inflate_confidence(board):
    """
    scenario_01's $12,000 tuition transfer sits on its own source system
    (ach_wire) in the same window as the medical story. Counting it would make
    a third "independent system" corroborating MEDICAL HARDSHIP and push the
    12 March checkpoint to high when ground truth expects medium.
    """
    board.publish(finding("transaction", "major_medical_expense", SignalStrength.STRONG, ("card_payments",)))
    board.publish(finding("transaction", "income_replacement", SignalStrength.STRONG, ("core_banking_ledger",)))
    board.publish(finding("transaction", "large_outbound_transfer", SignalStrength.MODERATE, ("ach_wire",)))

    result = SynthesisAgent().synthesise(ts("2026-03-08T00:00:00Z"), board)
    assert result.inferred_state is InferredState.MEDICAL_HARDSHIP
    assert result.confidence_band is ConfidenceBand.MEDIUM
    assert "ach_wire" not in result.corroboration.source_systems


# ===========================================================================
# State and persistence
# ===========================================================================

def test_no_findings_carries_the_previous_belief_forward(board):
    board.memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.NO_ACTION,
            notes="EVT_000461",
        )
    )
    result = SynthesisAgent().synthesise(ts("2026-03-20T00:00:00Z"), board)
    assert result.inferred_state is InferredState.CHURN_RISK
    assert result.confidence_band is ConfidenceBand.HIGH
    assert result.carried_forward is True


def test_belief_decays_after_prolonged_silence(board):
    board.memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-01-01T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.NO_ACTION,
            notes="EVT_000461",
        )
    )
    result = SynthesisAgent().synthesise(ts("2026-06-01T00:00:00Z"), board)
    assert result.confidence_band is ConfidenceBand.MEDIUM


def test_confidence_does_not_slip_back_for_the_same_state(board):
    """
    Ground truth holds scenario_03 at high from 8 March through 10 April. A
    system whose band wobbled when one quiet fortnight thinned the window would
    fail the later checkpoint despite having been right earlier.
    """
    board.memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.NO_ACTION,
            notes="EVT_000461",
        )
    )
    board.publish(finding("usage", "engagement_drop", SignalStrength.MODERATE, ("web_app_events",),
                          when="2026-03-18T00:00:00Z"))
    result = SynthesisAgent().synthesise(ts("2026-03-20T00:00:00Z"), board)
    assert result.inferred_state is InferredState.CHURN_RISK
    assert result.confidence_band is ConfidenceBand.HIGH


def test_notes_always_cite_event_ids(board):
    """
    The graded explainability requirement, satisfied without an LLM. The rationale
    is built from finding metadata, so citations survive an offline run.
    """
    board.publish(finding("usage", "cancellation_feature_used", SignalStrength.STRONG, ("web_app_events",)))
    board.publish(finding("transaction", "self_transfer_external", SignalStrength.STRONG, ("instant_payments",)))
    result = SynthesisAgent().synthesise(ts("2026-03-08T00:00:00Z"), board)
    assert "EVT_" in result.rationale


# ===========================================================================
# End to end against ground truth
# ===========================================================================

def run_scenario(scenario: Path):
    """The real pipeline, offline, returning {as_of_iso: Synthesis}."""
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
    synth = SynthesisAgent()
    results: dict[str, object] = {}

    for tick in engine.stream():
        if isinstance(tick, EventTick):
            memory.record_event(tick.event)
            board.publish_all(run_swarm_on_event(agents, tick.event, tick.as_of, ctx))
        else:
            board.publish_all(run_swarm_on_tick(agents, tick.as_of, ctx))
            result = synth.synthesise(tick.as_of, board)
            memory.record_decision(
                Checkpoint(
                    as_of_time=tick.as_of,
                    inferred_state=result.inferred_state,
                    confidence_band=result.confidence_band,
                    action=Action.NO_ACTION,
                    hitl_status=HitlStatus.AUTO_APPROVED,
                    notes=result.rationale[:600],
                )
            )
            results[tick.as_of.strftime("%Y-%m-%dT%H:%M:%SZ")] = result
    memory.close()
    return results


@needs_data
@every_scenario
def test_all_graded_checkpoints_match_ground_truth(scenario):
    """
    inferred_state AND confidence_band correct at every graded checkpoint.

    Calibration check, not generalisation evidence -- the thresholds were fitted
    to these eight points. Its job is to catch regressions.
    """
    gt = json.loads((scenario / "ground_truth.json").read_text())
    results = run_scenario(scenario)

    for expected in gt["checkpoints"]:
        key = expected["as_of_time"]
        assert key in results, f"no checkpoint emitted at {key}"
        got = results[key]
        assert got.inferred_state.value == expected["expected_inferred_state"], (
            f"{scenario.name} {key}: state {got.inferred_state.value} "
            f"!= {expected['expected_inferred_state']} (affinity={got.affinity})"
        )
        assert got.confidence_band.value == expected["expected_confidence_band"], (
            f"{scenario.name} {key}: confidence {got.confidence_band.value} "
            f"!= {expected['expected_confidence_band']} (strong={got.strong_signals}, "
            f"sources={got.corroboration.source_systems})"
        )


@needs_data
@every_scenario
def test_state_settles_and_stays_settled(scenario):
    """
    Stability check. A system whose inferred_state oscillated day to day might
    still hit the graded checkpoints by luck while being obviously unusable.

    Early changes are ALLOWED and expected: scenario_02 honestly moves through
    churn_risk and job_loss_or_income_disruption in the first fortnight -- at LOW
    confidence, on two weak signals -- before the baby purchases make
    new_child_life_event the better explanation. Pretending to be stable while
    genuinely uncertain would be worse. What must not happen is churn AFTER the
    narrative is established.
    """
    results = run_scenario(scenario)
    ordered = [results[k] for k in sorted(results)]
    states = [r.inferred_state for r in ordered]

    early = sum(1 for a, b in zip(states[:21], states[1:21]) if a is not b)
    late = sum(1 for a, b in zip(states[21:], states[22:]) if a is not b)
    assert early <= 5, f"{scenario.name}: {early} changes in the first three weeks"
    assert late == 0, f"{scenario.name}: state still changing after week three"


@needs_data
def test_a_windfall_cannot_displace_an_established_belief():
    """
    Hysteresis, checked on the real red herring. scenario_03's $5,200 tax refund
    lands on 28 February and scores towards wealth_growth_or_windfall. By then the
    system already holds churn_risk, and it must stay held.
    """
    results = run_scenario(DATA / "scenario_03")
    for key in ("2026-02-28T00:00:00Z", "2026-03-01T00:00:00Z", "2026-03-02T00:00:00Z"):
        assert results[key].inferred_state is InferredState.CHURN_RISK, (
            f"tax refund displaced the belief at {key}: {results[key].affinity}"
        )


@needs_data
def test_scenario_03_reaches_high_confidence_before_the_deadline():
    """
    The lead-time claim. Ground truth wants the escalation with 3 days of lead
    time before 8 March, so high confidence must arrive on or before 8 March --
    and NOT in February, where the expectation is still low.
    """
    results = run_scenario(DATA / "scenario_03")
    first_high = next(
        (k for k, r in sorted(results.items()) if r.confidence_band is ConfidenceBand.HIGH), None
    )
    assert first_high is not None
    assert first_high <= "2026-03-08", f"high confidence only reached at {first_high}"
    assert first_high > "2026-02-28", f"high confidence claimed too early at {first_high}"
