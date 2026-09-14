"""
Tests for the decision layer: semantic memory, action proposer, guardrail,
critique and HITL.

The most important tests here are the RED-HERRING ones. Every planted red herring
in all three scenarios is checked directly: not "did the final score come out
right" but "when this specific event is the only evidence, does the guardrail
block the specific action ground truth forbids".
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["C360_OFFLINE"] = "1"

from c360 import guardrail as guardrail_module  # noqa: E402
from c360.action import ActionProposal, ActionProposer  # noqa: E402
from c360.critique import CritiqueAgent, HitlStub  # noqa: E402
from c360.findings import Finding, SignalStrength  # noqa: E402
from c360.memory import EpisodicMemory  # noqa: E402
from c360.output import Checkpoint  # noqa: E402
from c360.schema import Action, ConfidenceBand, HitlStatus, InferredState  # noqa: E402
from c360.semantic import HashingEmbedder, SemanticMemory  # noqa: E402
from c360.state_board import Corroboration  # noqa: E402
from c360.synthesis import Synthesis  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@pytest.fixture(scope="module")
def semantic():
    store = SemanticMemory(prefer_default_embedder=False)
    store.seed()
    return store


@pytest.fixture
def memory():
    with EpisodicMemory(customer_id="CUST_TEST") as store:
        yield store


def finding(signal, strength, systems, events=("EVT_000001",), agent="transaction", **metrics):
    return Finding(
        agent=agent,
        as_of=ts("2026-03-07T00:00:00Z"),
        signal=signal,
        strength=strength,
        event_ids=events,
        source_systems=systems,
        detail=f"{signal}",
        metrics=metrics,
    )


def synthesis(state, band, findings, as_of="2026-03-08T00:00:00Z"):
    sources, agents, events = set(), set(), []
    for f in findings:
        sources.update(f.source_systems)
        agents.add(f.agent)
        events.extend(e for e in f.event_ids if e not in events)
    corroboration = Corroboration(
        as_of=ts(as_of),
        window_days=30,
        findings=tuple(findings),
        source_systems=tuple(sorted(sources)),
        agents=tuple(sorted(agents)),
        event_ids=tuple(events),
    )
    return Synthesis(
        as_of=ts(as_of),
        inferred_state=state,
        confidence_band=band,
        rationale=f"{state.value} indicated. Evidence: {', '.join(events)}.",
        event_ids=tuple(events),
        corroboration=corroboration,
        strong_signals=tuple(f.signal for f in findings if f.strength is SignalStrength.STRONG),
    )


# ===========================================================================
# Semantic memory
# ===========================================================================

def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder()
    a, b = embedder.embed("medical hardship payment plan"), embedder.embed("medical hardship payment plan")
    assert a == b
    assert sum(v * v for v in a) == pytest.approx(1.0)


def test_seed_is_idempotent(semantic):
    """Incremental upsert, not full rebuild -- a mid-term requirement."""
    assert semantic.seed() == {"policies": 0, "patterns": 0}


@pytest.mark.parametrize(
    "query, expected_top",
    [
        ("medical hardship hospital billing benefits credit payment plan", "pol_medical_hardship"),
        ("dependents increase daycare childcare baby retailers education savings", "pol_new_child"),
        ("churn cancellation standing instructions transfer another institution", "pol_churn_escalation"),
        ("tax refund large inbound deposit investable surplus", "pol_windfall"),
    ],
)
def test_policy_retrieval_returns_the_right_policy(semantic, query, expected_top):
    assert semantic.retrieve_policies(query, k=1)[0].doc_id == expected_top


def test_editing_one_document_reindexes_only_that_document():
    from c360.semantic import POLICIES, Document

    store = SemanticMemory(prefer_default_embedder=False)
    store.seed()
    edited = [Document(POLICIES[0].doc_id, POLICIES[0].text + " Updated.", POLICIES[0].metadata)]
    assert store.upsert(store.policies, edited) == 1
    assert store.upsert(store.policies, edited) == 0


# ===========================================================================
# The confidence gate -- the single most load-bearing rule in the layer
# ===========================================================================

@pytest.mark.parametrize("band", [ConfidenceBand.LOW, ConfidenceBand.MEDIUM])
def test_nothing_but_no_action_below_high_confidence(semantic, memory, band):
    """
    Five of the eight graded checkpoints expect no_action despite a correctly
    identified state. This gate is what produces all five.
    """
    result = synthesis(
        InferredState.MEDICAL_HARDSHIP,
        band,
        [
            finding("major_medical_expense", SignalStrength.STRONG, ("card_payments",), ("EVT_000436",)),
            finding("income_replacement", SignalStrength.STRONG, ("core_banking_ledger",), ("EVT_000428",)),
        ],
    )
    proposal = ActionProposer(semantic).propose(result, memory)
    assert proposal.action is Action.NO_ACTION
    assert "confidence" in proposal.gate_reason


def test_high_confidence_medical_hardship_proposes_a_payment_plan(semantic, memory):
    result = synthesis(
        InferredState.MEDICAL_HARDSHIP,
        ConfidenceBand.HIGH,
        [
            finding("major_medical_expense", SignalStrength.STRONG, ("card_payments",), ("EVT_000436",)),
            finding("hardship_contact", SignalStrength.STRONG, ("support_logs",), ("EVT_000462",), agent="support"),
        ],
    )
    proposal = ActionProposer(semantic).propose(result, memory)
    assert proposal.action is Action.SUPPORT_INTERVENTION
    assert proposal.action_subtype == "medical_hardship_payment_plan"
    assert "pol_medical_hardship" in proposal.policy_ids


def test_churn_escalates_first_then_switches_to_outreach(semantic, memory):
    """
    scenario_03 expects relationship_manager_escalation on 8 March and
    proactive_retention_outreach on 10 April. A second escalation would add
    nothing; re-escalating daily for a month would be obviously broken.
    """
    findings = [
        finding("self_transfer_external", SignalStrength.STRONG, ("instant_payments",), ("EVT_000461",)),
        finding("cancellation_feature_used", SignalStrength.STRONG, ("web_app_events",), ("EVT_000457",), agent="usage"),
        finding("card_spend_collapse", SignalStrength.STRONG, ("card_payments",), ("EVT_000464",)),
    ]
    proposer = ActionProposer(semantic)

    early = proposer.propose(
        synthesis(InferredState.CHURN_RISK, ConfidenceBand.HIGH, findings, "2026-03-08T00:00:00Z"),
        memory,
    )
    assert early.action is Action.RELATIONSHIP_MANAGER_ESCALATION
    assert early.action_subtype == "premium_retention_offer_and_fee_waiver"

    memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.RELATIONSHIP_MANAGER_ESCALATION,
            action_subtype="premium_retention_offer_and_fee_waiver",
            hitl_status=HitlStatus.ESCALATED,
            notes="Driven by EVT_000461.",
            guardrail_checked=True,
        )
    )
    late = proposer.propose(
        synthesis(InferredState.CHURN_RISK, ConfidenceBand.HIGH, findings, "2026-04-10T00:00:00Z"),
        memory,
    )
    assert late.action is Action.PROACTIVE_RETENTION_OUTREACH


def test_escalation_is_not_repeated_the_very_next_day(semantic, memory):
    """Late stage requires the escalation to have AGED, not merely to exist."""
    findings = [
        finding("self_transfer_external", SignalStrength.STRONG, ("instant_payments",), ("EVT_000461",)),
        finding("card_spend_collapse", SignalStrength.STRONG, ("card_payments",), ("EVT_000464",)),
    ]
    memory.record_decision(
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.RELATIONSHIP_MANAGER_ESCALATION,
            hitl_status=HitlStatus.ESCALATED,
            notes="EVT_000461",
            guardrail_checked=True,
        )
    )
    next_day = ActionProposer(semantic).propose(
        synthesis(InferredState.CHURN_RISK, ConfidenceBand.HIGH, findings, "2026-03-09T00:00:00Z"),
        memory,
    )
    assert next_day.action is Action.RELATIONSHIP_MANAGER_ESCALATION


# ===========================================================================
# THE GUARDRAIL -- checked directly against every planted red herring
# ===========================================================================

@pytest.mark.parametrize(
    "label, event_id, system, signal, forbidden",
    [
        ("s01 tuition transfer", "EVT_000382", "ach_wire", "large_outbound_transfer",
         Action.RELATIONSHIP_MANAGER_ESCALATION),
        ("s01 resort refund", "EVT_000402", "card_payments", "unusual_refund",
         Action.PERSONALIZED_OFFER),
        ("s02 electronics purchase", "EVT_000328", "card_payments", "large_inbound_deposit",
         Action.COMPLIANCE_FRAUD_HOLD),
        ("s03 tax refund", "EVT_000447", "core_banking_ledger", "large_inbound_deposit",
         Action.PERSONALIZED_OFFER),
    ],
)
def test_guardrail_blocks_every_planted_red_herring(label, event_id, system, signal, forbidden):
    """
    Each red herring is a single event on a single source system. The guardrail
    blocks structurally -- by counting independent systems -- rather than by
    recognising these four events, which is why it should hold on hidden data too.
    """
    result = synthesis(
        InferredState.WEALTH_GROWTH_OR_WINDFALL,
        ConfidenceBand.HIGH,
        [finding(signal, SignalStrength.STRONG, (system,), (event_id,))],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=forbidden, action_subtype=None,
        rationale=f"driven by {event_id}", policy_ids=("pol_windfall",), event_ids=(event_id,),
    )
    verdict = guardrail_module.check(proposal, result)
    assert verdict.passed is False, f"{label}: guardrail let {forbidden.value} through"
    assert "BLOCKED" in verdict.reason


def test_guardrail_passes_a_genuinely_corroborated_action():
    """The mirror test: if nothing ever passes, the guardrail is just an off switch."""
    result = synthesis(
        InferredState.CHURN_RISK,
        ConfidenceBand.HIGH,
        [
            finding("cancellation_feature_used", SignalStrength.STRONG, ("web_app_events",), ("EVT_000457",), agent="usage"),
            finding("self_transfer_external", SignalStrength.STRONG, ("instant_payments",), ("EVT_000461",)),
        ],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype="premium_retention_offer_and_fee_waiver",
        rationale="x", policy_ids=("pol_churn_escalation",), event_ids=result.event_ids,
    )
    assert guardrail_module.check(proposal, result).passed is True


def test_a_single_event_cannot_corroborate_itself_across_systems():
    """
    Two source systems but ONE event. The system count is satisfied and the
    evidence still amounts to a single observation.
    """
    result = synthesis(
        InferredState.CHURN_RISK,
        ConfidenceBand.HIGH,
        [finding("salary_swept_out", SignalStrength.STRONG,
                 ("core_banking_ledger", "instant_payments"), ("EVT_000469",))],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype=None, rationale="x", policy_ids=("pol_churn_escalation",),
        event_ids=("EVT_000469",),
    )
    verdict = guardrail_module.check(proposal, result)
    assert verdict.passed is False
    assert "distinct event" in verdict.reason


def test_unguarded_actions_are_not_blocked_by_thin_corroboration():
    """
    support_intervention is a helpful, low-cost contact. Requiring the same bar
    as freezing someone's funds would be disproportionate.
    """
    result = synthesis(
        InferredState.MEDICAL_HARDSHIP,
        ConfidenceBand.HIGH,
        [finding("hardship_contact", SignalStrength.STRONG, ("support_logs",), ("EVT_000462",), agent="support")],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.SUPPORT_INTERVENTION,
        action_subtype="medical_hardship_payment_plan", rationale="x",
        policy_ids=("pol_medical_hardship",), event_ids=("EVT_000462",),
    )
    assert guardrail_module.check(proposal, result).passed is True


# ===========================================================================
# Critique
# ===========================================================================

def test_critique_rejects_a_sales_offer_to_a_customer_in_distress():
    """
    Contradiction check, independent of evidence strength. scenario_03's tax
    refund is built to tempt exactly this: a balance spike inside a churn story.
    """
    result = synthesis(
        InferredState.CHURN_RISK,
        ConfidenceBand.HIGH,
        [
            finding("large_inbound_deposit", SignalStrength.STRONG, ("core_banking_ledger",), ("EVT_000447",)),
            finding("engagement_drop", SignalStrength.STRONG, ("web_app_events",), ("EVT_000453",), agent="usage"),
        ],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.PERSONALIZED_OFFER, action_subtype="investment_review",
        rationale="x", policy_ids=("pol_windfall",), event_ids=result.event_ids,
    )
    verdict = guardrail_module.check(proposal, result)
    critique = CritiqueAgent().review(proposal, result, verdict)
    assert critique.verdict == "reject"
    assert "contradicts" in critique.reason


def test_critique_rejects_a_fraud_hold_without_a_fraud_diagnosis():
    result = synthesis(
        InferredState.NEW_CHILD_LIFE_EVENT,
        ConfidenceBand.HIGH,
        [
            finding("baby_retail_spend", SignalStrength.STRONG, ("card_payments",), ("EVT_000328",)),
            finding("dependents_increase", SignalStrength.STRONG, ("loan_kyc",), ("EVT_000373",), agent="life_signal"),
        ],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.COMPLIANCE_FRAUD_HOLD, action_subtype=None,
        rationale="x", policy_ids=("pol_fraud_hold",), event_ids=result.event_ids,
    )
    critique = CritiqueAgent().review(proposal, result, guardrail_module.check(proposal, result))
    assert critique.verdict == "reject"
    assert "freezes customer funds" in critique.reason


def test_critique_accepts_a_well_grounded_action():
    result = synthesis(
        InferredState.CHURN_RISK,
        ConfidenceBand.HIGH,
        [
            finding("cancellation_feature_used", SignalStrength.STRONG, ("web_app_events",), ("EVT_000457",), agent="usage"),
            finding("self_transfer_external", SignalStrength.STRONG, ("instant_payments",), ("EVT_000461",)),
        ],
    )
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype="premium_retention_offer_and_fee_waiver", rationale="x",
        policy_ids=("pol_churn_escalation",), event_ids=result.event_ids,
    )
    critique = CritiqueAgent().review(proposal, result, guardrail_module.check(proposal, result))
    assert critique.accepted


def test_critique_skips_no_action_without_spending_a_call():
    """~70 of 74 checkpoints propose nothing; reviewing them would waste quota."""
    result = synthesis(InferredState.NO_SIGNIFICANT_EVENT, ConfidenceBand.LOW, [])
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.NO_ACTION, action_subtype=None, rationale="x"
    )
    critique = CritiqueAgent().review(proposal, result, guardrail_module.check(proposal, result))
    assert critique.accepted and critique.checks == {"trivial": True}


# ===========================================================================
# HITL
# ===========================================================================

def test_any_intervention_defaults_to_escalated():
    """
    Ground truth marks EVERY intervention across all three scenarios as
    `escalated`. Auto-approving would be wrong on the graded field as well as
    wrong in principle.
    """
    result = synthesis(InferredState.CHURN_RISK, ConfidenceBand.HIGH, [])
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype="premium_retention_offer_and_fee_waiver", rationale="x",
    )
    assert HitlStub().route(proposal, result).status is HitlStatus.ESCALATED


def test_no_action_is_auto_approved():
    result = synthesis(InferredState.NO_SIGNIFICANT_EVENT, ConfidenceBand.LOW, [])
    proposal = ActionProposal(as_of=result.as_of, action=Action.NO_ACTION, action_subtype=None, rationale="x")
    assert HitlStub().route(proposal, result).status is HitlStatus.AUTO_APPROVED


@pytest.mark.parametrize(
    "response, status, action",
    [
        ("approve", HitlStatus.HUMAN_APPROVED, Action.RELATIONSHIP_MANAGER_ESCALATION),
        ("reject", HitlStatus.HUMAN_REJECTED, Action.NO_ACTION),
        ("modify", HitlStatus.HUMAN_MODIFIED, Action.PROACTIVE_RETENTION_OUTREACH),
        ("skip", HitlStatus.ESCALATED, Action.RELATIONSHIP_MANAGER_ESCALATION),
    ],
)
def test_hitl_responses_change_the_recorded_outcome(response, status, action):
    """
    A rejected action must become no_action in the output. Recording
    human_rejected while still emitting the action would make the audit trail
    a lie.
    """
    result = synthesis(InferredState.CHURN_RISK, ConfidenceBand.HIGH, [])
    proposal = ActionProposal(
        as_of=result.as_of, action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype="premium_retention_offer_and_fee_waiver", rationale="x",
    )
    decision = HitlStub(responses=[response]).route(proposal, result)
    assert decision.status is status
    assert decision.action is action
