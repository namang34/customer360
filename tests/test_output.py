"""
Tests for the inferred-events output writer.

These run against DUMMY decisions on purpose. The point of building the writer
now is to prove the file format is correct before any agent exists to blame.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c360.output import (  # noqa: E402
    Checkpoint,
    CheckpointError,
    InferredEventsWriter,
    load_checkpoints,
)
from c360.pii import Redactor  # noqa: E402
from c360.schema import Action, ConfidenceBand, HitlStatus, InferredState  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = [DATA / f"scenario_0{n}" for n in (1, 2, 3)]

needs_data = pytest.mark.skipif(
    not all(s.exists() for s in ALL_SCENARIOS), reason="scenario data not present"
)

REQUIRED_KEYS = {
    "as_of_time",
    "inferred_state",
    "confidence_band",
    "action",
    "action_subtype",
    "hitl_status",
    "notes",
}


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def quiet(when: str = "2026-02-15T00:00:00Z") -> Checkpoint:
    """A no_action checkpoint -- the shape of most rows in a run."""
    return Checkpoint(
        as_of_time=ts(when),
        inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
        confidence_band=ConfidenceBand.LOW,
        action=Action.NO_ACTION,
        notes="Routine activity only.",
    )


def escalation(when: str = "2026-03-08T00:00:00Z") -> Checkpoint:
    return Checkpoint(
        as_of_time=ts(when),
        inferred_state=InferredState.CHURN_RISK,
        confidence_band=ConfidenceBand.HIGH,
        action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype="premium_retention_offer_and_fee_waiver",
        hitl_status=HitlStatus.ESCALATED,
        notes="Standing instructions cancelled (EVT_000457) and savings transferred out (EVT_000461).",
        citations=("EVT_000457", "EVT_000461"),
        guardrail_checked=True,
    )


# ===========================================================================
# The file shape the grader reads
# ===========================================================================

def test_row_has_exactly_the_required_keys():
    row = quiet().to_row()
    assert set(row) == REQUIRED_KEYS, "extra or missing keys will confuse a strict scorer"


def test_timestamps_use_z_not_plus_offset():
    """
    Python renders UTC as '+00:00'; the dataset and the README example both use
    'Z'. A scorer string-matching as_of_time would miss every single row.
    """
    assert quiet().to_row()["as_of_time"] == "2026-02-15T00:00:00Z"
    assert "+00:00" not in json.dumps(quiet().to_row())


def test_non_utc_input_is_normalised_to_utc_z():
    ist = Checkpoint(
        as_of_time=ts("2026-02-15T05:30:00+05:30"),
        inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
        confidence_band=ConfidenceBand.LOW,
        action=Action.NO_ACTION,
    )
    assert ist.to_row()["as_of_time"] == "2026-02-15T00:00:00Z"


def test_enum_values_serialise_as_plain_strings():
    row = escalation().to_row()
    assert row["inferred_state"] == "churn_risk"
    assert row["action"] == "relationship_manager_escalation"
    assert row["hitl_status"] == "escalated"
    assert all(isinstance(v, (str, type(None))) for v in row.values())


def test_action_subtype_is_null_not_missing_when_absent():
    row = quiet().to_row()
    assert "action_subtype" in row and row["action_subtype"] is None


def test_plain_strings_are_accepted_and_coerced():
    """An agent may return 'churn_risk' rather than the enum member."""
    checkpoint = Checkpoint(
        as_of_time=ts("2026-03-08T00:00:00Z"),
        inferred_state="churn_risk",
        confidence_band="high",
        action="no_action",
    )
    assert checkpoint.inferred_state is InferredState.CHURN_RISK


@pytest.mark.parametrize(
    "field, bad",
    [
        ("inferred_state", "churn"),            # plausible-looking near miss
        ("inferred_state", "Churn_Risk"),       # wrong case
        ("confidence_band", "very_high"),
        ("action", "escalate"),
        ("hitl_status", "pending"),
    ],
)
def test_invented_enum_values_are_refused(field, bad):
    kwargs = dict(
        as_of_time=ts("2026-03-08T00:00:00Z"),
        inferred_state=InferredState.CHURN_RISK,
        confidence_band=ConfidenceBand.HIGH,
        action=Action.NO_ACTION,
    )
    kwargs[field] = bad
    with pytest.raises(CheckpointError, match="not a permitted value"):
        Checkpoint(**kwargs)


# ===========================================================================
# Rules that are ours, not the dataset's
# ===========================================================================

def test_action_without_a_cited_event_id_is_refused():
    """Explainability is graded at 15%. Enforced in code, not asked of an LLM."""
    with pytest.raises(CheckpointError, match="cite no event_id"):
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.SUPPORT_INTERVENTION,
            hitl_status=HitlStatus.ESCALATED,
            notes="The customer appears to be disengaging.",  # vague prose, no ids
        )


def test_no_action_needs_no_citation():
    """Otherwise ~70 of 74 rows per scenario would be impossible to emit."""
    assert quiet().to_row()["notes"] == "Routine activity only."


def test_action_cannot_be_auto_approved():
    with pytest.raises(CheckpointError, match="must not be auto_approved"):
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=Action.PROACTIVE_RETENTION_OUTREACH,
            hitl_status=HitlStatus.AUTO_APPROVED,
            notes="Driven by EVT_000461.",
        )


def test_no_action_cannot_carry_a_subtype():
    with pytest.raises(CheckpointError, match="cannot carry action_subtype"):
        Checkpoint(
            as_of_time=ts("2026-02-15T00:00:00Z"),
            inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
            confidence_band=ConfidenceBand.LOW,
            action=Action.NO_ACTION,
            action_subtype="premium_retention_offer",
        )


@pytest.mark.parametrize(
    "guarded",
    [Action.COMPLIANCE_FRAUD_HOLD, Action.RELATIONSHIP_MANAGER_ESCALATION, Action.PERSONALIZED_OFFER],
)
def test_guarded_actions_cannot_bypass_the_corroboration_check(guarded):
    """
    The >=2 independent source_systems rule lands in step 7. This makes it
    impossible to ship one of the three guarded actions without it, even by
    accident, today.
    """
    with pytest.raises(CheckpointError, match="corroboration guardrail"):
        Checkpoint(
            as_of_time=ts("2026-03-08T00:00:00Z"),
            inferred_state=InferredState.CHURN_RISK,
            confidence_band=ConfidenceBand.HIGH,
            action=guarded,
            hitl_status=HitlStatus.ESCALATED,
            notes="Driven by EVT_000461.",
            guardrail_checked=False,
        )


def test_unguarded_actions_do_not_need_the_flag():
    """support_intervention and proactive_retention_outreach are not guarded."""
    checkpoint = Checkpoint(
        as_of_time=ts("2026-03-26T00:00:00Z"),
        inferred_state=InferredState.MEDICAL_HARDSHIP,
        confidence_band=ConfidenceBand.HIGH,
        action=Action.SUPPORT_INTERVENTION,
        action_subtype="medical_hardship_payment_plan",
        hitl_status=HitlStatus.ESCALATED,
        notes="Hospital billing (EVT_000428) with income replacement (EVT_000436).",
    )
    assert checkpoint.to_row()["action"] == "support_intervention"


# ===========================================================================
# Writer-level ordering and PII
# ===========================================================================

def test_checkpoints_must_be_chronological():
    writer = InferredEventsWriter()
    writer.add(quiet("2026-02-16T00:00:00Z"))
    with pytest.raises(CheckpointError, match="chronological"):
        writer.add(quiet("2026-02-15T00:00:00Z"))


def test_duplicate_timestamps_are_refused():
    writer = InferredEventsWriter()
    writer.add(quiet("2026-02-15T00:00:00Z"))
    with pytest.raises(CheckpointError, match="duplicate checkpoint"):
        writer.add(quiet("2026-02-15T00:00:00Z"))


def test_notes_containing_the_customer_name_are_refused():
    writer = InferredEventsWriter(redactor=Redactor(customer_id="CUST_00184", name="David Chen"))
    with pytest.raises(CheckpointError, match="un-redacted PII"):
        writer.add(
            Checkpoint(
                as_of_time=ts("2026-03-08T00:00:00Z"),
                inferred_state=InferredState.CHURN_RISK,
                confidence_band=ConfidenceBand.HIGH,
                action=Action.NO_ACTION,
                notes="David Chen moved his savings out (EVT_000461).",
            )
        )


def test_redaction_preserves_the_self_transfer_signal():
    """
    The point of tokenising rather than deleting: an agent must still be able to
    tell that EVT_000461's counterparty IS the customer -- that is what makes a
    $22,500 outbound transfer a churn signal rather than an ordinary payment.
    """
    redactor = Redactor(customer_id="CUST_00184", name="David Chen")
    payload = {"counterparty_name": "David Chen - Chase Bank", "amount": 22500}
    scrubbed = redactor.scrub_payload(payload)

    assert "David" not in scrubbed["counterparty_name"]
    assert "Chase Bank" in scrubbed["counterparty_name"]     # signal survives
    assert scrubbed["amount"] == 22500                        # numbers untouched
    assert redactor.counterparty_is_customer(payload) is True
    assert payload["counterparty_name"] == "David Chen - Chase Bank"  # copy, not in place


def test_redaction_masks_contact_details_in_support_text():
    redactor = Redactor(customer_id="CUST_00184", name="David Chen")
    scrubbed = redactor.scrub("Call me on 555-123-4567 or david.chen@example.com")
    assert "555-123-4567" not in scrubbed
    assert "@example.com" not in scrubbed


# ===========================================================================
# Writing to disk
# ===========================================================================

def test_written_file_is_a_json_array_of_objects(tmp_path):
    writer = InferredEventsWriter()
    writer.add(quiet("2026-02-15T00:00:00Z"))
    writer.add(escalation("2026-03-08T00:00:00Z"))
    path = writer.write(tmp_path / "out" / "inferred_events.json")

    reloaded = load_checkpoints(path)
    assert isinstance(reloaded, list) and len(reloaded) == 2
    assert all(isinstance(row, dict) and set(row) == REQUIRED_KEYS for row in reloaded)
    assert reloaded[1]["action_subtype"] == "premium_retention_offer_and_fee_waiver"


def test_write_creates_missing_directories(tmp_path):
    writer = InferredEventsWriter()
    writer.add(quiet())
    path = writer.write(tmp_path / "deep" / "nested" / "out.json")
    assert path.exists()


def test_covers_reports_missing_graded_timestamps():
    writer = InferredEventsWriter()
    writer.add(quiet("2026-02-15T00:00:00Z"))
    required = [ts("2026-02-15T00:00:00Z"), ts("2026-03-08T00:00:00Z")]
    missing = writer.covers(required)
    assert missing == [ts("2026-03-08T00:00:00Z")]


# ===========================================================================
# End-to-end with the replay engine, using dummy decisions
# ===========================================================================

@needs_data
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)
def test_dummy_run_produces_a_valid_gradeable_file(scenario, tmp_path):
    """
    Drive the real replay engine end to end with a trivial decision rule, and
    prove the resulting file is valid and covers every graded checkpoint.

    No agent, no LLM, no intelligence -- that is the point. If this passes, the
    plumbing is correct and anything that goes wrong later is a reasoning bug,
    not a formatting one.
    """
    from c360.replay import ClockTick, EventTick, ReplayEngine

    engine = ReplayEngine(scenario, speed=0).load()
    writer = InferredEventsWriter(
        redactor=Redactor.from_entities(engine.entities), scenario_id=engine.config.scenario_id
    )

    seen_today: list[str] = []
    for tick in engine.stream():
        if isinstance(tick, EventTick):
            seen_today.append(tick.event_id)
        elif isinstance(tick, ClockTick):
            writer.add(
                Checkpoint(
                    as_of_time=tick.as_of,
                    inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
                    confidence_band=ConfidenceBand.LOW,
                    action=Action.NO_ACTION,
                    notes=(
                        f"DUMMY: {len(seen_today)} event(s) since last checkpoint"
                        + (f" ({', '.join(seen_today[:3])})" if seen_today else "")
                    ),
                )
            )
            seen_today = []

    gt = json.loads((scenario / "ground_truth.json").read_text())
    required = [ts(c["as_of_time"]) for c in gt["checkpoints"]]
    assert writer.covers(required) == [], "a graded checkpoint has no row"

    path = writer.write(tmp_path / f"{scenario.name}.json")
    rows = load_checkpoints(path)
    assert len(rows) == len(engine.checkpoint_times) == 74
    assert all(set(row) == REQUIRED_KEYS for row in rows)
    assert all(row["as_of_time"].endswith("Z") for row in rows)

    stamps = [row["as_of_time"] for row in rows]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)
