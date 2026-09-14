"""
Tests for the ambiguity review queue.

Two things are being protected here.

The FIRST is that the queue notices what the graded path is structurally unable
to notice: a checkpoint where several source systems agree that something is
happening, but not strongly enough for Gate 1 to permit an action. Those become
no_action, and no_action is auto-approved, so without this queue nobody is ever
told. Production Bar Checklist 6.3 asks for exactly that escalation.

The SECOND, and the one more likely to regress, is that the queue stays QUIET.
Its first implementation raised an item on every qualifying checkpoint and
produced 38 across scenario_01's 74 -- the same situation restated daily for five
weeks. A queue that long is one nobody reads. test_persistent_situation_raised_once
is the guard against that coming back.
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

from c360.action import ActionProposal  # noqa: E402
from c360.findings import Finding, SignalStrength  # noqa: E402
from c360.pipeline import Pipeline  # noqa: E402
from c360.review import ReviewQueue, assess  # noqa: E402
from c360.schema import Action, ConfidenceBand, InferredState  # noqa: E402
from c360.state_board import Corroboration  # noqa: E402
from c360.synthesis import Synthesis  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = [DATA / f"scenario_0{n}" for n in (1, 2, 3)]
needs_data = pytest.mark.skipif(
    not all(p.is_dir() for p in ALL_SCENARIOS), reason="scenario data not present"
)

NOW = datetime(2026, 2, 15, tzinfo=timezone.utc)


def _synthesis(
    state=InferredState.CHURN_RISK,
    band=ConfidenceBand.LOW,
    sources=("support_logs", "web_app_events"),
    affinity=None,
):
    corroboration = Corroboration(
        as_of=NOW,
        window_days=30,
        findings=(),
        source_systems=tuple(sources),
        agents=("support", "usage"),
        event_ids=("EVT_1", "EVT_2"),
    )
    return Synthesis(
        as_of=NOW,
        inferred_state=state,
        confidence_band=band,
        rationale="test",
        event_ids=("EVT_1", "EVT_2"),
        corroboration=corroboration,
        affinity=affinity if affinity is not None else {state.value: 8.0},
    )


def _no_action():
    return ActionProposal(as_of=NOW, action=Action.NO_ACTION, action_subtype=None, rationale="")


# ---------------------------------------------------------------- what it catches

def test_weak_but_corroborated_is_flagged():
    """The gap this module exists to close: breadth without strength."""
    item = assess(_synthesis(), _no_action())
    assert item is not None
    assert item.reason == "weak_but_corroborated"
    assert "support_logs" in item.detail


def test_contested_diagnosis_is_flagged():
    """Two states within 25% -- the arithmetic picked a winner, but only just."""
    item = assess(
        _synthesis(affinity={"churn_risk": 8.0, "job_loss_or_income_disruption": 7.0}),
        _no_action(),
    )
    assert item is not None
    assert item.reason == "contested"
    assert "job_loss_or_income_disruption" in item.detail


# ---------------------------------------------------------------- what it ignores

def test_not_flagged_when_an_action_was_taken():
    """An action already puts a human in the loop; a second signal is noise."""
    proposal = ActionProposal(
        as_of=NOW, action=Action.RELATIONSHIP_MANAGER_ESCALATION,
        action_subtype="retention_call", rationale="",
    )
    assert assess(_synthesis(), proposal) is None


def test_not_flagged_when_nothing_is_suspected():
    assert assess(_synthesis(state=InferredState.NO_SIGNIFICANT_EVENT), _no_action()) is None


def test_not_flagged_at_high_confidence():
    """High confidence with no action is a considered refusal, not an open question."""
    assert assess(_synthesis(band=ConfidenceBand.HIGH), _no_action()) is None


def test_not_flagged_on_a_single_source_system():
    """One system alone is the isolated-anomaly shape the guardrail already rejects."""
    assert assess(_synthesis(sources=("support_logs",)), _no_action()) is None


# ---------------------------------------------------------------- staying quiet

def test_persistent_situation_raised_once():
    """
    THE ALERT-FATIGUE GUARD.

    Thirty identical days must produce one item, not thirty. The first version of
    this queue got this wrong and flagged 38 of scenario_01's 74 checkpoints.
    """
    queue = ReviewQueue()
    for _ in range(30):
        queue.consider(_synthesis(), _no_action())
    assert len(queue) == 1


def test_queue_raises_again_when_the_picture_changes():
    """Persistence is silent; a change is news."""
    queue = ReviewQueue()
    queue.consider(_synthesis(band=ConfidenceBand.LOW), _no_action())
    queue.consider(_synthesis(band=ConfidenceBand.LOW), _no_action())
    queue.consider(_synthesis(band=ConfidenceBand.MEDIUM), _no_action())
    assert len(queue) == 2

    # a new source system joining is also a change worth surfacing
    queue.consider(
        _synthesis(band=ConfidenceBand.MEDIUM,
                   sources=("support_logs", "web_app_events", "core_banking_ledger")),
        _no_action(),
    )
    assert len(queue) == 3


def test_resolved_situation_can_be_raised_again_later():
    queue = ReviewQueue()
    queue.consider(_synthesis(), _no_action())
    queue.consider(_synthesis(state=InferredState.NO_SIGNIFICANT_EVENT), _no_action())
    queue.consider(_synthesis(), _no_action())
    assert len(queue) == 2


# ---------------------------------------------------------------- integration

@needs_data
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)
def test_queue_stays_short_on_every_scenario(scenario):
    """
    A queue is only useful if a human would actually work through it. Ten items
    across a 74-day window is a morning's glance; forty is a thing people ignore.
    """
    pipeline = Pipeline(scenario, offline=True)
    _, stats = pipeline.run()
    try:
        assert len(pipeline.review) <= 10, pipeline.review.summary()
        assert stats.flagged_for_review == len(pipeline.review)
    finally:
        pipeline.close()


@needs_data
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)
def test_review_queue_never_touches_the_graded_output(scenario):
    """
    The whole design rests on this: hitl_status is graded, and ground truth marks
    these checkpoints auto_approved. The queue must be a parallel channel, so a
    flagged checkpoint has to look identical in the graded file to an unflagged
    one.
    """
    pipeline = Pipeline(scenario, offline=True)
    writer, _ = pipeline.run()
    try:
        flagged = {item.as_of for item in pipeline.review.items}
        assert flagged, "expected at least one flagged checkpoint to test against"
        for checkpoint in writer.checkpoints:
            if checkpoint.as_of_time in flagged:
                assert checkpoint.action is Action.NO_ACTION
                assert checkpoint.hitl_status.value == "auto_approved"
    finally:
        pipeline.close()


@needs_data
def test_rows_are_json_serialisable():
    pipeline = Pipeline(ALL_SCENARIOS[2], offline=True)
    pipeline.run()
    try:
        rows = pipeline.review.to_rows()
        json.dumps(rows)  # must not raise
        assert all("as_of_time" in r and r["as_of_time"].endswith("Z") for r in rows)
    finally:
        pipeline.close()
