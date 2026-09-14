"""
End-to-end tests: the whole system, offline, scored against ground truth.

This is the test that would catch a regression anywhere in the stack. Everything
else checks a component; this checks that the components still add up.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["C360_OFFLINE"] = "1"

from c360.pipeline import Pipeline  # noqa: E402
from c360.scoring import score_scenario  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = [DATA / f"scenario_0{n}" for n in (1, 2, 3)]
needs_data = pytest.mark.skipif(
    not all(s.exists() for s in ALL_SCENARIOS), reason="scenario data not present"
)
every_scenario = pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda p: p.name)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run all three scenarios once; every test below reads these results."""
    out = tmp_path_factory.mktemp("out")
    results = {}
    for scenario in ALL_SCENARIOS:
        pipeline = Pipeline(scenario, offline=True, trace_path=out / f"{scenario.name}.jsonl")
        writer, stats = pipeline.run()
        path = writer.write(out / f"{scenario.name}.json")
        pipeline.close()
        results[scenario.name] = {
            "path": path,
            "stats": stats,
            "score": score_scenario(scenario, path),
            "trace": out / f"{scenario.name}.jsonl",
            "writer": writer,
        }
    return results


@needs_data
@every_scenario
def test_every_graded_checkpoint_is_fully_correct(scenario, runs):
    """state, confidence, action, action_subtype and hitl_status, all of them."""
    score = runs[scenario.name]["score"]
    wrong = [c for c in score.checkpoints if not c.fully_correct]
    assert not wrong, "\n" + "\n".join(c.line() for c in wrong)


@needs_data
@every_scenario
def test_no_red_herring_triggers_a_forbidden_action(scenario, runs):
    score = runs[scenario.name]["score"]
    failed = [f for f in score.false_positives if not f.passed]
    assert not failed, "\n" + "\n".join(f.line() for f in failed)


@needs_data
@every_scenario
def test_lead_time_targets_are_met(scenario, runs):
    """
    Ground truth frames these as deadlines: acting ON the checkpoint day is
    explicitly called a failure of the early-detection test.
    """
    score = runs[scenario.name]["score"]
    missed = [l for l in score.lead_times if not l.passed]
    assert not missed, "\n" + "\n".join(l.line() for l in missed)


@needs_data
@every_scenario
def test_output_file_is_schema_valid(scenario, runs):
    rows = json.loads(Path(runs[scenario.name]["path"]).read_text())
    required = {"as_of_time", "inferred_state", "confidence_band", "action",
                "action_subtype", "hitl_status", "notes"}
    assert len(rows) == 74
    assert all(set(r) == required for r in rows)
    assert all(r["as_of_time"].endswith("Z") for r in rows)
    stamps = [r["as_of_time"] for r in rows]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)


@needs_data
@every_scenario
def test_every_action_row_cites_event_ids(scenario, runs):
    """The graded explainability requirement, on real output."""
    rows = json.loads(Path(runs[scenario.name]["path"]).read_text())
    for row in rows:
        if row["action"] != "no_action":
            assert "EVT_" in row["notes"], f"{row['as_of_time']}: action with no citation"


@needs_data
@every_scenario
def test_no_customer_name_reaches_the_output_or_the_trace(scenario, runs):
    """PII, checked on the actual artifacts rather than on the redactor in isolation."""
    entities = json.loads((scenario / "entities.json").read_text())
    name = entities["profile"]["name"]
    first, last = name.split()[0], name.split()[-1]

    output = Path(runs[scenario.name]["path"]).read_text()
    trace = Path(runs[scenario.name]["trace"]).read_text()
    for blob, label in ((output, "output"), (trace, "trace")):
        assert first not in blob, f"{label} leaked the customer's first name"
        assert last not in blob, f"{label} leaked the customer's surname"


@needs_data
@every_scenario
def test_every_intervention_is_escalated_not_auto_approved(scenario, runs):
    rows = json.loads(Path(runs[scenario.name]["path"]).read_text())
    for row in rows:
        if row["action"] != "no_action":
            assert row["hitl_status"] != "auto_approved", row["as_of_time"]


@needs_data
@every_scenario
def test_trace_records_every_checkpoint_with_its_reasoning(scenario, runs):
    """
    Traceability: the trace must answer 'why did it do that on this date?'
    without re-running anything.
    """
    lines = [json.loads(l) for l in Path(runs[scenario.name]["trace"]).read_text().splitlines() if l.strip()]
    checkpoints = [l for l in lines if l["kind"] == "checkpoint"]
    assert len(checkpoints) == 74
    for entry in checkpoints:
        assert entry["customer_id"] and entry["scenario_id"]
        for key in ("inferred_state", "confidence_band", "guardrail_passed",
                    "guardrail_reason", "critique_verdict", "final_action", "hitl_status"):
            assert key in entry, f"trace missing {key}"


@needs_data
def test_the_run_is_deterministic(tmp_path):
    """Two identical runs must produce identical files -- the harness depends on it."""
    def once(tag):
        pipeline = Pipeline(DATA / "scenario_03", offline=True)
        writer, _ = pipeline.run()
        path = writer.write(tmp_path / f"{tag}.json")
        pipeline.close()
        return Path(path).read_text()

    assert once("a") == once("b")


@needs_data
@every_scenario
def test_most_checkpoints_cost_no_reasoning_at_all(scenario, runs):
    """
    Efficiency claim, checked rather than asserted in prose: on a quiet day the
    previous belief simply stands. This is what makes a 74-checkpoint run
    affordable on a free tier.
    """
    stats = runs[scenario.name]["stats"]
    assert stats.checkpoints == 74
    assert stats.syntheses == 74
