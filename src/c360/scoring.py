"""
Scoring harness -- compares an inferred-events file against a scenario's
ground_truth.json.

The problem statement offers credit for building this (section 8.1). It is also
the only way to answer "did that change help?" with a number rather than an
impression, which matters more than the credit does.

WHAT IT MEASURES, MIRRORING THE STATED EVALUATION CRITERIA
----------------------------------------------------------
  1. ACCURACY OF INFERENCE   inferred_state at each graded checkpoint
  2. TIMELINESS              confidence_band at each checkpoint, plus lead time
                             against `ideal_action_lead_time_days`
  3. APPROPRIATE ACTION      action, action_subtype and hitl_status
  4. FALSE POSITIVES         each `false_positive_check` -- did a red herring
                             trigger an action it must not, inside its window

Scored HONESTLY. Nothing here rounds in our favour, partial credit is labelled as
partial, and a missing checkpoint scores zero rather than being skipped -- because
skipping it is exactly the bug the harness exists to catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .output import load_checkpoints


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class CheckpointScore:
    as_of: str
    state_ok: bool
    confidence_ok: bool
    action_ok: bool
    subtype_ok: bool | None
    hitl_ok: bool | None
    expected: dict[str, Any]
    got: dict[str, Any] | None

    @property
    def missing(self) -> bool:
        return self.got is None

    @property
    def fully_correct(self) -> bool:
        return (
            self.state_ok
            and self.confidence_ok
            and self.action_ok
            and self.subtype_ok is not False
            and self.hitl_ok is not False
        )

    def line(self) -> str:
        if self.missing:
            return f"  {self.as_of[:10]}  MISSING -- no checkpoint emitted at this timestamp"
        mark = lambda ok: "ok  " if ok else "FAIL"  # noqa: E731
        bits = [
            f"state {mark(self.state_ok)} {self.got['inferred_state']:<28} want {self.expected['expected_inferred_state']}",
            f"conf  {mark(self.confidence_ok)} {self.got['confidence_band']:<28} want {self.expected['expected_confidence_band']}",
            f"act   {mark(self.action_ok)} {self.got['action']:<28} want {self.expected['expected_action']}",
        ]
        if self.subtype_ok is not None:
            bits.append(
                f"sub   {mark(self.subtype_ok)} {str(self.got['action_subtype']):<28} "
                f"want {self.expected.get('expected_action_subtype')}"
            )
        if self.hitl_ok is not None:
            bits.append(
                f"hitl  {mark(self.hitl_ok)} {self.got['hitl_status']:<28} "
                f"want {self.expected.get('expected_hitl_status')}"
            )
        return f"  {self.as_of[:10]}\n" + "\n".join(f"      {b}" for b in bits)


@dataclass
class FalsePositiveScore:
    event_id: str
    forbidden: list[str]
    window_hours: int
    triggered: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.triggered

    def line(self) -> str:
        if self.passed:
            return f"  {self.event_id}  ok   -- no forbidden action inside {self.window_hours}h"
        return f"  {self.event_id}  FAIL -- triggered {self.triggered} inside {self.window_hours}h"


@dataclass
class LeadTimeScore:
    as_of: str
    ideal_days: int
    achieved_days: float | None
    action: str

    @property
    def passed(self) -> bool:
        return self.achieved_days is not None and self.achieved_days >= self.ideal_days

    def line(self) -> str:
        if self.achieved_days is None:
            return f"  {self.as_of[:10]}  FAIL -- {self.action} never fired before the checkpoint"
        verdict = "ok  " if self.passed else "FAIL"
        return (
            f"  {self.as_of[:10]}  {verdict} -- {self.action} fired "
            f"{self.achieved_days:.0f} days before the checkpoint (ideal >= {self.ideal_days})"
        )


@dataclass
class ScenarioScore:
    scenario: str
    checkpoints: list[CheckpointScore] = field(default_factory=list)
    false_positives: list[FalsePositiveScore] = field(default_factory=list)
    lead_times: list[LeadTimeScore] = field(default_factory=list)

    @property
    def state_accuracy(self) -> float:
        return _ratio([c.state_ok for c in self.checkpoints])

    @property
    def confidence_accuracy(self) -> float:
        return _ratio([c.confidence_ok for c in self.checkpoints])

    @property
    def action_accuracy(self) -> float:
        return _ratio([c.action_ok for c in self.checkpoints])

    @property
    def exact_match(self) -> float:
        return _ratio([c.fully_correct for c in self.checkpoints])

    @property
    def false_positive_pass_rate(self) -> float:
        return _ratio([f.passed for f in self.false_positives])

    def report(self) -> str:
        lines = [f"=== {self.scenario}", "", "CHECKPOINTS"]
        lines += [c.line() for c in self.checkpoints]
        if self.lead_times:
            lines += ["", "LEAD TIME"] + [l.line() for l in self.lead_times]
        if self.false_positives:
            lines += ["", "FALSE-POSITIVE CHECKS"] + [f.line() for f in self.false_positives]
        lines += [
            "",
            f"  inferred_state   {self.state_accuracy:.0%}",
            f"  confidence_band  {self.confidence_accuracy:.0%}",
            f"  action           {self.action_accuracy:.0%}",
            f"  all fields exact {self.exact_match:.0%}",
            f"  false positives  {self.false_positive_pass_rate:.0%} clean",
        ]
        return "\n".join(lines)


def _ratio(flags: list[bool]) -> float:
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def score_scenario(
    scenario_dir: str | Path, output_path: str | Path, name: str | None = None
) -> ScenarioScore:
    scenario_dir = Path(scenario_dir)
    truth = json.loads((scenario_dir / "ground_truth.json").read_text(encoding="utf-8"))
    rows = load_checkpoints(output_path)
    by_time = {row["as_of_time"]: row for row in rows}
    score = ScenarioScore(scenario=name or scenario_dir.name)

    # ---- checkpoints ----------------------------------------------------
    for expected in truth["checkpoints"]:
        key = expected["as_of_time"]
        got = by_time.get(key)
        if got is None:
            score.checkpoints.append(
                CheckpointScore(key, False, False, False, None, None, expected, None)
            )
            continue

        want_subtype = expected.get("expected_action_subtype")
        want_hitl = expected.get("expected_hitl_status")
        score.checkpoints.append(
            CheckpointScore(
                as_of=key,
                state_ok=got["inferred_state"] == expected["expected_inferred_state"],
                confidence_ok=got["confidence_band"] == expected["expected_confidence_band"],
                action_ok=got["action"] == expected["expected_action"],
                subtype_ok=(got.get("action_subtype") == want_subtype) if want_subtype else None,
                hitl_ok=(got.get("hitl_status") == want_hitl) if want_hitl else None,
                expected=expected,
                got=got,
            )
        )

        # ---- lead time --------------------------------------------------
        ideal = expected.get("ideal_action_lead_time_days")
        if ideal:
            score.lead_times.append(
                _lead_time(rows, key, expected["expected_action"], int(ideal))
            )

    # ---- false positives -------------------------------------------------
    for check in truth.get("false_positive_checks", []):
        score.false_positives.append(_false_positive(rows, scenario_dir, check))

    return score


def _lead_time(rows, checkpoint_key: str, action: str, ideal_days: int) -> LeadTimeScore:
    """
    How many days BEFORE the graded checkpoint did the required action first fire?

    The ground truth frames these checkpoints as deadlines, not targets:
    scenario_03's note says that waiting until 10 April means the system "has
    failed the early lead time test". So firing ON the checkpoint day scores zero
    lead time, and only firing earlier counts.
    """
    deadline = _ts(checkpoint_key)
    first = next(
        (row["as_of_time"] for row in rows if row["action"] == action and _ts(row["as_of_time"]) <= deadline),
        None,
    )
    if first is None:
        return LeadTimeScore(checkpoint_key, ideal_days, None, action)
    achieved = (deadline - _ts(first)).total_seconds() / 86400
    return LeadTimeScore(checkpoint_key, ideal_days, achieved, action)


def _false_positive(rows, scenario_dir: Path, check: dict[str, Any]) -> FalsePositiveScore:
    """
    Did a forbidden action fire inside the red herring's window?

    The window is anchored on the herring's OWN event_time, read back out of the
    scenario files -- ground_truth gives the event_id and the window length but
    not the timestamp, and hard-coding it would silently rot if the data changed.
    """
    event_id = check["event_id"]
    forbidden = list(check.get("must_not_trigger_action", []))
    window_hours = int(check.get("window_hours", 72))

    when = _find_event_time(scenario_dir, event_id)
    result = FalsePositiveScore(event_id, forbidden, window_hours)
    if when is None:
        return result

    end = when + timedelta(hours=window_hours)
    for row in rows:
        moment = _ts(row["as_of_time"])
        if when <= moment <= end and row["action"] in forbidden:
            result.triggered.append(f"{row['action']}@{row['as_of_time'][:10]}")
    return result


def _find_event_time(scenario_dir: Path, event_id: str) -> datetime | None:
    for name in ("live_stream.jsonl", "history_seed.jsonl"):
        path = scenario_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or event_id not in line:
                    continue
                row = json.loads(line)
                if row.get("event_id") == event_id:
                    return _ts(row["event_time"])
    return None


def overall_report(scores: list[ScenarioScore]) -> str:
    lines = ["", "=" * 72, "OVERALL", "=" * 72]
    all_checkpoints = [c for s in scores for c in s.checkpoints]
    all_fp = [f for s in scores for f in s.false_positives]
    all_lead = [l for s in scores for l in s.lead_times]

    lines += [
        f"  checkpoints scored     {len(all_checkpoints)}",
        f"  inferred_state         {_ratio([c.state_ok for c in all_checkpoints]):.0%}",
        f"  confidence_band        {_ratio([c.confidence_ok for c in all_checkpoints]):.0%}",
        f"  action                 {_ratio([c.action_ok for c in all_checkpoints]):.0%}",
        f"  all fields exact       {_ratio([c.fully_correct for c in all_checkpoints]):.0%}",
        f"  false-positive checks  {_ratio([f.passed for f in all_fp]):.0%} clean "
        f"({len(all_fp)} checked)",
    ]
    if all_lead:
        lines.append(f"  lead-time targets      {_ratio([l.passed for l in all_lead]):.0%} met")
    failures = [c for c in all_checkpoints if not c.fully_correct]
    if failures:
        lines += ["", "  NOT FULLY CORRECT:"]
        lines += [f"    {c.as_of[:10]} ({'missing' if c.missing else 'field mismatch'})" for c in failures]
    return "\n".join(lines)
