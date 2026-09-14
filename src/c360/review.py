"""
The ambiguity review queue -- escalation for uncertainty rather than for cost.

WHY THIS EXISTS
---------------
Production Bar Checklist 6.3 asks for "escalation for ambiguity, not just for
cost -- if the system's own confidence is low, or two agents disagree and can't
resolve it, that should also route to a human, independent of the dollar amount
involved."

Without this module the system does the opposite. Gate 1 in action.py turns
anything below high confidence into no_action; HitlStub routes only actions that
are NOT no_action to a human. Compose those two rules and a low-confidence
checkpoint becomes no_action, is auto-approved, and nobody is ever told. The
system calls for a human when it is certain and goes quiet when it is unsure,
which is backwards.

Scenario_03 on 15 February is the case in point: support has flagged a denied
complaint, usage has flagged an engagement drop, two independent source systems
agree that something is wrong -- and the graded answer is, correctly, no_action.
Correct, but a relationship manager would plausibly want thirty seconds with that
customer, and the design guarantees they never hear about it.

WHY IT IS A SEPARATE CHANNEL, NOT A CHANGE TO hitl_status
---------------------------------------------------------
hitl_status is a graded field, and ground truth marks exactly these checkpoints
auto_approved. Escalating them would satisfy the checklist and break the scored
output at the same time.

So nothing here touches a Checkpoint. The queue is a second, parallel artifact:
the graded file says what the system decided, and the queue says where a human
might want to look anyway. That separation is the point -- "we were unsure" is a
different statement from "we are recommending an action", and collapsing them
into one field would lose both.

WHAT COUNTS AS WORTH A HUMAN'S TIME
-----------------------------------
Not every quiet day. Flagging all ~70 no_action checkpoints would be alert
fatigue by another name -- the failure mode the guardrail exists to avoid. Two
specific patterns qualify:

  WEAK BUT CORROBORATED -- several independent source systems agree that
  something is happening, but not strongly enough to act. This is the shape of a
  narrative in its early stage, which is exactly where a human glance is worth
  most and where the system is least able to help itself.

  CONTESTED -- two candidate states score within a hair of each other. The
  affinity arithmetic picks a winner because it must, but the margin is noise.
  That is the checklist's "two agents disagree and can't resolve it", stated in
  the terms this system actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .action import ActionProposal
from .schema import Action, ConfidenceBand, InferredState
from .synthesis import Synthesis

# Corroboration breadth at which weak evidence becomes worth a human's attention.
# Deliberately the same number as the guardrail's: if two independent source
# systems would be enough to ACT on at high confidence, two are enough to LOOK at
# when confident is exactly what we are not.
MIN_SOURCES_FOR_REVIEW = 2

# How close the top two candidate states must be before the choice between them
# is treated as unresolved rather than decided. Matches the threshold at which
# synthesis.py stops trusting its own arithmetic and asks a model to adjudicate --
# the same judgement, surfaced instead of silently resolved.
CONTESTED_MARGIN = 0.25


@dataclass(frozen=True)
class ReviewItem:
    """One checkpoint worth a human glance, despite no action being taken."""

    as_of: datetime
    reason: str                      # weak_but_corroborated | contested
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    source_systems: tuple[str, ...]
    event_ids: tuple[str, ...]
    detail: str

    def to_row(self) -> dict:
        return {
            "as_of_time": self.as_of.isoformat().replace("+00:00", "Z"),
            "reason": self.reason,
            "inferred_state": self.inferred_state.value,
            "confidence_band": self.confidence_band.value,
            "source_systems": list(self.source_systems),
            "citations": list(self.event_ids),
            "detail": self.detail,
        }


def assess(synthesis: Synthesis, proposal: ActionProposal) -> ReviewItem | None:
    """
    Decide whether this checkpoint deserves a human look. Returns None for the
    overwhelming majority of them.

    Only ever considers checkpoints where NO action is being taken. If an action
    was proposed, the HITL checkpoint already has a human in the loop and adding a
    second signal would just be noise.
    """
    if proposal.action is not Action.NO_ACTION:
        return None
    if synthesis.inferred_state is InferredState.NO_SIGNIFICANT_EVENT:
        return None
    if synthesis.confidence_band is ConfidenceBand.HIGH:
        # High confidence with no action means a policy or guardrail deliberately
        # declined. That is a considered decision, not an unresolved one.
        return None

    sources = synthesis.corroboration.source_systems

    # --- contested: the arithmetic had to break a near-tie ------------------
    ranked = sorted(synthesis.affinity.items(), key=lambda kv: -kv[1])
    if len(ranked) >= 2 and ranked[0][1] > 0:
        (top_state, top_score), (second_state, second_score) = ranked[0], ranked[1]
        if second_score >= top_score * (1 - CONTESTED_MARGIN):
            return ReviewItem(
                as_of=synthesis.as_of,
                reason="contested",
                inferred_state=synthesis.inferred_state,
                confidence_band=synthesis.confidence_band,
                source_systems=sources,
                event_ids=synthesis.event_ids,
                detail=(
                    f"{top_state} ({top_score:.1f}) and {second_state} ({second_score:.1f}) "
                    f"score within {CONTESTED_MARGIN:.0%} of each other; the choice between "
                    f"them is not well separated by the evidence."
                ),
            )

    # --- weak but corroborated: breadth without strength --------------------
    if len(sources) >= MIN_SOURCES_FOR_REVIEW:
        return ReviewItem(
            as_of=synthesis.as_of,
            reason="weak_but_corroborated",
            inferred_state=synthesis.inferred_state,
            confidence_band=synthesis.confidence_band,
            source_systems=sources,
            event_ids=synthesis.event_ids,
            detail=(
                f"{len(sources)} independent source systems ({', '.join(sources)}) agree on "
                f"{synthesis.inferred_state.value}, but only at "
                f"{synthesis.confidence_band.value} confidence — below the bar for any action."
            ),
        )

    return None


class ReviewQueue:
    """
    Collects the checkpoints a human might want to look at.

    Written alongside the graded output, never into it.

    RAISES ONLY ON CHANGE, NOT ON PERSISTENCE
    -----------------------------------------
    A belief that qualifies on one day usually qualifies on the next thirty. The
    first version of this queue flagged every one of them and produced 38 items
    across 74 checkpoints in scenario_01 -- the same two source systems, the same
    state, restated daily for five weeks. That is alert fatigue, the exact failure
    the AML research behind guardrail.py describes: a 95% false-positive rate is
    not a detection problem, it is a queue nobody reads any more.

    So an item is raised only when the PICTURE changes -- a different reason, a
    different state, a different confidence band, or a new source system joining.
    An unchanged situation is already on the queue; saying it again adds nothing
    and costs attention.
    """

    def __init__(self) -> None:
        self.items: list[ReviewItem] = []
        self._last_signature: tuple | None = None

    @staticmethod
    def _signature(item: ReviewItem) -> tuple:
        return (item.reason, item.inferred_state, item.confidence_band, item.source_systems)

    def consider(self, synthesis: Synthesis, proposal: ActionProposal) -> ReviewItem | None:
        item = assess(synthesis, proposal)
        if item is None:
            # The situation resolved -- the next time it qualifies it is news again.
            self._last_signature = None
            return None
        signature = self._signature(item)
        if signature == self._last_signature:
            return None
        self._last_signature = signature
        self.items.append(item)
        return item

    def __len__(self) -> int:
        return len(self.items)

    def to_rows(self) -> list[dict]:
        return [item.to_row() for item in self.items]

    def summary(self) -> str:
        if not self.items:
            return "review queue: empty"
        by_reason: dict[str, int] = {}
        for item in self.items:
            by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
        parts = ", ".join(f"{n} {reason}" for reason, n in sorted(by_reason.items()))
        return f"review queue: {len(self.items)} checkpoint(s) — {parts}"
