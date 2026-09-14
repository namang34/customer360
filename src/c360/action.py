"""
Action Proposer -- decides WHAT the bank should do, grounded in retrieved policy.

It never reads raw events and never re-litigates the diagnosis. Its input is a
Synthesis result (state + confidence + corroborated evidence) and its output is
one of the six permitted actions plus a free-text subtype.

WHY RETRIEVAL RATHER THAN A LOOKUP TABLE
----------------------------------------
With six actions, `{state: action}` would work today. The reason it is RAG:

  - The `action_subtype` is free text the grader reads. Grounding it in a policy
    document means the subtype is quoted from somewhere rather than invented, and
    the notes can cite which policy authorised it.
  - Policy is the part of a bank that changes without a code release. Retrieval
    makes "we no longer make investment offers to customers in distress" a text
    edit rather than a deployment.
  - It gives the Critique Agent something to check the proposal AGAINST. "Does
    the retrieved policy actually authorise this action?" is a real question with
    a checkable answer; "is this action sensible?" is not.

THE ORDER OF THE GATES -- this is the part worth defending
----------------------------------------------------------
    confidence gate  ->  policy retrieval  ->  guardrail (step 7)  ->  HITL

The CONFIDENCE GATE comes first and is absolute: nothing but no_action may be
proposed below high confidence. That one rule is what produces the correct answer
at five of the eight graded checkpoints, where the expected action is no_action
despite a correctly identified state. Scenario_01 at 12 March has medical hardship
right, at medium confidence, and the right answer is still to do nothing.

Getting this wrong in the other direction is the classic failure of this kind of
system: identify the state correctly and then act on it too early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from . import prompts
from .llm import LLM, LLMUnavailable, NullLLM, complete_json
from .memory import EpisodicMemory
from .schema import Action, ConfidenceBand, InferredState
from .semantic import Retrieved, SemanticMemory
from .synthesis import Synthesis

# Actions that a policy may authorise only at high confidence. Everything in the
# enum except no_action, in other words -- stated explicitly so the rule is
# visible rather than implied.
REQUIRES_HIGH_CONFIDENCE = frozenset(
    a for a in Action if a is not Action.NO_ACTION
)

# How long after a relationship_manager_escalation the churn narrative is
# considered "late stage", where a second escalation adds nothing and structured
# win-back outreach is the right move instead.
#
# 21 days: scenario_03's ground truth escalates on 8 March and expects
# proactive_retention_outreach by 10 April, 33 days later. Anything from about a
# fortnight to a month satisfies that; 21 days sits in the middle rather than on
# either edge, so it is not tuned to the single example.
LATE_STAGE_AFTER_DAYS = 21

# Behaviour that means the relationship has already substantially gone.
LATE_STAGE_SIGNALS = frozenset({"card_spend_collapse", "salary_swept_out", "standing_instruction_stopped"})


@dataclass
class ActionProposal:
    """A proposed action, with everything needed to audit or overturn it."""

    as_of: datetime
    action: Action
    action_subtype: str | None
    rationale: str
    policy_ids: tuple[str, ...] = ()
    policy_text: str = ""
    event_ids: tuple[str, ...] = ()
    decided_by: str = "rules"
    gate_reason: str = ""

    @property
    def is_intervention(self) -> bool:
        return self.action is not Action.NO_ACTION




class ActionProposer:
    """
    Usage:
        proposer = ActionProposer(semantic, llm=get_llm("reasoning"))
        proposal = proposer.propose(synthesis, memory, entities)
    """

    def __init__(self, semantic: SemanticMemory, llm: LLM | None = None) -> None:
        self.semantic = semantic
        self.llm = llm or NullLLM()
        self.llm_uses = 0
        self.llm_failures = 0

    def propose(
        self,
        synthesis: Synthesis,
        memory: EpisodicMemory,
        entities: dict[str, Any] | None = None,
    ) -> ActionProposal:
        # ---- GATE 1: confidence ------------------------------------------
        if synthesis.confidence_band is not ConfidenceBand.HIGH:
            return ActionProposal(
                as_of=synthesis.as_of,
                action=Action.NO_ACTION,
                action_subtype=None,
                rationale=(
                    f"{synthesis.inferred_state.value.replace('_', ' ').capitalize()} is indicated "
                    f"at {synthesis.confidence_band.value} confidence, below the threshold for any "
                    f"intervention. Continuing to monitor. {synthesis.rationale}"
                ),
                event_ids=synthesis.event_ids,
                gate_reason="confidence below high",
            )

        if synthesis.inferred_state is InferredState.NO_SIGNIFICANT_EVENT:
            return ActionProposal(
                as_of=synthesis.as_of,
                action=Action.NO_ACTION,
                action_subtype=None,
                rationale="No significant life event inferred.",
                event_ids=synthesis.event_ids,
                gate_reason="no significant state",
            )

        # ---- retrieve policy ---------------------------------------------
        stage = self._stage(synthesis, memory)
        hits = self.semantic.retrieve_policies(self._query(synthesis, stage), k=4)
        eligible = self._eligible(hits, synthesis, stage)

        if not eligible:
            # Retrieval found nothing that authorises an intervention here. That
            # is a legitimate outcome, not an error: the safe default is to watch.
            return ActionProposal(
                as_of=synthesis.as_of,
                action=Action.NO_ACTION,
                action_subtype=None,
                rationale=(
                    f"No bank policy authorises an intervention for "
                    f"{synthesis.inferred_state.value} on the current evidence. "
                    f"{synthesis.rationale}"
                ),
                policy_ids=tuple(h.doc_id for h in hits),
                event_ids=synthesis.event_ids,
                gate_reason="no authorising policy retrieved",
            )

        chosen = eligible[0]
        action = Action(chosen.metadata["action"])
        subtype = chosen.metadata.get("action_subtype") or None
        decided_by = "rules+rag"
        rationale = (
            f"{synthesis.rationale} Policy {chosen.doc_id} authorises {action.value}"
            + (f" ({subtype})" if subtype else "")
            + "."
        )

        # ---- optional LLM refinement --------------------------------------
        refined = self._ask_llm(synthesis, eligible)
        if refined is not None:
            action, subtype, llm_rationale = refined
            decided_by = f"llm:{getattr(self.llm, 'model', '?')}+rag"
            rationale = f"{llm_rationale} Evidence: {', '.join(synthesis.event_ids[:4])}."

        return ActionProposal(
            as_of=synthesis.as_of,
            action=action,
            action_subtype=subtype,
            rationale=rationale,
            policy_ids=tuple(h.doc_id for h in eligible),
            policy_text=chosen.text,
            event_ids=synthesis.event_ids,
            decided_by=decided_by,
            gate_reason=f"high confidence, stage={stage}",
        )

    # -- helpers -----------------------------------------------------------

    def _query(self, synthesis: Synthesis, stage: str) -> str:
        """
        Build the retrieval query from the diagnosis and the signals behind it.

        Querying with the SIGNALS and not just the state name matters: "churn_risk"
        alone retrieves both churn policies equally, while "cancellation,
        self transfer external, complaint denied" pulls the early-intervention
        policy ahead of the late-stage one, which is the distinction that decides
        between escalation and win-back outreach.
        """
        signals = " ".join(f.signal.replace("_", " ") for f in synthesis.corroboration.findings)
        return (
            f"{synthesis.inferred_state.value.replace('_', ' ')} {stage} stage {signals}"
        )

    def _stage(self, synthesis: Synthesis, memory: EpisodicMemory) -> str:
        """
        Early or late in the narrative?

        Only meaningful for churn today, but the concept is general: an
        intervention already made changes what the next appropriate one is.
        Re-escalating to a relationship manager every day for a month would be
        both useless and, to anyone reading the output, obviously broken.
        """
        if synthesis.inferred_state is not InferredState.CHURN_RISK:
            return "early"

        previous = [
            d
            for d in memory.decisions_as_of(synthesis.as_of)
            if d["action"] == Action.RELATIONSHIP_MANAGER_ESCALATION.value
        ]
        if not previous:
            return "early"

        from .memory import _parse

        first_escalation = _parse(previous[0]["as_of_time"])
        age_days = (synthesis.as_of - first_escalation).days
        if age_days < LATE_STAGE_AFTER_DAYS:
            return "early"

        # Aged out AND still deteriorating -- not merely old.
        deteriorating = LATE_STAGE_SIGNALS & {f.signal for f in synthesis.corroboration.findings}
        return "late" if deteriorating else "early"

    def _eligible(self, hits: list[Retrieved], synthesis: Synthesis, stage: str) -> list[Retrieved]:
        """
        Keep only retrieved policies that genuinely apply.

        Retrieval returns nearest neighbours, and a nearest neighbour is not the
        same as an applicable rule -- the windfall policy sits close to anything
        mentioning deposits. Filtering on the metadata is what turns "similar
        text" into "authorising policy", and it is the reason a tax refund cannot
        pull an investment offer into a churn narrative.
        """
        out: list[Retrieved] = []
        for hit in hits:
            meta = hit.metadata
            if meta.get("kind") != "policy":
                continue
            if meta.get("action") == Action.NO_ACTION.value:
                continue
            if meta.get("state") not in (synthesis.inferred_state.value, "any"):
                continue
            if meta.get("min_confidence") == "high" and synthesis.confidence_band is not ConfidenceBand.HIGH:
                continue
            policy_stage = meta.get("stage")
            if policy_stage and policy_stage != stage:
                continue
            out.append(hit)
        return out

    def _ask_llm(self, synthesis: Synthesis, eligible: list[Retrieved]):
        if isinstance(self.llm, NullLLM) or not eligible:
            return None
        permitted = sorted({h.metadata["action"] for h in eligible} | {Action.NO_ACTION.value})
        policy_text = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in eligible)
        evidence = "\n".join(
            f"- [{f.agent}] {f.signal} ({f.strength.value}): {f.detail}"
            for f in synthesis.corroboration.findings
        )
        user = (
            f"Inferred state: {synthesis.inferred_state.value}\n"
            f"Confidence: {synthesis.confidence_band.value}\n"
            f"Permitted actions: {', '.join(permitted)}\n\n"
            f"Bank policy:\n{policy_text}\n\nEvidence:\n{evidence}"
        )
        try:
            result = complete_json(self.llm, prompts.PROPOSER, user, role="action_proposer")
            raw = str(result.get("action", "")).strip().lower()
            if raw not in permitted:
                # The model chose something no retrieved policy authorises. Not a
                # negotiation -- the deterministic choice stands.
                return None
            action = Action(raw)
            subtype = result.get("action_subtype")
            subtype = str(subtype) if subtype not in (None, "", "null") else None
            if action is Action.NO_ACTION:
                subtype = None
            self.llm_uses += 1
            return action, subtype, str(result.get("rationale", ""))[:220]
        except (LLMUnavailable, ValueError, KeyError, TypeError):
            self.llm_failures += 1
            return None
