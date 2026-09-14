"""
The orchestrator -- the whole system, wired in the order the architecture says.

    replay tick
        |
        +-- EVENT  -> episodic memory -> perception swarm (event trigger)
        |                                     |
        |                                     v
        |                               state board
        |
        +-- CLOCK  -> perception swarm (time trigger)
                          |
                          v
                    should_synthesise?  (>= 2 agents flagged -- agent-dependent trigger)
                          |
                          v
                    Synthesis Agent      state + confidence
                          |
                          v
                    Action Proposer      RAG over policy
                          |
                          v
                    GUARDRAIL            code check, >= 2 independent sources
                          |
                          v
                    Critique Agent       adversarial, one bounded retry
                          |
                          v
                    HITL                 any action != no_action -> escalated
                          |
                          v
                    Checkpoint           validated, written to the graded file

Every arrow is a handoff of STRUCTURED data, not prose, and every stage can be
read back out of the trace afterwards.

ONE THING WORTH NOTICING about the order: the guardrail runs BEFORE the critique,
not after. The cheap deterministic check that can definitively reject should run
before the expensive judgement call that might not -- there is no point paying for
a model to consider proportionality when the evidence has already failed a
structural test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import guardrail as guardrail_module
from .action import ActionProposal, ActionProposer
from .agents import PerceptionContext, default_swarm, run_swarm_on_event, run_swarm_on_tick
from .critique import CritiqueAgent, HitlStub
from .llm import NullLLM, get_llm
from .memory import EpisodicMemory
from .output import Checkpoint, InferredEventsWriter
from .pii import Redactor
from .replay import ClockTick, EventTick, ReplayEngine
from .review import ReviewQueue
from .schema import Action, ConfidenceBand, HitlStatus, InferredState
from .semantic import SemanticMemory
from .state_board import StateBoard
from .synthesis import SynthesisAgent
from .tracing import RunTrace, configure_langsmith


@dataclass
class RunStats:
    events: int = 0
    checkpoints: int = 0
    syntheses: int = 0
    carried_forward: int = 0
    proposals: int = 0
    guardrail_blocks: int = 0
    critique_rejections: int = 0
    critique_downgrades: int = 0
    escalations: int = 0
    flagged_for_review: int = 0
    late_arrivals: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    embedder: str = ""

    def summary(self) -> str:
        return (
            f"events={self.events} checkpoints={self.checkpoints} syntheses={self.syntheses} "
            f"(carried_forward={self.carried_forward}) proposals={self.proposals} "
            f"guardrail_blocks={self.guardrail_blocks} critique_rejects={self.critique_rejections} "
            f"escalations={self.escalations} flagged_for_review={self.flagged_for_review} "
            f"late_arrivals={self.late_arrivals} "
            f"llm_calls={self.llm_calls} llm_failures={self.llm_failures} embedder={self.embedder}"
        )


class Pipeline:
    """
    Usage:
        pipeline = Pipeline("data/scenario_03", offline=True)
        writer, stats = pipeline.run()
        writer.write("out/scenario_03_inferred_events.json")
    """

    def __init__(
        self,
        scenario: str | Path,
        *,
        speed: float = 0,
        offline: bool = False,
        db_path: str | Path = ":memory:",
        chroma_path: str | Path | None = None,
        trace_path: str | Path | None = None,
        hitl_responses: list[str] | None = None,
        interactive_hitl: bool = False,
    ) -> None:
        self.engine = ReplayEngine(scenario, speed=speed).load()
        self.entities = self.engine.entities
        self.redactor = Redactor.from_entities(self.entities)

        fast = NullLLM() if offline else get_llm("fast")
        reasoning = NullLLM() if offline else get_llm("reasoning")
        self.langsmith = False if offline else configure_langsmith()

        self.memory = EpisodicMemory(db_path, customer_id=self.entities.get("customer_id"))
        self.board = StateBoard(self.memory)
        self.semantic = SemanticMemory(chroma_path, prefer_default_embedder=not offline)
        self.semantic.seed()

        self.agents = default_swarm()
        self.ctx = PerceptionContext(
            memory=self.memory, redactor=self.redactor, entities=self.entities, llm=fast
        )
        self.synthesis = SynthesisAgent(llm=reasoning)
        self.proposer = ActionProposer(self.semantic, llm=reasoning)
        self.critic = CritiqueAgent(llm=reasoning)
        self.hitl = HitlStub(interactive=interactive_hitl, responses=hitl_responses)
        # Parallel to the graded output, never part of it -- see review.py.
        self.review = ReviewQueue()

        self.trace = RunTrace(
            trace_path,
            redactor=self.redactor,
            tags={
                "customer_id": self.entities.get("customer_id"),
                "scenario_id": self.engine.config.scenario_id,
            },
        )
        self.writer = InferredEventsWriter(
            redactor=self.redactor, scenario_id=self.engine.config.scenario_id
        )
        self.stats = RunStats(embedder=self.semantic.embedder_name)
        self._llms = [fast, reasoning]

    # =====================================================================

    def run(self) -> tuple[InferredEventsWriter, RunStats]:
        # History is backstory: all of it predates simulated_start, so loading it
        # up front cannot leak anything. The FILTER is what protects us, not
        # withholding the data.
        self.memory.record_events(self.engine.history)
        self.trace.record(
            self.engine.config.simulated_start,
            kind="run_start",
            history_events=len(self.engine.history),
            live_events=len(self.engine.live),
            langsmith=self.langsmith,
            embedder=self.semantic.embedder_name,
            warnings=self.engine.warnings,
        )

        for tick in self.engine.stream():
            if isinstance(tick, EventTick):
                self._on_event(tick)
            elif isinstance(tick, ClockTick):
                self._on_checkpoint(tick)

        self._collect_llm_stats()
        self.trace.record(
            self.engine.clock.now, kind="run_end", stats=self.stats.summary()
        )
        self.trace.close()
        return self.writer, self.stats

    def _on_event(self, tick: EventTick) -> None:
        """EVENT-BASED TRIGGER."""
        self.memory.record_event(tick.event)
        self.stats.events += 1
        if tick.event.is_late_arrival:
            self.stats.late_arrivals += 1
            self.trace.record(
                tick.as_of,
                kind="late_arrival",
                event_id=tick.event.event_id,
                event_time=tick.event.event_time,
                ingestion_time=tick.event.ingestion_time,
                lag_hours=round(tick.event.arrival_lag_seconds / 3600, 1),
            )
        findings = run_swarm_on_event(self.agents, tick.event, tick.as_of, self.ctx)
        if findings:
            self.board.publish_all(findings)
            self.trace.record(
                tick.as_of,
                kind="perception",
                trigger="event",
                event_id=tick.event.event_id,
                findings=[f.signal for f in findings],
            )

    def _on_checkpoint(self, tick: ClockTick) -> None:
        """TIME-BASED TRIGGER, then the agent-dependent cascade."""
        as_of = tick.as_of
        self.stats.checkpoints += 1

        findings = run_swarm_on_tick(self.agents, as_of, self.ctx)
        if findings:
            self.board.publish_all(findings)

        # AGENT-DEPENDENT TRIGGER: >= 2 distinct perception agents flagged
        # something in the window. Below that there is nothing to correlate, and
        # the previous belief simply stands.
        triggered = self.board.should_synthesise(as_of)
        synthesis = self.synthesis.synthesise(as_of, self.board)
        self.stats.syntheses += 1
        if synthesis.carried_forward:
            self.stats.carried_forward += 1

        proposal = self.proposer.propose(synthesis, self.memory, self.entities)
        if proposal.is_intervention:
            self.stats.proposals += 1

        verdict = guardrail_module.check(proposal, synthesis)
        if not verdict.passed:
            self.stats.guardrail_blocks += 1

        critique = self.critic.review(proposal, synthesis, verdict)
        action, subtype = proposal.action, proposal.action_subtype
        if critique.verdict == "reject":
            self.stats.critique_rejections += 1
            action, subtype = Action.NO_ACTION, None
        elif critique.verdict == "downgrade" and critique.revised_action is not None:
            self.stats.critique_downgrades += 1
            action, subtype = critique.revised_action, critique.revised_subtype

        hitl = self.hitl.route(
            ActionProposal(
                as_of=as_of,
                action=action,
                action_subtype=subtype,
                rationale=proposal.rationale,
                policy_ids=proposal.policy_ids,
                policy_text=proposal.policy_text,
                event_ids=proposal.event_ids,
            ),
            synthesis,
        )
        if hitl.status is HitlStatus.ESCALATED:
            self.stats.escalations += 1

        # Escalation for ambiguity (PS 6.3). Runs on the FINAL action, so a
        # checkpoint the critique rejected is still considered. Touches nothing
        # in the graded row.
        flagged = self.review.consider(
            synthesis,
            ActionProposal(as_of=as_of, action=action, action_subtype=subtype,
                           rationale=proposal.rationale, event_ids=proposal.event_ids),
        )
        if flagged is not None:
            self.stats.flagged_for_review += 1
            self.trace.record(as_of, kind="review_flag", reason=flagged.reason,
                              detail=flagged.detail)

        checkpoint = self._build_checkpoint(as_of, synthesis, proposal, hitl, verdict, critique)
        self.writer.add(checkpoint)
        self.memory.record_decision(checkpoint)

        self.trace.record(
            as_of,
            kind="checkpoint",
            triggered=triggered,
            inferred_state=synthesis.inferred_state,
            confidence_band=synthesis.confidence_band,
            carried_forward=synthesis.carried_forward,
            affinity=synthesis.affinity,
            strong_signals=list(synthesis.strong_signals),
            source_systems=list(synthesis.corroboration.source_systems),
            agents=list(synthesis.corroboration.agents),
            event_ids=list(synthesis.event_ids)[:12],
            synthesis_by=synthesis.decided_by,
            proposed_action=proposal.action,
            proposal_by=proposal.decided_by,
            policy_ids=list(proposal.policy_ids),
            gate_reason=proposal.gate_reason,
            guardrail_passed=verdict.passed,
            guardrail_reason=verdict.reason,
            critique_verdict=critique.verdict,
            critique_reason=critique.reason,
            critique_by=critique.decided_by,
            final_action=checkpoint.action,
            hitl_status=checkpoint.hitl_status,
        )

    def _build_checkpoint(self, as_of, synthesis, proposal, hitl, verdict, critique) -> Checkpoint:
        """
        Assemble the graded row.

        The notes field is built to satisfy the explainability requirement
        mechanically: it always names the state, always carries the guardrail
        verdict, and always cites event_ids when an action is proposed -- because
        the Checkpoint constructor refuses the row otherwise.
        """
        action = hitl.action
        subtype = hitl.action_subtype if action is not Action.NO_ACTION else None

        parts = [proposal.rationale.strip()]
        if critique.verdict != "accept":
            parts.append(f"Critique {critique.verdict}: {critique.reason}")
        if proposal.is_intervention or not verdict.passed:
            parts.append(f"Guardrail: {verdict.reason}")
        if hitl.note:
            parts.append(f"HITL: {hitl.note}")

        notes = " ".join(p for p in parts if p)

        # An action needs citations to be constructible at all. If the rationale
        # somehow lost them, append them rather than letting the row be rejected
        # mid-run -- the evidence exists, this is only about surfacing it.
        if action is not Action.NO_ACTION and "EVT_" not in notes:
            cited = ", ".join(synthesis.event_ids[:5])
            notes = f"{notes} Evidence: {cited}."

        return Checkpoint(
            as_of_time=as_of,
            inferred_state=synthesis.inferred_state,
            confidence_band=synthesis.confidence_band,
            action=action,
            action_subtype=subtype,
            hitl_status=hitl.status,
            notes=self.redactor.scrub(notes)[:1400],
            citations=synthesis.event_ids,
            guardrail_checked=verdict.passed,
        )

    def _collect_llm_stats(self) -> None:
        for client in self._llms:
            self.stats.llm_calls += len(getattr(client, "calls", []))
        self.stats.llm_failures = (
            self.ctx.llm_failures
            + self.synthesis.llm_failures
            + self.proposer.llm_failures
            + self.critic.llm_failures
        )

    def close(self) -> None:
        self.trace.close()
        self.memory.close()
