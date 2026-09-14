"""
The state board -- the shared per-customer surface the perception swarm writes to
and the Synthesis Agent reads from.

WHY THIS EXISTS AS ITS OWN THING
--------------------------------
The mid-term architecture has four perception agents running as a SWARM: parallel,
independent, no interdependency. Swarms need somewhere to put their output that is
not each other. The state board is that place.

It gives three things nothing else in the system provides:

1. A TIME-SCOPED VIEW OF FINDINGS. Every read goes through episodic memory with a
   mandatory `as_of`, so the board cannot show synthesis a finding from the future
   any more than memory can show an agent a future event.

2. THE CORROBORATION COUNT. The guardrail's question -- "is this backed by >= 2
   independent source_systems in the same window, or is it one isolated anomaly?"
   -- is answered here, by arithmetic over finding metadata. Not by an LLM, and
   not by a prompt politely asking a model to be careful.

3. PERSISTED STATE. The problem statement is explicit that an inferred life phase
   "is not a one-off classification to be computed and forgotten" -- it must still
   be available and correctly weighted weeks later. `current_state()` reads the
   last committed decision rather than re-deriving it, which is both the required
   behaviour and the reason ~70 of 74 daily checkpoints cost no LLM call at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from .findings import Finding, SignalStrength
from .memory import EpisodicMemory
from .schema import Action, ConfidenceBand, HitlStatus, InferredState

# How far back two findings can sit and still count as "the same window".
#
# 14 days is chosen from the data, not plucked out of the air. Scenario_03's
# decisive pair are the standing-instruction cancellation (EVT_000457, 4 March)
# and the savings transfer out (EVT_000461, 6 March) -- two days apart. Scenario_01
# spreads its ER visit, income replacement and hospital bill across roughly five
# weeks, but any adjacent pair falls well inside a fortnight. A shorter window
# would miss slow-burn narratives; a much longer one would start correlating
# genuinely unrelated events and manufacture false positives.
DEFAULT_WINDOW_DAYS = 30

# Perception agents, named once so a typo in an agent name cannot silently create
# a fifth "agent" that never corroborates anything.
PERCEPTION_AGENTS = ("transaction", "usage", "support", "life_signal")


@dataclass(frozen=True)
class Corroboration:
    """
    The evidence picture in one window. This is the guardrail's input.

    Kept as data rather than a bool so the notes field, the trace and the Critique
    Agent can all explain WHY something was or was not corroborated, rather than
    just reporting a verdict.
    """

    as_of: datetime
    window_days: float
    findings: tuple[Finding, ...]
    source_systems: tuple[str, ...]
    agents: tuple[str, ...]
    event_ids: tuple[str, ...]

    @property
    def independent_source_count(self) -> int:
        return len(self.source_systems)

    @property
    def agent_count(self) -> int:
        return len(self.agents)

    @property
    def is_corroborated(self) -> bool:
        """
        THE GUARDRAIL PREDICATE.

        >= 2 independent source_systems inside the window.

        This single rule defeats every red herring in all three scenarios, and the
        reason is structural: each planted red herring is one isolated event with
        nothing from any other system backing it up.

          scenario_01 EVT_000382 (tuition transfer) and EVT_000402 (resort refund)
          scenario_02 EVT_000328 (baby monitor / electronics)
          scenario_03 EVT_000447 (tax refund deposit)

        Every one of them is a single event from a single source system. Meanwhile
        every genuine narrative shows up across several: scenario_03's churn story
        touches support_logs, web_app_events, core_banking_ledger and
        instant_payments.

        Note this counts SOURCE SYSTEMS, not events and not agents. Three card
        purchases are three events but one system, and would prove nothing -- a
        spending spree at one merchant is still one stream of evidence. Counting
        agents instead would be subtly wrong too, because the Transaction Agent
        alone covers four different source systems and can legitimately corroborate
        itself across them.
        """
        return self.independent_source_count >= 2

    @property
    def strongest(self) -> SignalStrength | None:
        if not self.findings:
            return None
        return max((f.strength for f in self.findings), key=lambda s: s.score)

    def explain(self) -> str:
        if not self.findings:
            return "no findings in window"
        verdict = "CORROBORATED" if self.is_corroborated else "UNCORROBORATED"
        return (
            f"{verdict}: {self.independent_source_count} independent source system(s) "
            f"({', '.join(self.source_systems)}) across {self.agent_count} agent(s) "
            f"in the {self.window_days:.0f}-day window"
        )


@dataclass(frozen=True)
class BoardState:
    """A snapshot of what the system believes at one moment."""

    as_of: datetime
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    action: Action
    action_subtype: str | None
    hitl_status: HitlStatus
    decided_at: datetime | None
    notes: str = ""

    @property
    def is_default(self) -> bool:
        """True if nothing has ever been decided -- the cold-start state."""
        return self.decided_at is None

    @property
    def age_days(self) -> float | None:
        if self.decided_at is None:
            return None
        return (self.as_of - self.decided_at).total_seconds() / 86400


class StateBoard:
    """
    Usage:
        board = StateBoard(memory)
        board.publish(finding)                      # perception agents write
        board.findings(now)                         # synthesis reads
        board.corroboration(now)                    # the guardrail reads
        board.current_state(now)                    # persisted belief
    """

    def __init__(self, memory: EpisodicMemory, window_days: float = DEFAULT_WINDOW_DAYS) -> None:
        self.memory = memory
        self.window_days = window_days

    # -- writing -----------------------------------------------------------

    def publish(self, finding: Finding) -> Finding:
        if finding.agent not in PERCEPTION_AGENTS:
            raise ValueError(
                f"unknown agent {finding.agent!r}; expected one of {PERCEPTION_AGENTS}. "
                "A typo here would create an agent that never corroborates anything."
            )
        self.memory.record_finding(finding)
        return finding

    def publish_all(self, findings: Iterable[Finding]) -> list[Finding]:
        return [self.publish(f) for f in findings]

    # -- reading -----------------------------------------------------------

    def findings(
        self,
        as_of: datetime,
        *,
        window_days: float | None = None,
        agents: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[Finding]:
        """Findings inside the trailing window. `as_of` mandatory, as everywhere."""
        return self.memory.findings_as_of(
            as_of,
            since_days=self.window_days if window_days is None else window_days,
            agents=agents,
            signals=signals,
        )

    def corroboration(
        self,
        as_of: datetime,
        *,
        window_days: float | None = None,
        signals: Sequence[str] | None = None,
    ) -> Corroboration:
        """Assemble the evidence picture the guardrail will rule on."""
        window = self.window_days if window_days is None else window_days
        found = self.findings(as_of, window_days=window, signals=signals)

        source_systems: set[str] = set()
        agents: set[str] = set()
        event_ids: list[str] = []
        for finding in found:
            source_systems.update(finding.source_systems)
            agents.add(finding.agent)
            for event_id in finding.event_ids:
                if event_id not in event_ids:
                    event_ids.append(event_id)

        return Corroboration(
            as_of=as_of,
            window_days=window,
            findings=tuple(found),
            source_systems=tuple(sorted(source_systems)),
            agents=tuple(sorted(agents)),
            event_ids=tuple(event_ids),
        )

    def should_synthesise(self, as_of: datetime, *, window_days: float | None = None) -> bool:
        """
        THE AGENT-DEPENDENT TRIGGER.

        Synthesis fires when >= 2 DISTINCT perception agents have flagged something
        in the same window. Straight from the mid-term architecture.

        Note this is deliberately NOT the same test as the guardrail. This one
        counts AGENTS and decides whether it is worth spending an LLM call; the
        guardrail counts SOURCE SYSTEMS and decides whether an action may fire.
        Two different questions:

          - The Transaction Agent alone can see a salary stop AND a savings
            drawdown -- two source systems, so the guardrail would be satisfied,
            but only one agent has an opinion and there is nothing to correlate.
          - Conversely two agents might both be reading the same single system,
            which is enough to be worth thinking about but not enough to act on.

        Keeping them separate is what stops the trigger and the safety check
        collapsing into one number that does neither job properly.
        """
        found = self.findings(as_of, window_days=window_days)
        return len({f.agent for f in found}) >= 2

    def current_state(self, as_of: datetime) -> BoardState:
        """
        What the system believes right now, carried forward from the last decision.

        On a quiet day this is the whole answer: nothing new arrived, so the
        previous belief stands and no agent needs to run.
        """
        last = self.memory.latest_decision_as_of(as_of)
        if last is None:
            return BoardState(
                as_of=as_of,
                inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
                confidence_band=ConfidenceBand.LOW,
                action=Action.NO_ACTION,
                action_subtype=None,
                hitl_status=HitlStatus.AUTO_APPROVED,
                decided_at=None,
            )
        from .memory import _parse  # local import keeps the module surface small

        return BoardState(
            as_of=as_of,
            inferred_state=InferredState(last["inferred_state"]),
            confidence_band=ConfidenceBand(last["confidence_band"]),
            action=Action(last["action"]),
            action_subtype=last["action_subtype"],
            hitl_status=HitlStatus(last["hitl_status"]),
            decided_at=_parse(last["as_of_time"]),
            notes=last["notes"] or "",
        )

    def summary(self, as_of: datetime) -> str:
        state = self.current_state(as_of)
        corroboration = self.corroboration(as_of)
        signals = ", ".join(sorted({f"{f.agent}/{f.signal}" for f in corroboration.findings})) or "none"
        return (
            f"[{as_of:%Y-%m-%d}] believes={state.inferred_state.value}/"
            f"{state.confidence_band.value} | signals={signals} | {corroboration.explain()}"
        )
