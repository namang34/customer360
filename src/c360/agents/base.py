"""
The perception agent base class and the shared signal vocabulary.

THE SWARM CONTRACT
------------------
Four agents, disjoint source systems, zero interdependency. None of them may read
another's findings -- that is what makes this a swarm rather than a pipeline, and
it is why they can run in any order (or in parallel) without changing the result.
A test asserts that property directly.

Each agent exposes two entry points, matching the two trigger types in the
mid-term architecture:

    on_event(event, ctx)  EVENT-BASED. Fires the instant a relevant event
                          arrives. Used for things that are true the moment you
                          see them: a cancellation click, a denied complaint, a
                          KYC dependents change, a large self-transfer.

    on_tick(as_of, ctx)   TIME-BASED. Fires on the daily clock tick. Used for
                          things no single event can tell you: a spend rate
                          collapsing, logins drying up, a standing instruction
                          that FAILED TO HAPPEN.

The second one is the interesting half. An absence produces no event, so an
event-only system is structurally blind to it -- and in scenario_03 the absence
IS the story.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from ..findings import Finding, SignalStrength
from ..llm import LLM, NullLLM
from ..memory import EpisodicMemory
from ..pii import Redactor

# The full signal vocabulary, declared in one place.
#
# A closed vocabulary rather than free-form strings: the Synthesis Agent maps
# signals to inferred_state values, and a typo'd signal would silently never
# match anything. A test asserts every emitted signal is in this set.
SIGNALS = frozenset({
    # Transaction Agent
    "income_disruption",
    "income_replacement",
    "major_medical_expense",
    "healthcare_spend",
    "savings_drawdown",
    "large_outbound_transfer",
    "self_transfer_external",
    "salary_swept_out",
    "standing_instruction_stopped",
    "new_recurring_commitment",
    "card_spend_collapse",
    "baby_retail_spend",
    "large_inbound_deposit",
    "unusual_refund",
    # Usage Agent
    "engagement_drop",
    "session_length_collapse",
    "cancellation_feature_used",
    "search_intent",
    # Support Agent
    "complaint_denied",
    "complaint_open",
    "hardship_contact",
    "reassurance_contact",
    # Life-Signal Agent
    "dependents_increase",
    "marital_status_change",
    "address_change",
    "loan_application",
    "social_life_event",
})


@dataclass
class PerceptionContext:
    """
    Everything an agent is allowed to touch.

    Note what is NOT here: the state board's findings. An agent gets memory (raw
    events, time-scoped) but never another agent's conclusions. Passing the board
    in would make it trivially easy to write an agent that reads a peer's finding
    and "corroborates" it -- which would quietly turn two independent witnesses
    into one witness counted twice, and defeat the guardrail.
    """

    memory: EpisodicMemory
    redactor: Redactor
    entities: dict[str, Any] = field(default_factory=dict)
    llm: LLM = field(default_factory=NullLLM)
    llm_failures: int = 0

    @property
    def accounts(self) -> dict[str, str]:
        """account_id -> account type, e.g. {'ACC_SAV_003': 'savings'}."""
        return {
            a["account_id"]: a.get("type", "")
            for a in self.entities.get("accounts", [])
            if a.get("account_id")
        }

    def account_type(self, account_id: str | None) -> str:
        return self.accounts.get(account_id or "", "")

    def note_llm_failure(self) -> None:
        self.llm_failures += 1


class PerceptionAgent(ABC):
    """Base class. Subclasses set `name` and `source_systems`."""

    name: str = "base"
    source_systems: tuple[str, ...] = ()

    def handles(self, event) -> bool:
        return event.source_system in self.source_systems

    def on_event(self, event, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        """Event-based trigger. Default: nothing."""
        return []

    def on_tick(self, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        """Time-based trigger. Default: nothing."""
        return []

    # -- helpers ----------------------------------------------------------

    def finding(
        self,
        as_of: datetime,
        signal: str,
        strength: SignalStrength,
        events: Sequence[Any] | None = None,
        *,
        detail: str = "",
        source_systems: Sequence[str] | None = None,
        **metrics: Any,
    ) -> Finding:
        """
        Build a Finding, deriving event_ids and source_systems from the evidence.

        Deriving rather than accepting them separately is deliberate: it makes it
        impossible to claim corroboration from a source system that contributed no
        actual event. The guardrail's arithmetic is only as honest as this list.
        """
        events = list(events or [])
        if signal not in SIGNALS:
            raise ValueError(f"{self.name}: unknown signal {signal!r} -- add it to SIGNALS")
        return Finding(
            agent=self.name,
            as_of=as_of,
            signal=signal,
            strength=strength,
            event_ids=tuple(e.event_id for e in events),
            source_systems=tuple(source_systems or {e.source_system for e in events}),
            detail=detail,
            metrics=metrics,
        )


def cited(events: Sequence[Any], limit: int = 3) -> str:
    """Render event ids for a detail string: 'EVT_000457, EVT_000461 (+2 more)'."""
    ids = [e.event_id for e in events]
    shown = ", ".join(ids[:limit])
    return shown + (f" (+{len(ids) - limit} more)" if len(ids) > limit else "")
