"""
Findings -- what a perception agent publishes to the state board.

WHY A FINDING IS NOT A CHAIN OF THOUGHT
---------------------------------------
The mid-term architecture says the perception swarm publishes "structured
findings (not raw chain-of-thought)" to a shared state board. That constraint is
doing real work, and it is worth being able to defend:

1. The Synthesis Agent reads the board, not the raw events. If perception
   published prose, synthesis would be re-reading four essays and re-deriving the
   same conclusions -- which is the "single agent with a vector database" pattern
   the Architectural Novelty criterion explicitly marks down.

2. The GUARDRAIL needs to count independent source systems. That is arithmetic
   over `source_systems`, not text analysis. If a finding were prose, the
   corroboration rule would have to be an LLM judgement, and the problem
   statement requires guardrails to be actual code checks.

3. `event_ids` is what makes the graded `notes` field citable. Explainability
   stops being something we ask an LLM to remember and becomes something the
   data structure carries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalStrength(str, Enum):
    """
    How loudly one agent is shouting.

    Deliberately three coarse bands rather than a float. A perception agent
    saying "0.73 confident" implies a precision it does not have -- these are
    heuristics over a few dozen events, not calibrated probabilities. Three bands
    also map cleanly onto the three confidence_bands the output schema requires.
    """

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

    @property
    def score(self) -> int:
        return {"weak": 1, "moderate": 2, "strong": 3}[self.value]


@dataclass(frozen=True)
class Finding:
    """
    One observation by one perception agent at one moment in simulated time.

    Immutable, like Event, and for the same reason: findings are read by the
    Synthesis Agent, the Critique Agent and the guardrail. If any of them could
    mutate one, a bug there would silently change what the others see.
    """

    agent: str
    as_of: datetime
    signal: str
    strength: SignalStrength
    event_ids: tuple[str, ...] = ()
    source_systems: tuple[str, ...] = ()
    detail: str = ""
    window_start: datetime | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("Finding.as_of must be timezone-aware")
        if isinstance(self.strength, str):
            object.__setattr__(self, "strength", SignalStrength(self.strength))
        object.__setattr__(self, "event_ids", tuple(self.event_ids))
        # Sorted + de-duplicated so the guardrail's "how many independent
        # systems?" count can never be inflated by the same system listed twice.
        object.__setattr__(self, "source_systems", tuple(sorted(set(self.source_systems))))

    @property
    def independent_source_count(self) -> int:
        return len(self.source_systems)

    def to_row(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "as_of": self.as_of.isoformat(),
            "signal": self.signal,
            "strength": self.strength.value,
            "event_ids": json.dumps(list(self.event_ids)),
            "source_systems": json.dumps(list(self.source_systems)),
            "detail": self.detail,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "metrics": json.dumps(self.metrics, default=str),
        }

    def __repr__(self) -> str:
        return (
            f"Finding({self.agent}/{self.signal} {self.strength.value} "
            f"@{self.as_of:%Y-%m-%d} <- {len(self.event_ids)} event(s))"
        )
