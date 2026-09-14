"""
Support Agent -- support_logs only.

The most language-heavy agent, and the one where an LLM genuinely earns its place.
A support ticket is free prose written by a person; `resolution_status` tells you
the outcome but nothing about what the customer was actually going through.

Two things this agent looks for:

  1. A STRUCTURAL OUTCOME. `resolution_status == "human_rejected"` means the bank
     told the customer no. That is a fact, read from a field, no interpretation
     needed -- and in scenario_03 it is the catalyst for everything that follows.

  2. WHAT THE TEXT MEANS. "I've been in the hospital and my income dropped, can I
     set up a payment plan" is a hardship disclosure. "Just confirming it's me, I
     bought a video baby monitor" is reassurance -- and, quietly, a new-child
     signal. Keywords get both of those right on the practice data and would be
     fragile on hidden data, so the LLM leads and keywords catch.

A NOTE ON WHO WROTE THE TEXT
----------------------------
`raw_text` in a resolved ticket is sometimes the BANK's reply, not the customer's
(scenario_03's EVT_000412 is the bank refusing a fee waiver). Classifying that as
"the customer sounds calm" would be exactly backwards. The prompt says who is
speaking, and the structural check on resolution_status does not depend on the
text at all.
"""

from __future__ import annotations

from datetime import datetime

from ..findings import Finding, SignalStrength
from ..llm import LLMUnavailable, complete_json
from .. import prompts
from .base import PerceptionAgent, PerceptionContext, cited

SUPPORT = "support_logs"

# Outcomes that mean the customer asked for something and was refused.
REJECTED_STATUSES = {"human_rejected", "rejected", "denied", "declined"}
OPEN_STATUSES = {"open", "pending", "escalated", "unresolved"}

# Ticket categories that are grievances rather than queries.
GRIEVANCE_CATEGORIES = {"dispute", "complaint", "fee_dispute", "chargeback"}
HARDSHIP_CATEGORIES = {"payment_arrangements", "hardship", "collections", "forbearance"}

HARDSHIP_KEYWORDS = (
    "payment plan", "hardship", "cannot pay", "can't pay", "income dropped", "lost my job",
    "hospital", "medical", "struggling", "defer", "behind on",
)
CHILD_KEYWORDS = ("baby", "child", "newborn", "maternity", "pregnan", "daycare", "nursery")
LEAVING_KEYWORDS = ("close my account", "switch", "competitor", "leaving", "take my business")


THEME_SIGNAL = {
    "financial_hardship": ("hardship_contact", SignalStrength.STRONG),
    "medical_hardship": ("hardship_contact", SignalStrength.STRONG),
    "new_child": ("reassurance_contact", SignalStrength.MODERATE),
    "leaving_intent": ("complaint_open", SignalStrength.STRONG),
    "service_grievance": ("complaint_open", SignalStrength.MODERATE),
    "fraud_concern": ("complaint_open", SignalStrength.MODERATE),
    "routine_query": ("reassurance_contact", SignalStrength.WEAK),
}


class SupportAgent(PerceptionAgent):
    name = "support"
    source_systems = (SUPPORT,)

    def on_event(self, event, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        if event.source_system != SUPPORT:
            return []

        findings: list[Finding] = []
        payload = event.payload
        status = (payload.get("resolution_status") or "").lower()
        category = (payload.get("category") or "").lower()

        # --- structural, no interpretation needed --------------------------
        if status in REJECTED_STATUSES:
            # scenario_03 EVT_000412. A tenured, high-value customer asks for a
            # $35 fee waiver and is refused. Small in money, large in meaning:
            # every later churn behaviour dates from this.
            findings.append(
                self.finding(
                    as_of,
                    "complaint_denied",
                    SignalStrength.STRONG if category in GRIEVANCE_CATEGORIES else SignalStrength.MODERATE,
                    [event],
                    detail=(
                        f"A {category or 'support'} request was refused (resolution_status="
                        f"{status}) -- the customer asked and was told no ({cited([event])})"
                    ),
                    category=category,
                    resolution_status=status,
                )
            )
        elif status in OPEN_STATUSES and category in GRIEVANCE_CATEGORIES:
            findings.append(
                self.finding(
                    as_of,
                    "complaint_open",
                    SignalStrength.MODERATE,
                    [event],
                    detail=f"Unresolved {category} ticket opened ({cited([event])})",
                    category=category,
                )
            )

        if category in HARDSHIP_CATEGORIES:
            findings.append(
                self.finding(
                    as_of,
                    "hardship_contact",
                    SignalStrength.STRONG,
                    [event],
                    detail=(
                        f"Customer opened a {category} case -- an explicit request for relief "
                        f"({cited([event])})"
                    ),
                    category=category,
                )
            )

        # --- what the words mean -------------------------------------------
        findings.extend(self._classify_text(event, as_of, ctx))

        # De-duplicate: the category check and the text check can both land on
        # hardship_contact for the same ticket, and one ticket should not count
        # twice towards a confidence score.
        return _dedupe(findings)

    def _classify_text(self, event, as_of, ctx) -> list[Finding]:
        raw = event.payload.get("raw_text") or ""
        if not raw.strip():
            return []
        safe = ctx.redactor.scrub(raw)
        # A resolved ticket's text is usually the bank replying; a created one is
        # the customer writing in.
        author = "BANK" if event.event_type == "ticket_resolved" else "CUSTOMER"

        theme, distress, reason, via = self._llm_theme(safe, author, ctx)
        if theme is None:
            theme, distress, reason, via = (*self._keyword_theme(safe), "keywords")

        if theme in (None, "none"):
            return []

        signal, strength = THEME_SIGNAL.get(theme, ("reassurance_contact", SignalStrength.WEAK))
        if distress == "high" and strength is not SignalStrength.STRONG:
            strength = SignalStrength.STRONG

        return [
            self.finding(
                as_of,
                signal,
                strength,
                [event],
                detail=(
                    f"Support text ({author.lower()}-authored) reads as "
                    f"{theme.replace('_', ' ')}, distress={distress} [{via}]: {reason} "
                    f"({cited([event])})"
                ),
                theme=theme,
                distress=distress,
                classified_by=via,
            )
        ]

    def _llm_theme(self, text: str, author: str, ctx: PerceptionContext):
        try:
            result = complete_json(
                ctx.llm,
                prompts.TICKET_THEME,
                f"Written by: {author}\nText: {text!r}",
                role="support.ticket_theme",
            )
            theme = str(result.get("theme", "none")).strip().lower()
            if theme not in {*THEME_SIGNAL, "none"}:
                return None, None, None, None
            return (
                theme,
                str(result.get("customer_distress", "low")).lower(),
                str(result.get("reason", ""))[:90],
                f"llm:{getattr(ctx.llm, 'model', '?')}",
            )
        except (LLMUnavailable, ValueError, KeyError, TypeError):
            ctx.note_llm_failure()
            return None, None, None, None

    @staticmethod
    def _keyword_theme(text: str):
        lowered = text.lower()
        if any(k in lowered for k in LEAVING_KEYWORDS):
            return "leaving_intent", "medium", "leaving keywords"
        if any(k in lowered for k in HARDSHIP_KEYWORDS):
            medical = any(k in lowered for k in ("hospital", "medical", "surgery", "health"))
            return ("medical_hardship" if medical else "financial_hardship"), "high", "hardship keywords"
        if any(k in lowered for k in CHILD_KEYWORDS):
            return "new_child", "none", "child keywords"
        return "none", "none", "no keyword match"


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Keep the strongest finding per signal so one ticket counts once."""
    best: dict[str, Finding] = {}
    for finding in findings:
        current = best.get(finding.signal)
        if current is None or finding.strength.score > current.strength.score:
            best[finding.signal] = finding
    return list(best.values())
