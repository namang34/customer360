"""
Usage Agent -- web_app_events only.

The narrowest agent and, in scenario_03, the one that sees the story first. Two
kinds of thing live here:

  - A CLICK that means something on its own. Opening
    "manage_standing_instructions_cancel" is not ambiguous: nobody visits that
    page by accident. Event-based, fires immediately.

  - A HABIT CHANGING. Logins thinning out, sessions collapsing from ~5 minutes to
    18 seconds. No single login event says "this customer is disengaging"; the
    signal only exists in the rate. Time-based, fires on the daily tick.

This is also the one agent whose free text is worth an LLM. A search query like
"child education savings plan" or "medical hardship plan" is language, and
guessing its intent from keywords works for the three practice scenarios but
would be brittle on hidden data. So: LLM when available, keyword rules when not,
and the fallback is tested rather than assumed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..findings import Finding, SignalStrength
from ..llm import LLMUnavailable, complete_json
from .. import prompts
from .base import PerceptionAgent, PerceptionContext, cited

WEB = "web_app_events"

# Pages that indicate intent to reduce or end the relationship.
CANCELLATION_PAGES = (
    "manage_standing_instructions_cancel",
    "close_account",
    "cancel_",
    "account_closure",
    "stop_payment",
)

ENGAGEMENT_DROP_RATIO = 0.4     # recent logins vs the customer's own baseline
SHORT_SESSION_SECONDS = 45      # scenario_03 collapses to 18s and 10s
SESSION_COLLAPSE_RATIO = 0.35

# Keyword fallback for search intent. Used only when no LLM is configured.
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "financial_hardship": ("hardship", "payment plan", "defer", "forbearance", "cannot pay", "relief"),
    "child_planning": ("child", "education savings", "daycare", "childcare", "college fund", "baby"),
    "leaving": ("close account", "transfer out", "switch bank", "cancel", "fees", "competitor"),
    "medical": ("medical hardship", "medical", "hospital", "insurance claim", "health"),
}

INTENT_STRENGTH = {
    "financial_hardship": SignalStrength.STRONG,
    "child_planning": SignalStrength.MODERATE,
    "leaving": SignalStrength.STRONG,
    "medical": SignalStrength.MODERATE,
    "none": SignalStrength.WEAK,
}



class UsageAgent(PerceptionAgent):
    name = "usage"
    source_systems = (WEB,)

    # =====================================================================
    # EVENT-BASED
    # =====================================================================

    def on_event(self, event, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        if event.source_system != WEB:
            return []
        page = (event.payload.get("feature_or_page") or "").lower()

        if event.event_type == "feature_used" and any(p in page for p in CANCELLATION_PAGES):
            # scenario_03 EVT_000457. A single click, unambiguous, and two days
            # before the money moves -- this is where the lead time comes from.
            return [
                self.finding(
                    as_of,
                    "cancellation_feature_used",
                    SignalStrength.STRONG,
                    [event],
                    detail=(
                        f"Customer opened '{page}' -- an explicit step towards reducing or ending "
                        f"the relationship ({cited([event])})"
                    ),
                    page=page,
                )
            ]

        if event.event_type == "search_query":
            return self._classify_search(event, as_of, ctx)

        return []

    def _classify_search(self, event, as_of, ctx) -> list[Finding]:
        raw = event.payload.get("search_text") or ""
        if not raw.strip():
            return []
        # Scrubbed before it reaches the model. Graded requirement, and free text
        # is the most likely place for a name to appear.
        safe = ctx.redactor.scrub(raw)

        intent, confidence, reason, via = self._llm_intent(safe, ctx)
        if intent is None:
            intent, confidence, reason, via = (*self._keyword_intent(safe), "keywords")

        if intent in (None, "none"):
            return []

        strength = INTENT_STRENGTH.get(intent, SignalStrength.MODERATE)
        if confidence == "low" and strength is SignalStrength.STRONG:
            strength = SignalStrength.MODERATE

        return [
            self.finding(
                as_of,
                "search_intent",
                strength,
                [event],
                detail=(
                    f"In-app search '{safe}' reads as {intent.replace('_', ' ')} intent "
                    f"[{via}]: {reason} ({cited([event])})"
                ),
                intent=intent,
                query=safe,
                classified_by=via,
            )
        ]

    def _llm_intent(self, text: str, ctx: PerceptionContext):
        if isinstance(ctx.llm, type(None)):
            return None, None, None, None
        try:
            result = complete_json(
                ctx.llm, prompts.SEARCH_INTENT, f"Search query: {text!r}", role="usage.search_intent"
            )
            intent = str(result.get("intent", "none")).strip().lower()
            if intent not in {*INTENT_KEYWORDS, "retirement", "home_purchase", "none"}:
                return None, None, None, None
            return (
                intent,
                str(result.get("confidence", "medium")).lower(),
                str(result.get("reason", ""))[:80],
                f"llm:{getattr(ctx.llm, 'model', '?')}",
            )
        except (LLMUnavailable, ValueError, KeyError, TypeError):
            ctx.note_llm_failure()
            return None, None, None, None

    @staticmethod
    def _keyword_intent(text: str):
        """
        Longest matched phrase wins, not first-dict-entry wins.

        "medical hardship plan" matches both "hardship" (financial_hardship) and
        "medical hardship" (medical). Iterating in dict order picked whichever was
        declared first, which sent scenario_01's hardship search to
        financial_distress_general and cost the medical_hardship inference at the
        26 March checkpoint. Preferring the longer phrase is a real specificity
        rule rather than an accident of ordering.
        """
        lowered = text.lower()
        best_intent, best_len, best_kw = "none", 0, ""
        for intent, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered and len(keyword) > best_len:
                    best_intent, best_len, best_kw = intent, len(keyword), keyword
        if best_intent == "none":
            return "none", "low", "no keyword match"
        return best_intent, "medium", f"matched {best_kw!r}"

    # =====================================================================
    # TIME-BASED
    # =====================================================================

    def on_tick(self, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._engagement_drop(as_of, ctx))
        findings.extend(self._session_collapse(as_of, ctx))
        return findings

    def _engagement_drop(self, as_of, ctx) -> list[Finding]:
        """
        Logins thinning out against this customer's own normal.

        Per-customer baselines matter here more than anywhere else. scenario_03's
        David Chen logs in roughly daily; scenario_01's Marcus Vance far less. A
        fixed threshold like "fewer than 3 logins a week" would fire constantly
        for one and never for the other.
        """
        recent = ctx.memory.daily_rate(
            as_of, source_systems=[WEB], event_types=["login"], window_days=14
        )
        baseline = ctx.memory.baseline_daily_rate(
            as_of,
            source_systems=[WEB],
            event_types=["login"],
            baseline_days=90,
            exclude_recent_days=14,
        )
        if baseline < 0.15 or recent > baseline * ENGAGEMENT_DROP_RATIO:
            return []

        evidence = ctx.memory.events_as_of(
            as_of, since_days=30, source_systems=[WEB], event_types=["login"], limit=3
        )
        return [
            self.finding(
                as_of,
                "engagement_drop",
                SignalStrength.STRONG if recent == 0 else SignalStrength.MODERATE,
                evidence,
                source_systems=[WEB],
                detail=(
                    f"App logins fell to {recent:.2f}/day from a baseline of {baseline:.2f}/day "
                    f"({1 - recent / baseline:.0%} lower)"
                ),
                recent_rate=round(recent, 3),
                baseline_rate=round(baseline, 3),
            )
        ]

    def _session_collapse(self, as_of, ctx) -> list[Finding]:
        """
        Sessions getting drastically shorter -- logging in, glancing, leaving.

        A distinct signal from logins drying up, and it often comes FIRST: the
        customer is still checking, but has stopped doing anything. scenario_03
        goes from ~230s sessions to 18s and then 10s while still logging in.
        """
        sessions = [
            e
            for e in ctx.memory.events_as_of(
                as_of, since_days=21, source_systems=[WEB], event_types=["session_duration"]
            )
        ]
        history = [
            e
            for e in ctx.memory.events_as_of(
                as_of - _days(21), since_days=90, source_systems=[WEB], event_types=["session_duration"]
            )
        ]
        if len(sessions) < 2 or len(history) < 5:
            return []

        recent_avg = sum(_seconds(e) for e in sessions) / len(sessions)
        baseline_avg = sum(_seconds(e) for e in history) / len(history)
        if baseline_avg <= 0:
            return []
        if recent_avg > baseline_avg * SESSION_COLLAPSE_RATIO and recent_avg > SHORT_SESSION_SECONDS:
            return []

        return [
            self.finding(
                as_of,
                "session_length_collapse",
                SignalStrength.MODERATE,
                sessions[-3:],
                source_systems=[WEB],
                detail=(
                    f"Average session length collapsed to {recent_avg:.0f}s from a baseline of "
                    f"{baseline_avg:.0f}s -- logging in but no longer engaging "
                    f"({cited(sessions[-3:])})"
                ),
                recent_seconds=round(recent_avg, 1),
                baseline_seconds=round(baseline_avg, 1),
            )
        ]


def _seconds(event) -> float:
    try:
        return float(event.payload.get("session_length_sec") or 0)
    except (TypeError, ValueError):
        return 0.0


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)
