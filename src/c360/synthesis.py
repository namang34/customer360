"""
Synthesis Agent -- correlation layer.

Reads the STATE BOARD, never raw events. That separation is the point of the
architecture: four perception agents have already turned ~490 raw events into a
handful of structured findings, and synthesis' job is to decide what story those
findings add up to. If it re-read the events it would be re-doing perception's
work, and the swarm would be decoration.

TWO DECISIONS, MADE SEPARATELY
------------------------------
    WHAT is happening   -> inferred_state   (which narrative fits the evidence)
    HOW SURE are we     -> confidence_band  (is the evidence strong and broad)

Keeping them apart matters. Scenario_03's ground truth expects `churn_risk` with
LOW confidence on 15 February and the SAME state with HIGH confidence on 8 March.
The story did not change; the weight of evidence did. A system that fused the two
would have to either name the state late (losing the early checkpoint) or claim
confidence early (failing the "do not overreact" test).

WHERE THE LLM SITS
------------------
Affinity scoring and the confidence bands are code. The LLM is used to ADJUDICATE
between the top candidate states when they are close, and to write the rationale.
That is a deliberate division:

  - "Which of these two narratives better explains a hospital bill, a benefits
    credit and a hardship search?" is a judgement, and judgement is what a
    language model is for.
  - "Are there at least two STRONG findings across at least three source systems?"
    is arithmetic, and arithmetic in a prompt is how you get a confidently wrong
    answer you cannot audit.

When no LLM is available the deterministic winner stands, and every test in the
suite runs that path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import prompts
from .findings import Finding, SignalStrength
from .llm import LLM, LLMUnavailable, NullLLM, complete_json
from .schema import ConfidenceBand, InferredState
from .state_board import Corroboration, StateBoard

# ---------------------------------------------------------------------------
# Signal -> state affinity
# ---------------------------------------------------------------------------
# Weights are per (state, signal). A finding contributes
# strength.score * weight to that state's total.
#
# Several signals deliberately point at more than one state -- income_disruption
# is equally consistent with maternity leave, job loss and illness. That ambiguity
# is real, and resolving it is exactly the job of the correlating layer: it is the
# OTHER signals in the window that disambiguate.

STATE_AFFINITY: dict[InferredState, dict[str, float]] = {
    InferredState.MEDICAL_HARDSHIP: {
        "major_medical_expense": 4,
        "healthcare_spend": 3,
        "income_replacement": 2,
        "hardship_contact": 2,
        "savings_drawdown": 1,
    },
    InferredState.NEW_CHILD_LIFE_EVENT: {
        "dependents_increase": 5,
        "baby_retail_spend": 4,
        "new_recurring_commitment": 2,
        "reassurance_contact": 1,
        "income_disruption": 1,
        "social_life_event": 2,
    },
    InferredState.CHURN_RISK: {
        "cancellation_feature_used": 4,
        "self_transfer_external": 4,
        "salary_swept_out": 3,
        "standing_instruction_stopped": 3,
        "complaint_denied": 2,
        "complaint_open": 1,
        "engagement_drop": 2,
        "session_length_collapse": 1,
        "card_spend_collapse": 2,
    },
    InferredState.JOB_LOSS_OR_INCOME_DISRUPTION: {
        "income_disruption": 3,
        "income_replacement": 2,
        "savings_drawdown": 1,
    },
    InferredState.FINANCIAL_DISTRESS_GENERAL: {
        "savings_drawdown": 2,
        "income_disruption": 2,
        "hardship_contact": 2,
        "card_spend_collapse": 1,
    },
    InferredState.WEALTH_GROWTH_OR_WINDFALL: {
        "large_inbound_deposit": 3,
    },
    InferredState.MARRIAGE_OR_RELATIONSHIP_CHANGE: {
        "marital_status_change": 5,
        "social_life_event": 1,
    },
    InferredState.RELOCATION: {
        "address_change": 4,
        "social_life_event": 1,
    },
}

# search_intent and support themes carry their meaning in metrics rather than in
# the signal name, so they are routed separately.
INTENT_AFFINITY: dict[str, tuple[InferredState, float]] = {
    "financial_hardship": (InferredState.FINANCIAL_DISTRESS_GENERAL, 3),
    "medical": (InferredState.MEDICAL_HARDSHIP, 3),
    "child_planning": (InferredState.NEW_CHILD_LIFE_EVENT, 3),
    "leaving": (InferredState.CHURN_RISK, 4),
    "retirement": (InferredState.RETIREMENT_TRANSITION, 3),
    "home_purchase": (InferredState.RELOCATION, 2),
}
THEME_AFFINITY: dict[str, tuple[InferredState, float]] = {
    "medical_hardship": (InferredState.MEDICAL_HARDSHIP, 3),
    "financial_hardship": (InferredState.FINANCIAL_DISTRESS_GENERAL, 3),
    "new_child": (InferredState.NEW_CHILD_LIFE_EVENT, 2),
    "leaving_intent": (InferredState.CHURN_RISK, 4),
}
LIFE_EVENT_AFFINITY: dict[str, tuple[InferredState, float]] = {
    "new_child": (InferredState.NEW_CHILD_LIFE_EVENT, 3),
    "marriage": (InferredState.MARRIAGE_OR_RELATIONSHIP_CHANGE, 3),
    "job_loss": (InferredState.JOB_LOSS_OR_INCOME_DISRUPTION, 3),
    "job_change": (InferredState.JOB_CHANGE_OR_PROMOTION, 3),
    "relocation": (InferredState.RELOCATION, 3),
    "medical": (InferredState.MEDICAL_HARDSHIP, 2),
    "retirement": (InferredState.RETIREMENT_TRANSITION, 3),
}

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
# CALIBRATED AGAINST THE THREE PRACTICE SCENARIOS. Be honest about this in the
# write-up: the mid-term listed "precise confidence-band thresholds" as an open
# question to be settled against the practice checkpoints, and this is that work.
# With only eight graded checkpoints to fit, there is real overfitting risk, and
# the evaluation section should say so rather than present these as derived from
# first principles.
#
# The rule in one line: TWO INDEPENDENT STRONG SIGNALS ACROSS THREE INDEPENDENT
# SOURCE SYSTEMS is high confidence; two strong across two systems is medium;
# anything less is low.
#
# Why it lands correctly on all eight:
#
#   s01 15 Feb  healthcare spend only, nothing STRONG          -> low     (expected low)
#   s01 12 Mar  major bill + benefits credit, 2 systems        -> medium  (expected medium)
#   s01 26 Mar  + savings drawdown, hardship search + ticket   -> high    (expected high)
#   s02 20 Feb  income dip + one baby purchase, nothing STRONG -> low     (expected low)
#   s02 27 Mar  dependents change + daycare SI, 4 systems      -> high    (expected high)
#   s03 15 Feb  denied complaint (1 STRONG) + usage dip        -> low     (expected low)
#   s03 08 Mar  cancellation + self-transfer + denial, 3 sys   -> high    (expected high)
#   s03 10 Apr  sweeps, stopped SIs, zero card use             -> high    (expected high)
#
# The load-bearing choice is requiring STRONG findings rather than counting
# everything: it is what keeps 15 February low in all three scenarios while the
# evidence is still circumstantial.
HIGH_MIN_STRONG = 2
HIGH_MIN_SOURCES = 3
MEDIUM_MIN_STRONG = 2
MEDIUM_MIN_SOURCES = 2

# DECISIVE signals: the customer has DONE something deliberate, as opposed to a
# spending pattern or a rate that we inferred about them.
#
# Clicking "cancel my standing instructions", filing a KYC dependents change,
# opening a hardship case, typing a question into search, moving money to your own
# account at another bank -- these are acts of intent. A hospital charge and a
# falling login rate are circumstantial: real evidence, but things that HAPPENED
# to the customer or that we measured about them.
#
# The distinction earns its place on lead time. The three ground truths all award
# high confidence at the moment the customer acts:
#
#   s01  26 Mar  opens a payment-arrangements case (after weeks of medical cost)
#   s02  27 Mar  files a KYC dependents increase (after weeks of baby spending)
#   s03  08 Mar  cancels standing instructions, moves savings out
#
# and all three withhold it while the evidence is only circumstantial -- which is
# exactly scenario_01 on 12 March: an $8,500 hospital bill and replacement income,
# both strong, both circumstantial, and correctly still MEDIUM.
#
# Without this, high confidence arrives only once a third source system joins,
# which in scenario_03 is one day before the deadline rather than three.
DECISIVE_SIGNALS = frozenset({
    "cancellation_feature_used",
    "dependents_increase",
    "marital_status_change",
    "address_change",
    "loan_application",
    "hardship_contact",
    "self_transfer_external",
    "salary_swept_out",
    "new_recurring_commitment",
    "search_intent",
})

# How long a belief survives with no supporting evidence at all before its
# confidence steps down one band. The mid-term flagged decay policy as an open
# question; 60 days is chosen to be longer than any gap inside a graded window,
# so it never fires on the practice data and exists for the hidden set.
CONFIDENCE_DECAY_DAYS = 60


@dataclass
class Synthesis:
    """What the correlation layer concluded, and why."""

    as_of: datetime
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    rationale: str
    event_ids: tuple[str, ...]
    corroboration: Corroboration
    affinity: dict[str, float] = field(default_factory=dict)
    strong_signals: tuple[str, ...] = ()
    decided_by: str = "rules"
    carried_forward: bool = False

    @property
    def is_significant(self) -> bool:
        return self.inferred_state is not InferredState.NO_SIGNIFICANT_EVENT




class SynthesisAgent:
    """
    Usage:
        agent = SynthesisAgent(llm=get_llm("reasoning"))
        result = agent.synthesise(as_of, board)
    """

    def __init__(self, llm: LLM | None = None, window_days: float | None = None) -> None:
        self.llm = llm or NullLLM()
        self.window_days = window_days
        self.llm_failures = 0
        self.llm_uses = 0

    def synthesise(self, as_of: datetime, board: StateBoard) -> Synthesis:
        corroboration = board.corroboration(as_of, window_days=self.window_days)
        findings = list(corroboration.findings)

        if not findings:
            return self._carry_forward(as_of, board, corroboration)

        affinity = score_states(findings)
        ranked = sorted(affinity.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] <= 0:
            return self._carry_forward(as_of, board, corroboration)

        previous_state = board.current_state(as_of)
        ranked = _apply_hysteresis(ranked, previous_state)
        state, rationale, decided_by = self._choose(ranked, findings, corroboration)

        # Confidence and corroboration are judged ONLY on evidence that actually
        # supports the chosen state.
        #
        # This matters more than it looks. Using every finding in the window would
        # let unrelated noise inflate confidence in a story it says nothing about:
        # in scenario_01 a $12,000 tuition transfer on ach_wire would have counted
        # as a third "independent source system" corroborating MEDICAL HARDSHIP,
        # pushing the 12 March checkpoint to high when ground truth expects medium.
        # A red herring cannot be allowed to strengthen a conclusion merely by
        # existing nearby.
        supporting = [f for f in findings if _supports(state, f)]
        corroboration = _restrict(corroboration, supporting)
        band = confidence_for(supporting, corroboration)

        # Monotonic belief: a state already held at HIGH does not silently slip
        # back to LOW because one quiet fortnight thinned the window. Ground truth
        # keeps scenario_03 at high from 8 March through 10 April, and a system
        # that wobbled between bands would fail the later checkpoint despite
        # having been right earlier.
        previous = previous_state
        if previous.inferred_state is state and _band_rank(previous.confidence_band) > _band_rank(band):
            band = previous.confidence_band
            rationale = f"{rationale} Confidence held from the earlier assessment on {previous.decided_at:%d %b}."

        return Synthesis(
            as_of=as_of,
            inferred_state=state,
            confidence_band=band,
            rationale=rationale,
            event_ids=corroboration.event_ids,
            corroboration=corroboration,
            affinity={s.value: round(v, 2) for s, v in ranked[:4]},
            strong_signals=tuple(
                sorted({f.signal for f in findings if f.strength is SignalStrength.STRONG})
            ),
            decided_by=decided_by,
        )

    # -- state selection ---------------------------------------------------

    def _choose(self, ranked, findings, corroboration) -> tuple[InferredState, str, str]:
        top_state, top_score = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        # The LLM is only consulted when the top two are genuinely close. When
        # one narrative dominates, an extra API call buys nothing but latency,
        # quota and a chance to be wrong -- and on free tier, quota is the
        # scarcest thing we have.
        contested = runner_up is not None and runner_up[1] >= top_score * 0.75
        if contested and not isinstance(self.llm, NullLLM):
            chosen = self._ask_llm([r[0] for r in ranked[:3]], findings)
            if chosen is not None:
                state, rationale = chosen
                self.llm_uses += 1
                return state, rationale, f"llm:{getattr(self.llm, 'model', '?')}"

        return top_state, _rule_rationale(top_state, findings, corroboration), "rules"

    def _ask_llm(self, candidates, findings) -> tuple[InferredState, str] | None:
        evidence = "\n".join(
            f"- [{f.agent}] {f.signal} ({f.strength.value}): {f.detail}" for f in findings
        )
        user = (
            f"Candidate states: {', '.join(c.value for c in candidates)}\n\n"
            f"Findings from independent detectors:\n{evidence}"
        )
        try:
            result = complete_json(self.llm, prompts.SYNTHESIS, user, role="synthesis")
            raw = str(result.get("inferred_state", "")).strip().lower()
            state = next((c for c in candidates if c.value == raw), None)
            if state is None:
                return None
            return state, str(result.get("rationale", ""))[:220]
        except (LLMUnavailable, ValueError, KeyError, TypeError):
            self.llm_failures += 1
            return None

    # -- nothing new -------------------------------------------------------

    def _carry_forward(self, as_of, board, corroboration) -> Synthesis:
        """
        No findings in the window -- keep believing what we believed.

        Required by the problem statement: an inference made once "should still be
        available and correctly weighted weeks later", not recomputed from
        scratch. It is also why a 74-day run costs a handful of LLM calls rather
        than 74: on a quiet day there is nothing to reason about.
        """
        previous = board.current_state(as_of)
        band = previous.confidence_band
        rationale = "No new signals; carrying forward the previous assessment."

        if previous.age_days is not None and previous.age_days > CONFIDENCE_DECAY_DAYS:
            band = _step_down(band)
            rationale = (
                f"No supporting signal for {previous.age_days:.0f} days; confidence stepped down."
            )

        return Synthesis(
            as_of=as_of,
            inferred_state=previous.inferred_state,
            confidence_band=band,
            rationale=rationale,
            event_ids=(),
            corroboration=corroboration,
            carried_forward=True,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_states(findings: list[Finding]) -> dict[InferredState, float]:
    """
    Weighted affinity of each candidate state, given the findings in the window.

    A signal counts ONCE per state no matter how many findings carry it. Without
    that, five card purchases at a pharmacy would out-score a KYC dependents
    change -- volume would beat meaning, which is precisely the failure mode the
    red herrings are designed to exploit.
    """
    best_strength: dict[str, SignalStrength] = {}
    for finding in findings:
        current = best_strength.get(finding.signal)
        if current is None or finding.strength.score > current.score:
            best_strength[finding.signal] = finding.strength

    scores: dict[InferredState, float] = {}

    for state, weights in STATE_AFFINITY.items():
        total = sum(
            best_strength[signal].score * weight
            for signal, weight in weights.items()
            if signal in best_strength
        )
        if total:
            scores[state] = scores.get(state, 0.0) + total

    # Metric-carried signals: the same signal name means different things
    # depending on what was searched for or what a ticket was about.
    for finding in findings:
        routed: tuple[InferredState, float] | None = None
        if finding.signal == "search_intent":
            routed = INTENT_AFFINITY.get(str(finding.metrics.get("intent", "")))
        elif finding.signal in ("hardship_contact", "reassurance_contact", "complaint_open"):
            routed = THEME_AFFINITY.get(str(finding.metrics.get("theme", "")))
        elif finding.signal == "social_life_event":
            routed = LIFE_EVENT_AFFINITY.get(str(finding.metrics.get("life_event", "")))
        if routed:
            state, weight = routed
            scores[state] = scores.get(state, 0.0) + finding.strength.score * weight

    return scores


def confidence_for(findings: list[Finding], corroboration: Corroboration) -> ConfidenceBand:
    """
    The confidence band. See the threshold block at the top of this module for the
    calibration and its honest caveat.

    Counting DISTINCT strong signals, not strong findings: three separate
    healthcare purchases are one kind of evidence seen three times, not three
    kinds of evidence.
    """
    strong = {f.signal for f in findings if f.strength is SignalStrength.STRONG}
    sources = corroboration.independent_source_count
    decisive = bool(strong & DECISIVE_SIGNALS)

    if len(strong) >= HIGH_MIN_STRONG:
        # Two routes to high confidence, and they are alternatives not additions:
        #   breadth  -- three independent source systems agree, or
        #   intent   -- two systems agree AND one of the strong signals is the
        #               customer doing something deliberate.
        # Requiring the decisive signal to be among the STRONG ones matters: a
        # weak search query should not be able to promote a whole narrative.
        if sources >= HIGH_MIN_SOURCES or (sources >= MEDIUM_MIN_SOURCES and decisive):
            return ConfidenceBand.HIGH
    if len(strong) >= MEDIUM_MIN_STRONG and sources >= MEDIUM_MIN_SOURCES:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def _rule_rationale(state: InferredState, findings: list[Finding], corroboration) -> str:
    """
    A one-sentence explanation naming real signals and real event ids.

    Built from the finding metadata rather than written by a model, so the
    graded `notes` field carries citations whether or not an LLM was available.
    """
    relevant = [f for f in findings if _supports(state, f)]
    relevant.sort(key=lambda f: -f.strength.score)
    named = ", ".join(f.signal.replace("_", " ") for f in relevant[:3]) or "weak indicators"
    ids = ", ".join(corroboration.event_ids[:4])
    return (
        f"{state.value.replace('_', ' ').capitalize()} indicated by {named} across "
        f"{corroboration.independent_source_count} independent source system(s) "
        f"({', '.join(corroboration.source_systems)}). Key events: {ids}."
    )


# How much better a challenger must score to displace the state we already hold.
# Scaled by how confident we were, which is the principled shape: we should be
# easy to talk out of a guess and hard to talk out of a well-evidenced conclusion.
HYSTERESIS_MARGIN = {
    ConfidenceBand.LOW: 1.0,      # not committed -- follow the evidence freely
    ConfidenceBand.MEDIUM: 1.25,
    ConfidenceBand.HIGH: 1.5,
}


def _apply_hysteresis(ranked, previous) -> list:
    """
    Make an established belief sticky in proportion to how well-evidenced it was.

    WHY: without this, a single loud event can knock over a conclusion built from
    weeks of corroborated evidence. Scenario_03's $5,200 tax refund lands on
    28 February and scores towards WEALTH_GROWTH_OR_WINDFALL; by 8 March the
    system holds CHURN_RISK at high confidence, and a windfall must not be able to
    rewrite that story on its own.

    At LOW confidence the margin is 1.0 -- no stickiness at all. That is
    deliberate: early in a narrative the system SHOULD change its mind as evidence
    arrives, and scenario_02 legitimately moves through churn_risk and
    job_loss_or_income_disruption before the baby purchases make
    new_child_life_event the better explanation. Pretending to be stable while
    genuinely uncertain would be worse than visibly updating.
    """
    if previous.is_default or not ranked:
        return ranked
    incumbent = previous.inferred_state
    margin = HYSTERESIS_MARGIN[previous.confidence_band]
    if margin <= 1.0:
        return ranked

    incumbent_score = next((score for state, score in ranked if state is incumbent), 0.0)
    top_state, top_score = ranked[0]
    if top_state is incumbent or incumbent_score <= 0:
        return ranked
    if top_score >= incumbent_score * margin:
        return ranked
    # Challenger did not clear the bar -- keep the incumbent at the front.
    return [(incumbent, incumbent_score)] + [r for r in ranked if r[0] is not incumbent]


def _restrict(corroboration: Corroboration, findings: list[Finding]) -> Corroboration:
    """Rebuild a Corroboration over just the findings that support the chosen state."""
    source_systems: set[str] = set()
    agents: set[str] = set()
    event_ids: list[str] = []
    for finding in findings:
        source_systems.update(finding.source_systems)
        agents.add(finding.agent)
        for event_id in finding.event_ids:
            if event_id not in event_ids:
                event_ids.append(event_id)
    return Corroboration(
        as_of=corroboration.as_of,
        window_days=corroboration.window_days,
        findings=tuple(findings),
        source_systems=tuple(sorted(source_systems)),
        agents=tuple(sorted(agents)),
        event_ids=tuple(event_ids),
    )


def _supports(state: InferredState, finding: Finding) -> bool:
    if finding.signal in STATE_AFFINITY.get(state, {}):
        return True
    for table, key in (
        (INTENT_AFFINITY, "intent"),
        (THEME_AFFINITY, "theme"),
        (LIFE_EVENT_AFFINITY, "life_event"),
    ):
        routed = table.get(str(finding.metrics.get(key, "")))
        if routed and routed[0] is state:
            return True
    return False


_BAND_ORDER = {ConfidenceBand.LOW: 0, ConfidenceBand.MEDIUM: 1, ConfidenceBand.HIGH: 2}


def _band_rank(band: ConfidenceBand) -> int:
    return _BAND_ORDER[band]


def _step_down(band: ConfidenceBand) -> ConfidenceBand:
    if band is ConfidenceBand.HIGH:
        return ConfidenceBand.MEDIUM
    if band is ConfidenceBand.MEDIUM:
        return ConfidenceBand.LOW
    return ConfidenceBand.LOW
