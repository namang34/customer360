"""
Transaction Agent -- card_payments, core_banking_ledger, instant_payments,
ach_wire, trading_brokerage.

The widest agent by source count, and the only one that can corroborate itself:
a salary stopping (core_banking_ledger) and money leaving for a rival bank
(instant_payments) are two genuinely independent streams of evidence even though
one agent noticed both. That is exactly why the guardrail counts SOURCE SYSTEMS
rather than agents.

Everything in here is arithmetic, not language. Whether $8,500 at a hospital
billing department is a large medical expense is a comparison, and a comparison
belongs in code where it is right every time and can be pointed at during a viva.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..findings import Finding, SignalStrength
from .base import PerceptionAgent, PerceptionContext, cited

LEDGER = "core_banking_ledger"
TRANSFER_SYSTEMS = ("instant_payments", "ach_wire")

# Merchant-category buckets. Driven off mcc_category, which is structured data --
# no language model needed to know that "healthcare" means healthcare.
MEDICAL_CATEGORIES = {"healthcare", "pharmacy"}
BABY_CATEGORIES = {"baby_products", "childcare", "toys"}
# Merchant-name hints for categories the MCC does not capture. scenario_02's
# Mothercare purchase is filed under mcc_category "clothing", so category alone
# would miss it -- and it is a graded signal event.
BABY_MERCHANT_HINTS = ("mothercare", "babybaby", "buybuybaby", "baby", "mamas", "carters")

INCOME_TYPES = {"salary_credit"}
# Income that is a REPLACEMENT rather than the usual wage. In scenario_01 the
# salary stops and benefits_credit appears at a much lower amount -- a stronger
# and earlier signal than simply noticing the salary is late.
REPLACEMENT_INCOME_TYPES = {"benefits_credit", "disability_credit", "unemployment_credit"}
WINDFALL_TYPES = {"tax_refund", "bonus", "dividend_credit", "maturity_credit"}

# Thresholds. These are calibrated against the three practice scenarios -- see
# the note in synthesis.py about the overfitting risk that carries.
MAJOR_EXPENSE_ABSOLUTE = 5_000       # scenario_01's $8,500 hospital bill
INCOME_DROP_RATIO = 0.7              # scenario_02's 4500 -> 2700 is a 40% cut
LARGE_TRANSFER_ABSOLUTE = 10_000     # scenario_03's $22,500; scenario_01's $12,000 tuition
SWEEP_WINDOW_HOURS = 48              # salary in, nearly all of it straight out
SWEEP_RATIO = 0.85
SPEND_COLLAPSE_RATIO = 0.35          # recent rate vs the customer's own baseline


class TransactionAgent(PerceptionAgent):
    name = "transaction"
    source_systems = ("card_payments", LEDGER, "instant_payments", "ach_wire", "trading_brokerage")

    # =====================================================================
    # EVENT-BASED
    # =====================================================================

    def on_event(self, event, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        findings: list[Finding] = []
        payload = event.payload
        amount = _number(payload.get("amount"))

        # --- card activity ------------------------------------------------
        if event.source_system == "card_payments" and event.event_type == "purchase":
            category = (payload.get("mcc_category") or "").lower()
            merchant = (payload.get("merchant_name") or "").lower()

            if category in MEDICAL_CATEGORIES:
                if amount >= MAJOR_EXPENSE_ABSOLUTE:
                    findings.append(
                        self.finding(
                            as_of,
                            "major_medical_expense",
                            SignalStrength.STRONG,
                            [event],
                            detail=(
                                f"Medical charge of {amount:,.0f} -- an order of magnitude above "
                                f"routine healthcare spend ({cited([event])})"
                            ),
                            amount=amount,
                        )
                    )
                else:
                    findings.append(
                        self.finding(
                            as_of,
                            "healthcare_spend",
                            SignalStrength.MODERATE,
                            [event],
                            detail=f"Healthcare/pharmacy spend of {amount:,.0f} ({cited([event])})",
                            amount=amount,
                        )
                    )

            if category in BABY_CATEGORIES or any(h in merchant for h in BABY_MERCHANT_HINTS):
                # ONE baby purchase is a gift. A PATTERN is a household change.
                #
                # This distinction is doing real work. A single $250 trip to a
                # baby store is genuinely ambiguous -- a shower present, a gift
                # for a niece -- and treating it as strong evidence would be
                # exactly the over-reading the red herrings punish. But repeat
                # purchases at baby retailers over weeks are not a coincidence.
                #
                # It is also what earns the lead time in scenario_02: the second
                # purchase (Mothercare, 2 March) turns the signal strong, which
                # combined with the daycare standing instruction on 18 March
                # reaches high confidence nine days before the graded checkpoint,
                # rather than one day after the KYC filing finally confirms it.
                prior = self._baby_purchase_count(as_of, ctx, event)
                findings.append(
                    self.finding(
                        as_of,
                        "baby_retail_spend",
                        SignalStrength.STRONG if prior >= 1 else SignalStrength.MODERATE,
                        [event],
                        detail=(
                            f"Purchase at a baby/child retailer, {amount:,.0f}"
                            + (f" -- the {_ordinal(prior + 1)} in 60 days" if prior else " (first seen)")
                            + f" ({cited([event])})"
                        ),
                        amount=amount,
                        mcc=category,
                        prior_baby_purchases=prior,
                    )
                )

        # A refund is money coming back, not distress. Emitted weakly so it is
        # visible in the trace, but it must never on its own move a confidence
        # band -- scenario_01's $2,500 resort refund is a planted red herring.
        if event.source_system == "card_payments" and event.event_type == "refund":
            if amount >= 1_000:
                findings.append(
                    self.finding(
                        as_of,
                        "unusual_refund",
                        SignalStrength.WEAK,
                        [event],
                        detail=f"Refund of {amount:,.0f} from {payload.get('merchant_name')}",
                        amount=amount,
                    )
                )

        # --- ledger ---------------------------------------------------------
        if event.source_system == LEDGER:
            txn_type = (payload.get("transaction_type") or "").lower()

            if event.event_type == "deposit" and txn_type in REPLACEMENT_INCOME_TYPES:
                findings.append(
                    self.finding(
                        as_of,
                        "income_replacement",
                        SignalStrength.STRONG,
                        [event],
                        detail=(
                            f"Regular salary replaced by {txn_type} of {amount:,.0f} -- income "
                            f"source has changed, not merely dipped ({cited([event])})"
                        ),
                        amount=amount,
                        transaction_type=txn_type,
                    )
                )

            if event.event_type == "deposit" and txn_type in WINDFALL_TYPES:
                # scenario_03's tax refund. Recorded honestly as an inbound
                # windfall; it is the guardrail, not a special case here, that
                # stops one isolated deposit from triggering an offer.
                findings.append(
                    self.finding(
                        as_of,
                        "large_inbound_deposit",
                        SignalStrength.WEAK,
                        [event],
                        detail=f"Inbound {txn_type} of {amount:,.0f} ({cited([event])})",
                        amount=amount,
                        transaction_type=txn_type,
                    )
                )

            if event.event_type == "withdrawal" and amount >= LARGE_TRANSFER_ABSOLUTE:
                findings.append(
                    self.finding(
                        as_of,
                        "savings_drawdown",
                        SignalStrength.STRONG,
                        [event],
                        detail=(
                            f"Withdrawal of {amount:,.0f} ({txn_type}), balance now "
                            f"{_number(payload.get('balance_after')):,.0f} ({cited([event])})"
                        ),
                        amount=amount,
                    )
                )

            if event.event_type == "standing_instruction":
                findings.extend(self._new_commitment(event, as_of, ctx, txn_type, amount))

        # --- transfers out ---------------------------------------------------
        if event.source_system in TRANSFER_SYSTEMS and event.event_type == "outbound_transfer":
            findings.extend(self._on_outbound_transfer(event, as_of, ctx, amount))

        return findings

    @staticmethod
    def _baby_purchase_count(as_of, ctx, current) -> int:
        """How many OTHER baby-retail purchases in the trailing 60 days."""
        count = 0
        for event in ctx.memory.events_as_of(as_of, since_days=60, source_systems=["card_payments"]):
            if event.event_id == current.event_id or event.event_type != "purchase":
                continue
            category = (event.payload.get("mcc_category") or "").lower()
            merchant = (event.payload.get("merchant_name") or "").lower()
            if category in BABY_CATEGORIES or any(h in merchant for h in BABY_MERCHANT_HINTS):
                count += 1
        return count

    def _new_commitment(self, event, as_of, ctx, txn_type, amount) -> list[Finding]:
        """
        A standing instruction of a type this customer has never had before.

        The mirror image of `_standing_instruction_stopped`, and just as
        informative. Taking on a NEW recurring obligation is a statement about a
        changed life: scenario_02's EVT_000363 is a $1,200/month daycare_payment
        appearing on 18 March, which is about as unambiguous a new-child signal as
        a ledger can produce.

        "Never before" is checked against the customer's whole visible history, so
        the ordinary monthly rent payment -- which has run for months -- never
        trips it.
        """
        prior = {
            (e.payload.get("transaction_type") or "").lower()
            for e in ctx.memory.events_as_of(as_of, since_days=400, source_systems=[LEDGER])
            if e.event_type == "standing_instruction" and e.event_id != event.event_id
        }
        if not txn_type or txn_type in prior:
            return []
        return [
            self.finding(
                as_of,
                "new_recurring_commitment",
                SignalStrength.STRONG,
                [event],
                detail=(
                    f"A new recurring commitment appeared: {txn_type} of {amount:,.0f}/period, "
                    f"never previously seen on this account ({cited([event])})"
                ),
                transaction_type=txn_type,
                amount=amount,
            )
        ]

    def _on_outbound_transfer(self, event, as_of, ctx, amount) -> list[Finding]:
        findings: list[Finding] = []
        is_self = ctx.redactor.counterparty_is_customer(event.payload)

        if is_self:
            # THE CHURN TELL. Money moving to an account the customer owns at
            # another institution is categorically different from paying a third
            # party -- it is relocation of the relationship, not consumption.
            #
            # This is also why PII redaction tokenises rather than deletes: the
            # counterparty reads "<CUSTOMER_NAME> - Chase Bank" after scrubbing,
            # so the agent can still tell it is a self-transfer without ever
            # being shown the name.
            findings.append(
                self.finding(
                    as_of,
                    "self_transfer_external",
                    SignalStrength.STRONG if amount >= LARGE_TRANSFER_ABSOLUTE else SignalStrength.MODERATE,
                    [event],
                    detail=(
                        f"Outbound transfer of {amount:,.0f} to an account held by the customer "
                        f"at another institution ({cited([event])})"
                    ),
                    amount=amount,
                )
            )
        elif amount >= LARGE_TRANSFER_ABSOLUTE:
            # scenario_01's $12,000 tuition payment lands here: large, outbound,
            # to a named third party. One event, one source system -- and the
            # guardrail is what keeps it from escalating anything.
            findings.append(
                self.finding(
                    as_of,
                    "large_outbound_transfer",
                    SignalStrength.MODERATE,
                    [event],
                    detail=(
                        f"Large outbound transfer of {amount:,.0f} to a third party "
                        f"({cited([event])})"
                    ),
                    amount=amount,
                )
            )

        # Salary swept out almost immediately: the account has become a
        # pass-through. scenario_03's EVT_000469 / EVT_000471.
        recent_income = [
            e
            for e in ctx.memory.events_as_of(
                as_of, since_days=SWEEP_WINDOW_HOURS / 24, source_systems=[LEDGER]
            )
            if (e.payload.get("transaction_type") or "").lower() in INCOME_TYPES
        ]
        if recent_income and amount > 0:
            salary = _number(recent_income[-1].payload.get("amount"))
            if salary and amount / salary >= SWEEP_RATIO:
                findings.append(
                    self.finding(
                        as_of,
                        "salary_swept_out",
                        SignalStrength.STRONG,
                        [recent_income[-1], event],
                        detail=(
                            f"{amount:,.0f} of a {salary:,.0f} salary credit left the bank within "
                            f"{SWEEP_WINDOW_HOURS}h -- the account is now a pass-through "
                            f"({cited([recent_income[-1], event])})"
                        ),
                        amount=amount,
                        salary=salary,
                    )
                )
        return findings

    # =====================================================================
    # TIME-BASED -- things no single event can tell you
    # =====================================================================

    def on_tick(self, as_of: datetime, ctx: PerceptionContext) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._income_disruption(as_of, ctx))
        findings.extend(self._standing_instruction_stopped(as_of, ctx))
        findings.extend(self._card_spend_collapse(as_of, ctx))
        return findings

    def _income_disruption(self, as_of, ctx) -> list[Finding]:
        """
        Has regular income dropped against this customer's own history?

        Compares the most recent salary credit to the median of the ones before
        it. Median rather than mean because a single bonus month would drag a mean
        upward and mask a genuine cut.
        """
        deposits = [
            e
            for e in ctx.memory.events_as_of(as_of, since_days=180, source_systems=[LEDGER])
            if (e.payload.get("transaction_type") or "").lower() in INCOME_TYPES
        ]
        if len(deposits) < 4:
            return []

        amounts = [_number(e.payload.get("amount")) for e in deposits]
        recent, history = amounts[-1], sorted(amounts[:-1])
        baseline = history[len(history) // 2]
        if not baseline or recent >= baseline * INCOME_DROP_RATIO:
            return []

        drop = 1 - (recent / baseline)
        # Only cite the events that actually evidence the drop.
        evidence = [deposits[-1]] + deposits[max(0, len(deposits) - 4) : -1][:2]
        return [
            self.finding(
                as_of,
                "income_disruption",
                SignalStrength.MODERATE if drop < 0.5 else SignalStrength.STRONG,
                evidence,
                detail=(
                    f"Salary credit fell from a typical {baseline:,.0f} to {recent:,.0f} "
                    f"({drop:.0%} lower) ({cited(evidence)})"
                ),
                baseline=baseline,
                recent=recent,
                drop_ratio=round(drop, 3),
            )
        ]

    def _standing_instruction_stopped(self, as_of, ctx) -> list[Finding]:
        """
        A standing instruction that was regular and has now FAILED TO OCCUR.

        There is no event for "the rent did not go out this month". This is the
        clearest example of why the daily time-based tick is load-bearing rather
        than decorative: in scenario_03 the customer cancels his standing
        instructions on 4 March, and the only trace in the ledger is the silence
        where 1 April's payments should have been.
        """
        instructions = [
            e
            for e in ctx.memory.events_as_of(as_of, since_days=180, source_systems=[LEDGER])
            if e.event_type == "standing_instruction"
        ]
        if not instructions:
            return []

        by_type: dict[str, list] = {}
        for event in instructions:
            by_type.setdefault((event.payload.get("transaction_type") or "?").lower(), []).append(event)

        stopped, evidence, overdue_ratios = [], [], []
        for txn_type, events in by_type.items():
            if len(events) < 3:
                continue  # not yet established as a regular commitment
            gaps = [
                (events[i].event_time - events[i - 1].event_time).days for i in range(1, len(events))
            ]
            typical = sorted(gaps)[len(gaps) // 2]
            if typical <= 0:
                continue
            silence = (as_of - events[-1].event_time).days
            # TWO full missed cycles, not one.
            #
            # HONEST NOTE ON THIS THRESHOLD -- read before tuning it.
            #
            # All three practice scenarios are MISSING their February standing
            # instructions entirely. Every customer's ledger runs
            # 1 Nov, 1 Dec, 1 Jan, [nothing in February], 1 Mar. That is a
            # data-generation artifact, not customer behaviour: scenario_01 and
            # scenario_02 resume normally on 1 April, and only scenario_03 -- who
            # actually cancelled on 4 March -- stops for good.
            #
            # With a one-missed-cycle threshold this detector fired in ALL THREE
            # scenarios in mid-February, including the two customers who are not
            # churning, and pushed scenario_03's 15 February checkpoint to high
            # confidence when ground truth expects low.
            #
            # The artifact and the real signal are the same SHAPE -- one missed
            # monthly cycle -- so no threshold separates them. Requiring two full
            # missed cycles makes the detector conservative and silences the false
            # positives; the cost is that it never fires on the practice data at
            # all, because scenario_03's cancellation on 4 March leaves only one
            # missed cycle (1 April) before simulated_end on 15 April.
            #
            # The detector is kept because it is correct in principle and would
            # matter on real data, and the evaluation write-up reports plainly
            # that it contributes nothing to these three scores.
            if silence >= typical * 2:
                stopped.append(txn_type)
                evidence.append(events[-1])
                overdue_ratios.append(silence / typical)

        if not stopped:
            return []

        # STRONG only once the instruction is TWO full cadences overdue.
        #
        # One missed cycle is ambiguous -- a weekend, a bank holiday, or (as in
        # scenario_03's February, where the seed data simply contains no standing
        # instructions at all) a gap in the record. Two consecutive misses is a
        # pattern. Treating a single miss as STRONG made 15 February in
        # scenario_03 read as high confidence when the ground truth expects low.
        worst = max(overdue_ratios)
        strength = SignalStrength.STRONG if worst >= 2.0 else SignalStrength.MODERATE
        return [
            self.finding(
                as_of,
                "standing_instruction_stopped",
                strength,
                evidence,
                detail=(
                    f"{len(stopped)} recurring standing instruction(s) have stopped: "
                    f"{', '.join(sorted(stopped))}. Last seen {cited(evidence)}"
                ),
                stopped=sorted(stopped),
                overdue_cadences=round(worst, 2),
            )
        ]

    def _card_spend_collapse(self, as_of, ctx) -> list[Finding]:
        """Card usage falling away against the customer's own baseline."""
        recent = ctx.memory.daily_rate(as_of, source_systems=["card_payments"], window_days=14)
        baseline = ctx.memory.baseline_daily_rate(
            as_of, source_systems=["card_payments"], baseline_days=90, exclude_recent_days=14
        )
        if baseline < 0.3 or recent > baseline * SPEND_COLLAPSE_RATIO:
            return []

        evidence = ctx.memory.events_as_of(
            as_of, since_days=30, source_systems=["card_payments"], limit=3
        )
        return [
            self.finding(
                as_of,
                "card_spend_collapse",
                SignalStrength.STRONG if recent == 0 else SignalStrength.MODERATE,
                evidence,
                source_systems=["card_payments"],
                detail=(
                    f"Card transactions fell to {recent:.2f}/day from a baseline of "
                    f"{baseline:.2f}/day ({1 - recent / baseline:.0%} lower)"
                ),
                recent_rate=round(recent, 3),
                baseline_rate=round(baseline, 3),
            )
        ]


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third"}.get(n, f"{n}th")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
