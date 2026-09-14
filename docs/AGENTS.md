# Agent Register

Per deliverable 7.2 — what each agent is, what it can do, and the exact shape of
data it expects and returns. Every prompt lives in `src/c360/prompts.py`; every
schema is a dataclass in the file named below.

---

## Shared types

```python
# src/c360/schema.py
Event(event_id, event_time, ingestion_time, customer_id, account_id,
      source_system, event_type, schema_version, payload)          # frozen
  .release_time        -> max(ingestion_time, event_time)
  .is_late_arrival     -> ingestion_time > event_time
  .perception_agent    -> which agent owns this event, or None

# src/c360/findings.py
Finding(agent, as_of, signal, strength, event_ids, source_systems,
        detail, window_start, metrics)                             # frozen
  strength: SignalStrength = weak | moderate | strong
```

`Finding` is the only thing a perception agent may emit. It is structured, never
prose: the guardrail counts `source_systems`, the notes field cites `event_ids`,
and synthesis maps `signal` to a candidate state. Free text would break all three.

---

## Layer 1 — Perception swarm

Four agents, disjoint source systems, no interdependency. Each implements
`on_event(event, as_of, ctx) -> list[Finding]` and
`on_tick(as_of, ctx) -> list[Finding]`.

`ctx` is a `PerceptionContext(memory, redactor, entities, llm)` — deliberately
**not** including the state board, so no agent can read a peer's findings.

### Transaction Agent · `agents/transaction.py`

| | |
|---|---|
| **Reads** | `card_payments`, `core_banking_ledger`, `instant_payments`, `ach_wire`, `trading_brokerage` |
| **Tools** | `memory.events_as_of` (episodic, as_of-filtered) · `memory.daily_rate` · `memory.baseline_daily_rate` · `redactor.counterparty_is_customer` |
| **Prompt** | none — every detection here is arithmetic |
| **Emits** | `income_disruption`, `income_replacement`, `major_medical_expense`, `healthcare_spend`, `savings_drawdown`, `large_outbound_transfer`, `self_transfer_external`, `salary_swept_out`, `standing_instruction_stopped`, `new_recurring_commitment`, `card_spend_collapse`, `baby_retail_spend`, `large_inbound_deposit`, `unusual_refund` |

The only agent that can corroborate itself: a salary stopping
(`core_banking_ledger`) and money leaving for a rival bank (`instant_payments`)
are two independent streams of evidence from one agent. This is why the guardrail
counts source systems rather than agents.

**Event-based** — medical spend, baby retail, income replacement, windfall
deposits, savings drawdown, outbound transfers, salary sweeps, new commitments.
**Time-based** — income disruption, stopped standing instructions, card spend
collapse. None of these can be detected from a single event.

### Usage Agent · `agents/usage.py`

| | |
|---|---|
| **Reads** | `web_app_events` |
| **Tools** | `memory.daily_rate` / `baseline_daily_rate` · `redactor.scrub` · `llm` |
| **Prompt** | `prompts.SEARCH_INTENT` → `{intent, confidence, reason}` |
| **Fallback** | `INTENT_KEYWORDS` longest-phrase match |
| **Emits** | `cancellation_feature_used`, `engagement_drop`, `session_length_collapse`, `search_intent` |

`cancellation_feature_used` is the earliest churn signal in scenario_03 — a click
on `manage_standing_instructions_cancel` two days before any money moves, and the
source of that scenario's three-day lead time.

### Support Agent · `agents/support.py`

| | |
|---|---|
| **Reads** | `support_logs` |
| **Tools** | `redactor.scrub` · `llm` |
| **Prompt** | `prompts.TICKET_THEME` → `{theme, customer_distress, reason}` |
| **Fallback** | hardship / child / leaving keyword sets |
| **Emits** | `complaint_denied`, `complaint_open`, `hardship_contact`, `reassurance_contact` |

Two independent paths: a **structural** read of `resolution_status` (needs no
model) and a **textual** read of `raw_text`. The prompt is told who wrote the text,
because a resolved ticket's body is often the bank's reply — scenario_03's is the
bank refusing a fee waiver, and reading that as a calm customer inverts the signal.
Findings are de-duplicated per signal so one ticket counts once.

### Life-Signal Agent · `agents/life_signal.py`

| | |
|---|---|
| **Reads** | `loan_kyc`, `social_signal_consented` |
| **Tools** | `redactor.scrub` · `llm` |
| **Prompt** | `prompts.SOCIAL_LIFE_EVENT` → `{life_event, confidence, reason}` |
| **Fallback** | `SOCIAL_KEYWORDS` |
| **Emits** | `dependents_increase`, `marital_status_change`, `address_change`, `loan_application`, `social_life_event` |

Lowest volume, highest weight: a KYC filing is the customer declaring a change to
their bank, not an inference about them. Only a dependents *increase* counts as a
new-child signal. **Consent gate**: a `social_signal_consented` record with no
`consent_flag` is never read, whatever it contains. A social post never exceeds
`moderate` — it is self-reported and unverified.

---

## State board · `state_board.py`

Not an agent — the surface the swarm writes to and synthesis reads.

```python
board.publish(finding)                    # perception writes
board.findings(as_of, window_days=30)     # time-scoped read
board.corroboration(as_of) -> Corroboration(source_systems, agents, event_ids,
                                            is_corroborated)
board.should_synthesise(as_of) -> bool    # >= 2 DISTINCT agents flagged
board.current_state(as_of)  -> BoardState # the persisted belief
```

Two different counts, deliberately not the same number: `should_synthesise` counts
**agents** (is there anything to correlate?), the guardrail counts **source
systems** (may we act?).

---

## Layer 2 — Synthesis Agent · `synthesis.py`

| | |
|---|---|
| **Input** | `StateBoard` + `as_of`. Never raw events. |
| **Output** | `Synthesis(inferred_state, confidence_band, rationale, event_ids, corroboration, affinity, strong_signals, decided_by, carried_forward)` |
| **Prompt** | `prompts.SYNTHESIS` → `{inferred_state, rationale}` |
| **Called** | only when the top two candidate states are within 25% of each other |
| **Fallback** | highest-affinity state; rationale built from finding metadata so citations survive an offline run |

Decides **what** (weighted signal→state affinity) and **how sure**
(`2 strong signals AND (3 sources OR 2 sources + a decisive signal)`) as separate
questions. Confidence is arithmetic, never asked of a model. Hysteresis scaled by
confidence stops an established belief being displaced by a single loud event.

---

## Layer 3 — Decision

### Action Proposer · `action.py`

| | |
|---|---|
| **Input** | `Synthesis` + `EpisodicMemory` (for escalation history) + entities |
| **Output** | `ActionProposal(action, action_subtype, rationale, policy_ids, policy_text, event_ids, decided_by, gate_reason)` |
| **Tools** | `semantic.retrieve_policies` (ChromaDB) · `memory.decisions_as_of` |
| **Prompt** | `prompts.PROPOSER` → `{action, action_subtype, rationale}` |
| **Fallback** | the highest-ranked eligible policy's action and subtype |

**Gate 1 — confidence**: nothing but `no_action` below HIGH. This single rule
produces the correct answer at five of the eight graded checkpoints. Retrieved
policies are then filtered on metadata (`state`, `min_confidence`, `stage`) —
nearest-neighbour is not the same as applicable.

### Guardrail · `guardrail.py`

| | |
|---|---|
| **Input** | `ActionProposal` + `Synthesis` |
| **Output** | `GuardrailVerdict(passed, reason, checks, independent_sources, event_ids)` |
| **Prompt** | **none, by design** |

**Gate 2.** ≥ 2 independent `source_systems` AND ≥ 2 distinct events, for
`compliance_fraud_hold`, `relationship_manager_escalation`, `personalized_offer`.
`support_intervention` and `proactive_retention_outreach` are unguarded: the bar
should match the cost of being wrong.

### Critique Agent · `critique.py`

| | |
|---|---|
| **Input** | `ActionProposal` + `Synthesis` + `GuardrailVerdict` |
| **Output** | `Critique(verdict, reason, checks, decided_by, revised_action, revised_subtype)` |
| **Prompt** | `prompts.CRITIC` → `{verdict, reason}` |
| **Fallback** | accept — all four code checks already passed |

**Gate 3.** Corroboration, policy-backing, contradiction and distinct-event checks
are **code**; only proportionality goes to a model. The contradiction check refuses
a sales offer to a customer in distress on grounds of what the action *means*,
regardless of evidence strength.

### HITL · `critique.py::HitlStub`

| | |
|---|---|
| **Input** | `ActionProposal` + `Synthesis` |
| **Output** | `HitlDecision(status, action, action_subtype, note)` |
| **Prompt** | none — a CLI prompt to a human |

**Gate 4.** Any `action != no_action` → `escalated`. Ground truth marks every
intervention in all three scenarios as escalated. `reject` turns the action into
`no_action` in the output; recording `human_rejected` while still emitting the
action would make the audit trail a lie.

---

## Synchronous vs asynchronous

| Flow | Trigger | Path |
|---|---|---|
| **Synchronous** | `EventTick` | event → episodic memory → owning agent → state board. Runs the instant an event arrives, so a cancellation click is visible immediately. |
| **Asynchronous** | `ClockTick`, daily 00:00Z | all agents' `on_tick` → synthesis → action → guardrail → critique → HITL → checkpoint row. This is where absence is detected and where every graded row is produced. |

No decision is ever taken on the synchronous path: perception publishes, and the
daily cycle decides. That is what makes a run reproducible regardless of pacing.

---

## Orchestration · `pipeline.py`

`Pipeline.run()` wires all of the above and returns
`(InferredEventsWriter, RunStats)`. `_on_event` is the synchronous path,
`_on_checkpoint` the asynchronous one. `watch.py` drives these same two methods,
so the terminal view can never diverge from the graded output.
