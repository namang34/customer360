# Evaluation — results, and what they actually prove

Reported honestly, including the failures. `python run_evaluation.py` reproduces
every number below with no API keys and no network.

## Headline

| | scenario_01 | scenario_02 | scenario_03 | overall |
|---|---|---|---|---|
| `inferred_state` | 3/3 | 2/2 | 3/3 | **8/8** |
| `confidence_band` | 3/3 | 2/2 | 3/3 | **8/8** |
| `action` | 3/3 | 2/2 | 3/3 | **8/8** |
| `action_subtype` (where graded) | 1/1 | 1/1 | 1/1 | **3/3** |
| `hitl_status` (where graded) | 1/1 | 1/1 | 1/1 | **3/3** |
| false-positive checks | 2/2 | 1/1 | 1/1 | **4/4** |
| lead-time targets | 1/1 | 1/1 | 2/2 | **4/4** |

221 automated tests pass in ~8 seconds.

---

## The caveat that matters most

**The confidence thresholds were calibrated against these eight checkpoints.**

The mid-term submission listed "precise confidence-band thresholds" as an open
question to be resolved against the practice scenarios, and that is what was done.
Scoring 8/8 on the data used to fit the thresholds is therefore **confirmation
that the calibration was applied correctly, not evidence that it generalises.**

With eight graded points and two thresholds, the overfitting risk is real. What
mitigates it — partially, not completely:

- The rules are **structural rather than numeric**. "Two independent strong
  signals, plus either three corroborating source systems or one deliberate act
  by the customer" is a shape, not a tuned constant. There is no
  `if score > 8.7` anywhere.
- The **guardrail was not fitted at all**. "≥ 2 independent source systems" was
  chosen from the observation that every planted red herring is one isolated
  event on one system. It would have been the rule with zero graded checkpoints.
- Several detector thresholds are set from **domain reasoning**, not from the
  answer key: a $5,000 medical charge is large, a 30% income drop is a cut, a
  self-transfer to another bank differs in kind from paying a third party.

What would actually test generalisation is a fourth scenario, which we do not
have. The honest expectation is that `inferred_state` and the false-positive
behaviour hold up on hidden data, and that `confidence_band` is the field most
likely to slip.

---

## Failures and things that do not work

### 1. The standing-instruction detector never fires on this data

`standing_instruction_stopped` detects a recurring commitment that has silently
stopped. It contributes **nothing** to any of the three scores.

All three scenarios are missing their February standing instructions entirely:

```
scenario_01  mortgage   Nov 1, Dec 1, Jan 1, [   nothing   ], Mar 1, Apr 1
scenario_02  mortgage   Nov 1, Dec 1, Jan 1, [   nothing   ], Mar 1, Apr 1
scenario_03  rent/gym/utilities
                        Nov 1, Dec 1, Jan 1, [   nothing   ], Mar 1
```

That is a data-generation artifact, not behaviour: scenarios 01 and 02 resume
normally on 1 April, and only scenario_03 — who actually cancelled on 4 March —
stops for good. With a one-missed-cycle threshold the detector fired in **all
three** scenarios in mid-February, including the two customers who are not
churning, and pushed scenario_03's 15 February checkpoint to high confidence when
ground truth expects low.

The artifact and the real signal are the same shape — one missed monthly cycle —
so no threshold separates them. The detector now requires **two** missed cycles,
which silences the false positives. The cost is that scenario_03's cancellation
leaves only one missed cycle (1 April) before `simulated_end` on 15 April, so it
never fires at all. Kept because it is correct in principle and would matter on
real data; reported here because pretending otherwise would be dishonest.

Test: `test_standing_instruction_detector_is_silent_on_the_february_artifact`.

### 2. Retrieval runs on a fallback embedder in the sandbox

ChromaDB's default embedding function downloads an ONNX MiniLM model on first
use. Where that download is unavailable, the system falls back to
`HashingEmbedder` — a deterministic bag-of-words embedding with no model.

On this corpus (8 policies, 5 patterns, all short domain text) it retrieves the
correct policy for every query tested. But it has **no semantic generalisation**:
it cannot know that "infant" relates to "baby". A policy corpus using different
vocabulary from the queries would retrieve worse. `SemanticMemory.embedder_name`
records which embedder was used, and every run prints it.

### 3. Early-February state churn in scenario_02

`inferred_state` changes four times in the first three weeks of scenario_02 —
`no_significant_event` → `churn_risk` → `job_loss_or_income_disruption` →
`churn_risk` → `new_child_life_event`. All at **low** confidence, all on two weak
signals, and settled by 19 February — before the first graded checkpoint on the
20th.

This is arguably correct behaviour (the system is genuinely uncertain and updates
as evidence arrives) but it would look unstable to a human watching. Hysteresis
scaled by confidence makes an established belief sticky — 1.5× to displace a
high-confidence conclusion — while leaving low-confidence beliefs free to move.
A stricter margin would have prevented reaching `new_child_life_event` in time
for the 20 February checkpoint.

### 4. Three graded checkpoints per scenario is a very small sample

Eight checkpoints, four false-positive checks and four lead-time targets across
three customers. Every percentage above is out of a single-digit denominator.
100% on eight points is meaningfully different from 100% on eight hundred.

### 5. The LLM path is less tested than the deterministic path

Every test runs with `C360_OFFLINE=1`, so the language models are exercised
manually rather than in CI. The deterministic fallbacks are what the 221 tests
cover. `--live` should produce the same or better results — the LLM only
adjudicates between close candidate states, refines an action already authorised
by retrieved policy, and judges proportionality — but "should" is doing work in
that sentence.

---

## What the system does well, and why

### Every red herring is defeated, structurally

| Scenario | Event | What it is | Must not trigger | Result |
|---|---|---|---|---|
| 01 | `EVT_000382` | $12,000 tuition transfer (ach_wire) | fraud_hold, rm_escalation | clean |
| 01 | `EVT_000402` | $2,500 resort refund (card_payments) | personalized_offer | clean |
| 02 | `EVT_000328` | $600 electronics purchase (card_payments) | fraud_hold | clean |
| 03 | `EVT_000447` | $5,200 tax refund (core_banking_ledger) | personalized_offer | clean |

Each is a single event on a single source system. The guardrail counts
independent source systems, so all four are blocked by the same rule for the same
structural reason — not by four special cases. Three separate defences apply:

1. **Confidence gate** — nothing but `no_action` below high confidence.
2. **Guardrail** — guarded actions need ≥ 2 independent source systems and ≥ 2
   distinct events.
3. **Contradiction check** — a sales offer to a customer in distress or
   disengagement is refused on what it *means*, regardless of evidence strength.

### Late-arriving signal events are handled correctly

Scenarios 01 and 02 each plant exactly one out-of-order event, and in both cases
it is a ground-truth **signal** event:

| Scenario | Event | What | Occurred | Arrived |
|---|---|---|---|---|
| 01 | `EVT_000420` | $450 diagnostics charge | 1 Mar | 3 Mar |
| 02 | `EVT_000341` | Mothercare purchase | 2 Mar | 3 Mar |

Both stay invisible until they arrive and are released at ingestion time. A
system sorting by `event_time` would claim to have known two days early, and every
lead-time figure built on that would be wrong.

### Lead time is met in all four cases

| Scenario | Checkpoint | Ideal | Achieved |
|---|---|---|---|
| 01 | 26 Mar | ≥ 1 day | 5 days |
| 02 | 27 Mar | ≥ 2 days | 9 days |
| 03 | 08 Mar | ≥ 3 days | 3 days |

Scenario_03's three days come from the Usage Agent firing on the
`manage_standing_instructions_cancel` click on 4 March — two days before the money
moves. That is the single clearest argument for the swarm topology: the agent that
saw it first reads a source system the Transaction Agent never touches.

### Cost

74 checkpoints per scenario, of which the large majority carry the previous belief
forward with no reasoning call at all. The LLM is consulted only when the top two
candidate states are within 25% of each other, when an action is actually being
proposed, and when a proposal needs proportionality review — a handful of calls
per scenario rather than 74.

---

## Reproducing

```bash
python -m pytest tests/ -q       # 221 tests, ~8s
python run_evaluation.py         # the table at the top
python run_evaluation.py --live  # same, using Gemini/Groq from .env
python watch.py data/scenario_03 --speed 2
```

Every run writes `out/<scenario>_inferred_events.json` and
`out/<scenario>.trace.jsonl`. The trace carries one record per checkpoint with the
findings, affinity scores, guardrail verdict, critique verdict and HITL routing —
enough to answer "why did it do that on 8 March?" without re-running anything.
