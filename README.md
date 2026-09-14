# Agentic Customer 360 — Proactive Intervention Desk

Inter IIT Tech Meet 15.0 Prepathon. An ambient multi-agent system that watches a
retail banking customer's event streams, infers what is happening in their life,
and decides whether the bank should intervene.

## Results

Scored by `run_evaluation.py` against each scenario's `ground_truth.json`, offline
(no API calls, fully deterministic):

| | scenario_01 | scenario_02 | scenario_03 | overall |
|---|---|---|---|---|
| `inferred_state` | 100% | 100% | 100% | **8/8** |
| `confidence_band` | 100% | 100% | 100% | **8/8** |
| `action` | 100% | 100% | 100% | **8/8** |
| all fields exact | 100% | 100% | 100% | **8/8** |
| false-positive checks | clean | clean | clean | **4/4** |
| lead-time targets | met | met | met | **4/4** |

237 tests pass in under 20 seconds.

Deliverables: `docs/Customer360_MidTerm_Submission.pdf` (mid-term submission: research, architecture, progress) ·
`docs/architecture.svg` (system flow) · `docs/architecture_agents.svg` (agent & tool detail: every tool and the
data source it touches) · `docs/AGENTS.md` (agent register: what each agent
is, what it can do, its exact I/O schema) · `docs/EVALUATION.md` (full evaluation write-up).

Read `docs/EVALUATION.md` for what these numbers do and do not prove — the
confidence thresholds were calibrated against these eight checkpoints, so this is
a calibration result, not independent evidence of generalisation.

## Quick start

```bash
conda activate customer360
pip install -r requirements.txt

python -m pytest tests/ -q          # 237 tests
python run_evaluation.py            # run + score all three scenarios
python watch.py data/scenario_03 --speed 2   # watch it think, ~2.5 min
```

Add `--live` to `run_evaluation.py` or `watch.py` to use the Gemini/Groq models
configured in `.env`. Offline is the default: the deterministic path is what the
tests exercise and what an examiner can reproduce without keys or quota.

## Architecture

![System flow — every component and how work passes between them](docs/architecture.png)

![Agent and tool detail — every agent, its tools, and the data source each tool touches](docs/architecture_agents.png)

Full resolution: [`docs/architecture.svg`](docs/architecture.svg) · [`docs/architecture_agents.svg`](docs/architecture_agents.svg)

### Memory

| Tier | Store | Contents |
|---|---|---|
| Working | in-process | the tick being handled |
| Episodic | SQLite | every event, finding and decision, queried with a **mandatory** `as_of` |
| Semantic | ChromaDB | narrative patterns and bank policy text, incremental upsert |

### The one rule everything rests on

No agent may ever see an event whose `event_time` is later than the current
simulated now. Enforced in two independent places:

- `ReplayEngine` releases events at `max(ingestion_time, event_time)`, so an
  event cannot be released before it happened — the rule holds *by construction*.
- `EpisodicMemory.events_as_of(as_of, ...)` filters on `event_time <= as_of AND
  release_time <= as_of`, with `as_of` as a mandatory positional argument so a
  caller cannot forget it.

A test asserts the two agree at all 74 checkpoints of all three scenarios, and
another asserts the invariant on *every tick* of every run.

## Layout

```
src/c360/
  schema.py       events, enums, tolerant parsing
  clock.py        simulated clock, monotonic, pacing separate from semantics
  replay.py       the replay engine
  output.py       the graded file writer -- refuses to emit an invalid row
  pii.py          redaction (tokenising, not deleting)
  memory.py       episodic memory (SQLite)
  findings.py     what a perception agent publishes
  state_board.py  the shared swarm surface + corroboration arithmetic
  agents/         transaction, usage, support, life_signal
  synthesis.py    correlation: state + confidence
  semantic.py     ChromaDB policies and patterns
  action.py       RAG-grounded action proposer
  guardrail.py    the >= 2 independent source systems rule
  critique.py     adversarial review + HITL stub
  prompts.py      every LLM prompt in the system, in one file
  tracing.py      local JSONL trace + LangSmith
  pipeline.py     the orchestrator
  review.py       the ambiguity review queue -- escalation for uncertainty
  scoring.py      the evaluation harness
run_replay.py       step-1 smoke test
run_pipeline.py     replay -> decision -> file
run_evaluation.py   run + score all scenarios
watch.py            terminal visibility layer
docs/               diagram, agent register, solution document, evaluation write-up
```

## Non-negotiables

- **Explainability** — the output writer refuses to construct any row proposing an
  action unless its `notes` cite at least one `EVT_…`. A code check, not a prompt.
- **Traceability** — one JSONL trace record per checkpoint carrying the findings,
  affinity scores, guardrail verdict, critique verdict and HITL routing, tagged
  with `customer_id` and `as_of`. LangSmith is enabled on top when a key is set.
- **PII** — identifiers are replaced with stable tokens before any prompt or log
  line. Tokens rather than deletion, so `"<CUSTOMER_NAME> - Chase Bank"` still
  reads as a self-transfer. A test greps the real output and trace for the
  customer's name.
- **Guardrails** — every check in `guardrail.py` is arithmetic and set membership.
  No model is consulted and nothing can be talked out of its answer.
