# Research Log — Agentic Customer 360

Naman Goyal · Inter IIT Tech Meet 15.0 Prepathon · Natural Language Processing

Everything read while designing and building this system, with what each source gave me and which
part of the code it changed. Ordered by how much it actually mattered, not by date.

Key files in the codebase:

| | |
|---|---|
| System flow diagram | [`docs/architecture.svg`](docs/architecture.svg) |
| Agent & tool detail diagram | [`docs/architecture_agents.svg`](docs/architecture_agents.svg) |
| Mid-term write-up (report + architecture) | [`docs/Customer360_MidTerm_Submission.pdf`](docs/Customer360_MidTerm_Submission.pdf) |
| Evaluation, including what does not work | [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| Agent register — I/O schema per agent | [`docs/AGENTS.md`](docs/AGENTS.md) |

---

## 1. Read and applied

These changed a decision that is visible in the code.

### Akidau, "Streaming 101: The world beyond batch"
<https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/> · O'Reilly Radar
Sequel: ["Streaming 102"](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/) for
watermarks and allowed lateness.

**Takeaway.** Event time is when something occurred; processing time is when the system observed it.
The skew between them is "not only non-zero, but often a highly variable function" of the source and
the pipeline. So a correctness claim about a window has to be stated in event time, even though
records physically arrive in processing-time order. Watermarks estimate how far event time has
advanced; late data is normal, not an error.

**What it changed.** The whole two-timestamp design. `event_time` vs `ingestion_time` in
[`src/c360/schema.py`](src/c360/schema.py) is event time vs processing time; `release_time = max(ingestion_time,
event_time)` is the watermark, so an event cannot be released before it happened. It also produced
the second clause of the visibility filter in [`src/c360/memory.py`](src/c360/memory.py) — `event_time <= as_of AND
release_time <= as_of` — which is what stops a late-arriving event appearing to have been known
before it reached the bank. Two of the 298 live events in the provided stream arrive late, and both
are ground-truth signal events.

### Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023)
<https://arxiv.org/abs/2304.03442> · §4, Agent Architecture

**Takeaway.** Agents retrieve from a memory stream scored on recency, importance and relevance.
Separately, *reflection* synthesises raw observations into higher-level conclusions the agent then
reasons over instead of re-deriving them. The value is in promoting observations into stable
conclusions, not in storing more.

**What it changed.** The working / episodic / semantic split, and specifically the state board:
perception agents publish structured `Finding` objects ([`src/c360/findings.py`](src/c360/findings.py)) and
[`src/c360/synthesis.py`](src/c360/synthesis.py) reads only those, never the event stream. That is the reflection layer here.
It also justifies `StateBoard.current_state()` carrying a committed inference forward rather than
recomputing it daily.

### Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023)
<https://arxiv.org/abs/2310.08560> · with the Letta docs on tiered memory, <https://docs.letta.com>

**Takeaway.** Hierarchical memory borrowed from operating systems — a small in-context tier plus a
larger external store, with movement between them an explicit operation rather than an invisible
mechanism. The argument is less about capacity than about making retrieval addressable, and
therefore constrainable.

**What it changed.** Episodic memory as an explicitly queried SQLite store rather than context
stuffing — and, because every read is an explicit call, `as_of` could be made a mandatory positional
argument with no default. A caller physically cannot query memory without stating what "now" is.

### LangChain, "LangGraph: Multi-Agent Workflows"
<https://www.langchain.com/blog/langgraph-multi-agent-workflows> · with the LangGraph multi-agent
concepts docs, <https://docs.langchain.com/oss/python/langchain/multi-agent>

**Takeaway.** Compares collaboration, supervisor and hierarchical topologies. What separates them is
whether agents see each other's work in progress: under collaboration "all the work either of them
do is visible to the other", whereas a supervisor gives each agent "their own independent
scratchpads" with only final outputs reaching shared state.

**What it changed.** A swarm at Layer 1 writing to a shared board for outputs only, no
agent-to-agent messaging — made concrete by constructing each agent's `PerceptionContext`
([`src/c360/agents/base.py`](src/c360/agents/base.py)) *without* a reference to the board. If Support could see Usage's finding,
one piece of evidence would be counted twice while appearing to be two independent witnesses, and
corroboration is the guardrail's whole basis.

### Anthropic, "Building Effective Agents"
<https://www.anthropic.com/engineering/building-effective-agents>

**Takeaway.** Recommends "finding the simplest solution possible, and only increasing complexity when
needed", and warns that "agentic systems often trade latency and cost for better task performance".
Its evaluator-optimizer pattern — one call generates, a second evaluates in a loop — is the shape
that fits a bounded quality gate.

**What it changed.** Rejecting multi-agent debate. One adversarial reviewer with a single bounded
retry ([`src/c360/critique.py`](src/c360/critique.py), `max_retries=1`) instead. It also informed keeping the confidence
arithmetic and the guardrail in code — only genuine judgement is asked of a model.

### "The hidden cost of AML: how 95% false positives hurt banks, fintechs, and customers"
<https://www.retailbankerinternational.com/comment/hidden-cost-of-aml-how-false-positives-hurt-banks-fintechs-customers/>
· Retail Banker International

**Takeaway.** Traditional AML transaction monitoring runs at roughly a **95% false-positive rate**.
Regulators assume about two hours to file a SAR; independent studies put the real burden at up to
**22 hours per alert**. Global AML compliance spend is estimated over **$274bn annually**, much of it
on low-quality alerts. The customer-side cost is the part that matters here: people face "intrusive
verifications, frozen accounts, or blocked transactions triggered by nothing more than an atypical,
but entirely lawful, transaction."

**What it changed.** Grounded the asymmetry in [`src/c360/guardrail.py`](src/c360/guardrail.py). `compliance_fraud_hold`,
`relationship_manager_escalation` and `personalized_offer` require corroboration from ≥ 2 independent
source systems; `support_intervention` and `proactive_retention_outreach` do not. A support call to
someone who did not need one costs almost nothing — a wrongly frozen account is one of the harms
above. Before this the asymmetry was an assertion; now it has a number behind it.

### ChromaDB documentation — collections, upsert, custom embedding functions
<https://docs.trychroma.com>

**Takeaway.** `upsert` writes or updates by document id, so re-seeding a collection is idempotent and
editing one document re-embeds only that document. Embedding functions are pluggable, and the default
downloads a model on first use.

**What it changed.** Content-hashed incremental upsert in [`src/c360/semantic.py`](src/c360/semantic.py), so the policy store
stays live without a full re-index. The first-use download is why there is a dependency-free fallback
embedder — a sandboxed evaluation run with no network still completes end to end.

---

## 2. Domain background — existing approaches to this problem

Surveyed to understand how banks actually do next-best-action today, and what the agentic version is
replacing. Skimmed rather than studied; they shaped framing rather than a specific line of code.

- [Unified data for next best action in banking](https://www.backbase.com/blog/next-best-action-in-banking) — Backbase
- [How Discovery Bank delivers hyper-personalized banking at scale: behavioral AI, governed data, and real-time decisioning](https://www.databricks.com/blog/how-discovery-bank-delivers-hyper-personalized-banking-scale-behavioral-ai-governed-data-and) — Databricks
- [Next Best Action explained](https://evam.com/blog/next-best-action-engine) — Evam
- [Next Best Action](https://cdp.com/glossary/next-best-action/) — CDP.com glossary
- [Understanding false positives in transaction monitoring](https://www.flagright.com/post/understanding-false-positives-in-transaction-monitoring) — Flagright
- [How to reduce false positives in AML transaction monitoring](https://www.unit21.ai/blog/reduce-false-positives-in-aml-transaction-monitoring) — Unit21

The common shape: a customer data platform assembles a unified profile, a rules or ML engine scores
eligibility, and a human or campaign tool sends the message. It is a *pull* architecture — something
has to ask. The gap this PS names is that nobody is continuously watching, which is the whole
argument for ambient agents.

---

## 3. Tooling and API documentation consulted

Reference material, read as needed rather than studied.

- LangChain Google GenAI integration — <https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai>
- LangChain Groq integration — <https://docs.langchain.com/oss/python/integrations/chat/groq>
- LangSmith tracing — <https://docs.langchain.com/langsmith/observability>
- Python `sqlite3` — <https://docs.python.org/3/library/sqlite3.html>
- Python `dataclasses` (frozen dataclasses for immutable events) — <https://docs.python.org/3/library/dataclasses.html>
- pytest — <https://docs.pytest.org/>

---

## 4. Still to read

Identified as relevant, not yet worked through. Listed so the gap is visible rather than hidden.

- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) — <https://arxiv.org/abs/2303.11366>
- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (2023) — <https://arxiv.org/abs/2303.17651>
- Weng, Lilian, "LLM Powered Autonomous Agents" — <https://lilianweng.github.io/posts/2023-06-23-agent/>
- Anthropic, "Effective context engineering for AI agents" — <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>

---

## 5. Where each source landed, at a glance

| Source | Code it shaped |
|---|---|
| Akidau, Streaming 101/102 | `schema.py` two timestamps · `replay.py` release order · `memory.py` two-clause filter |
| Park et al., Generative Agents | `findings.py` · `state_board.py` · `synthesis.py` reads findings not events |
| Packer et al., MemGPT / Letta | `memory.py` — external store, `as_of` mandatory with no default |
| LangGraph multi-agent | `agents/base.py` — `PerceptionContext` excludes the state board |
| Anthropic, Building Effective Agents | `critique.py` — one reviewer, `max_retries=1`; arithmetic stays in code |
| AML false-positive cost | `guardrail.py` — which actions are guarded and which are not |
| ChromaDB docs | `semantic.py` — content-hashed upsert, fallback embedder |
