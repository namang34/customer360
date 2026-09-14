// Rebuilds Customer360_MidTerm_Submission.docx
// Same house style as build_solution_doc.js so the two documents match.
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  Footer, PageNumber, ImageRun, PageOrientation,
} = require("docx");
const fs = require("fs");

const ACCENT = "0F4C5C";
const MUTED = "50646B";
const CONTENT_W = 9360; // 12240 (Letter) - 2*1440 margins
const BODY = 19;

const p = (text, o = {}) => new Paragraph({
  spacing: { before: o.before ?? 0, after: o.after ?? 90, line: o.line ?? 250 },
  alignment: o.align,
  indent: o.indent,
  border: o.border,
  children: [new TextRun({
    text, size: o.size ?? BODY, color: o.color, bold: o.bold, italics: o.italics,
    font: o.font ?? "Calibri",
  })],
});

const rp = (parts, o = {}) => new Paragraph({
  spacing: { before: o.before ?? 0, after: o.after ?? 90, line: o.line ?? 250 },
  alignment: o.align,
  indent: o.indent,
  children: parts.map(([text, s = {}]) => new TextRun({
    text, size: o.size ?? BODY, font: "Calibri",
    bold: s.bold, italics: s.italics, color: s.color,
  })),
});

const h1 = (text) => new Paragraph({
  spacing: { before: 260, after: 100 },
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, size: 26, bold: true, color: ACCENT, font: "Calibri" })],
});

const h2 = (text) => new Paragraph({
  spacing: { before: 170, after: 60 },
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, size: 20, bold: true, color: "1A2327", font: "Calibri" })],
});

const bullet = (parts, o = {}) => new Paragraph({
  spacing: { after: 56, line: 250 },
  bullet: { level: 0 },
  children: (Array.isArray(parts) ? parts : [[parts]]).map(([text, s = {}]) => new TextRun({
    text, size: o.size ?? BODY, font: "Calibri", bold: s.bold, italics: s.italics, color: s.color,
  })),
});

const cell = (content, { w, bold, color, shade, size = 17, align } = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
  margins: { top: 50, bottom: 50, left: 95, right: 95 },
  children: (Array.isArray(content) ? content : [content]).map((line, idx) =>
    new Paragraph({
      spacing: { after: idx === (Array.isArray(content) ? content.length - 1 : 0) ? 0 : 40, line: 230 },
      alignment: align,
      children: [new TextRun({ text: line, size, bold, color, font: "Calibri" })],
    })),
});

const table = (widths, rows, { headerShade = "E8EEF0" } = {}) => new Table({
  columnWidths: widths,
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  borders: {
    top: { style: BorderStyle.SINGLE, size: 2, color: "B6C4C9" },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: "B6C4C9" },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "D6DFE2" },
    insideVertical: { style: BorderStyle.NONE },
  },
  rows: rows.map((cells, i) => new TableRow({
    tableHeader: i === 0,
    children: cells.map((c, j) => cell(c, {
      w: widths[j],
      bold: i === 0,
      shade: i === 0 ? headerShade : undefined,
      color: i === 0 ? ACCENT : undefined,
    })),
  })),
});

const spacer = (h = 80) => new Paragraph({ spacing: { after: h }, children: [] });
const rule = () => new Paragraph({
  spacing: { before: 40, after: 140 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "D6DFE2" } },
  children: [],
});

const children = [];

// ============================================================ COVER
children.push(new Paragraph({
  spacing: { after: 30 },
  children: [new TextRun({
    text: "Agentic Customer 360 — Proactive Intervention Desk",
    size: 32, bold: true, color: ACCENT, font: "Calibri",
  })],
}));
children.push(p("Mid-Term Submission — Research Log, Preliminary Architecture & Approach",
  { size: 22, color: MUTED, after: 40 }));
children.push(p("Inter IIT Tech Meet 15.0 · Prepathon · Natural Language Processing",
  { size: 18, color: MUTED, after: 130 }));
children.push(rp([["Submitted by: ", { color: MUTED }], ["Naman Goyal", { bold: true }]], { after: 20 }));
children.push(rp([["Date: ", { color: MUTED }], ["15 September 2026", { bold: true }]], { after: 70 }));
children.push(p(
  "The MAS topology, memory scoping and trigger design in Part C were settled before any code existed, and "
  + "Part B records what informed each one. AI tooling was used to implement and debug that design and to "
  + "calibrate thresholds against the practice scenarios — not to choose it. Where the design did change "
  + "during the build it changed in response to what the data showed, and those changes are recorded in the "
  + "evaluation write-up.",
  { italics: true, size: 18, color: MUTED, after: 60 }));
children.push(rule());

// ============================================================ PART A
children.push(h1("Part A — Findings & Approach Report"));

children.push(h2("What I understand about the domain"));
children.push(p(
  "The challenge here is not throughput or data volume — the streams provided (~500 events per customer "
  + "over ~2.5 months) are small. It is temporal judgement: withholding a decision while a signal is still "
  + "weak, then committing within a narrow window once it is strong enough — without being fooled by "
  + "isolated anomalous events (“red herrings”) that resemble a different life event or risk category."));
children.push(p(
  "A second, quieter difficulty sits underneath it. Each event carries two timestamps: when it happened, and "
  + "when the bank's systems learned of it. Any claim to have inferred something “early” is only meaningful "
  + "if the system provably could not have seen the future — including data that had occurred but had not yet "
  + "arrived. That is a stream-processing problem rather than a retrieval one, and it shapes the memory "
  + "design more than anything else."));
children.push(p(
  "So this is an agent-coordination and memory-scoping problem more than a data-engineering one — consistent "
  + "with the problem statement's own framing (§1.3: “this is a genuinely unfamiliar problem space… reading "
  + "how others have approached these problems is a load-bearing part of the work”)."));

children.push(h2("Approach I am following"));
children.push(bullet([["Multi-layer MAS. ", { bold: true }], ["A parallel (“swarm”) perception layer of "
  + "specialised agents (Usage, Transaction, Support, Life-Signal/KYC) feeding a single Synthesis agent, "
  + "followed by a Critique-Refiner pass before any action is finalised."]]));
children.push(bullet([["Per-customer shared state board ", { bold: true }], ["rather than siloed agent memory or "
  + "agent-to-agent messaging — the single surface each perception agent writes structured findings to and "
  + "the correlation layer reads from."]]));
children.push(bullet([["Three-tier memory. ", { bold: true }], ["Working (the tick being handled), episodic "
  + "(per-customer event history in a queryable store, strictly time-filtered), semantic (shared policy and "
  + "pattern knowledge in a vector store)."]]));
children.push(bullet([["A hard, code-level guardrail ", { bold: true }], ["requiring at least two independent "
  + "source_systems to corroborate before any escalation, offer or fraud-hold action — aimed directly at the "
  + "red-herring pattern present in all three provided scenarios."]]));
children.push(bullet([["Model routing. ", { bold: true }], ["A fast model for high-volume perception, a stronger "
  + "reasoning model for Synthesis and Action — a deliberate cost/quality tradeoff, not one model used "
  + "everywhere. Every model call has a deterministic fallback, so the system runs end to end with no API "
  + "access at all."]]));

children.push(h2("Progress at mid-term"));
children.push(p(
  "The architecture in Part C is fixed, and every component in it is built and integrated — ingestion through "
  + "to the inferred-events writer — with a local scoring harness and a terminal view running over the same "
  + "pipeline the graded run uses. Remaining work before the end-term submission: calibration reporting, one "
  + "recorded run against the live models, and extending the research log below."));

// ============================================================ PART B
children.push(h1("Part B — Research Log"));
children.push(p(
  "Not a bibliography — every row is here because it changed a decision that is visible in the system.",
  { italics: true, color: MUTED }));

children.push(table([1950, 1150, 3300, 2960], [
  ["Source", "Type", "Key takeaway", "Decision it informed"],

  [["Akidau, \u201CStreaming 101 / 102: The world beyond batch\u201D", "(O\u2019Reilly Radar)"],
   "Engineering write-up",
   "Event time is when something occurred; processing time is when the system observed it. The skew between "
   + "them is \u201Cnot only non-zero, but often a highly variable function\u201D of the source and the pipeline, so "
   + "correctness has to be stated in event time even though records physically arrive in processing-time "
   + "order. Watermarks estimate how far event time has advanced; late data is the normal case, not an error.",
   "The two-timestamp design. release_time = max(ingestion_time, event_time) is our watermark: an event "
   + "cannot be released before it happened, so the no-leakage rule holds by construction rather than by test. "
   + "It also gives the visibility filter its second clause, release_time \u2264 as_of, which stops a late-arriving "
   + "event appearing to have been known before it reached the bank."],

  ["Park et al., \u201CGenerative Agents: Interactive Simulacra of Human Behavior\u201D (2023), \u00A74",
   "Research paper",
   "Agents retrieve from a memory stream scored on recency, importance and relevance. Separately, reflection "
   + "synthesises raw observations into higher-level conclusions the agent then reasons over instead of "
   + "re-deriving them. The value is in promoting observations into stable conclusions, not in storing more.",
   "The working / episodic / semantic split, and the state board specifically: perception publishes structured "
   + "findings and Synthesis reads only those, never the event stream \u2014 that is the reflection layer here. It "
   + "also justifies carrying a committed inference forward rather than recomputing it each day."],

  ["Packer et al., \u201CMemGPT: Towards LLMs as Operating Systems\u201D (2023), and the Letta docs on tiered memory",
   "Research paper / engineering docs",
   "Hierarchical memory borrowed from operating systems: a small in-context tier plus a larger external store, "
   + "with movement between them an explicit operation rather than an invisible mechanism. The argument is less "
   + "about capacity than about making retrieval addressable, and therefore constrainable.",
   "Episodic memory as an explicitly queried SQLite store rather than context stuffing \u2014 and, because every "
   + "read is an explicit call, as_of could be made a mandatory positional argument with no default. A caller "
   + "physically cannot query memory without stating what \u201Cnow\u201D is."],

  [["LangChain \u2014 \u201CLangGraph: Multi-Agent Workflows\u201D", "and the LangGraph multi-agent concepts docs"],
   "Official blog / docs",
   "Compares collaboration, supervisor and hierarchical topologies. What separates them is whether agents see "
   + "each other\u2019s work in progress: under collaboration \u201Call the work either of them do is visible to the "
   + "other\u201D, whereas a supervisor gives each agent \u201Ctheir own independent scratchpads\u201D with only final "
   + "outputs reaching shared state.",
   "A swarm at Layer 1 writing to a shared board for outputs only, with no agent-to-agent messaging \u2014 made "
   + "concrete by constructing each agent\u2019s context without a reference to the board. If Support could see "
   + "Usage\u2019s finding, one piece of evidence would be counted twice while appearing to be two independent "
   + "witnesses, and corroboration is the guardrail\u2019s whole basis."],

  ["Anthropic \u2014 \u201CBuilding Effective Agents\u201D",
   "Engineering blog",
   "Recommends \u201Cfinding the simplest solution possible, and only increasing complexity when needed\u201D, and "
   + "warns that \u201Cagentic systems often trade latency and cost for better task performance\u201D. Its "
   + "evaluator-optimizer pattern \u2014 one call generates, a second evaluates in a loop \u2014 is the shape that fits "
   + "a bounded quality gate.",
   "Rejecting multi-agent debate: with one customer and six permitted actions it costs latency and free-tier "
   + "quota without buying accuracy, so one adversarial reviewer with a single bounded retry is enough. It also "
   + "informed keeping the confidence arithmetic and the guardrail in code \u2014 only genuine judgement is asked "
   + "of a model."],

  ["ChromaDB documentation \u2014 collections, upsert, custom embedding functions",
   "Official docs",
   "upsert writes or updates by document id, so re-seeding a collection is idempotent and editing one document "
   + "re-embeds only that document. Embedding functions are pluggable, and the default downloads a model on "
   + "first use.",
   "Content-hashed incremental upsert, so the policy store stays live without a full re-index. The first-use "
   + "download is why there is a dependency-free fallback embedder: a sandboxed evaluation run with no network "
   + "still completes end to end."],
], { headerShade: "E8EEF0" }));

// ============================================================ PART C
children.push(h1("Part C — System Architecture"));

children.push(h2("1. Problem framing, grounded in the provided scenarios"));
children.push(p(
  "Three scenarios were provided as practice data, each one customer's event timeline (~500 events across "
  + "history_seed.jsonl and live_stream.jsonl), graded at fixed checkpoints against a hidden ground truth:"));
children.push(table([1150, 1450, 2700, 1200, 1700, 1160], [
  ["Scenario", "Customer", "True narrative", "Checkpoints", "Final action", "Red herring(s)"],
  ["1 — Medical hardship", "Marcus Vance (CUST_00088)",
   "ER visit → income drop (benefits) → hospital bill → savings drawn down → support ticket",
   "low → medium → high", "support_intervention (escalated)", "Tuition transfer; resort refund"],
  ["2 — New child", "Priya Sharma (CUST_00105)",
   "Income dip → baby purchases → daycare standing instruction → KYC dependents change → search intent",
   "low → high (no medium)", "personalized_offer (escalated)", "Baby monitor purchase"],
  ["3 — Churn risk", "David Chen (CUST_00184)",
   "Denied complaint → usage drop → cancelled standing instructions + transfer out → zero card use → salary swept out",
   "low → high → high (2nd action)",
   "relationship_manager_escalation, later proactive_retention_outreach", "Tax refund deposit"],
]));
children.push(spacer(90));
children.push(p(
  "The common pattern across all three: every red herring is a single isolated event, uncorroborated by any "
  + "other source_system in the same window, while every genuine narrative shows up across several. That "
  + "observation is what the guardrail in §6 is built on."));

children.push(h2("2. The environment the agents live in"));
children.push(p(
  "Ambient agents need somewhere to be ambient. Here that is the provided event stream, replayed as a live "
  + "environment rather than read as a file: history_seed.jsonl is loaded once as backstory, and "
  + "live_stream.jsonl is released one event at a time in the order a real system would have received them. "
  + "Across the three practice scenarios that is 1,367 events from nine source systems over a 74-day window "
  + "(1 February to 15 April 2026)."));
children.push(p(
  "Release order is ingestion_time; judgement is event_time. An event is released at "
  + "max(ingestion_time, event_time), so it can never surface before it happened. Two of the 298 live events "
  + "arrive late — after the day they occurred — and both are ground-truth signal events, which is why "
  + "visibility is filtered on release_time as well as event_time."));
children.push(p(
  "Three trigger types drive the agents, and the system uses all three:"));
children.push(bullet([["Event-based. ", { bold: true }], ["An event arrives and is handed straight to the one "
  + "agent that owns its source system. This is the synchronous path, and it is why a cancellation click is "
  + "visible the moment it lands."]]));
children.push(bullet([["Time-based. ", { bold: true }], ["A daily 00:00Z boundary wakes every agent whether or "
  + "not anything arrived. This is how absence is detected — a salary that did not land, logins that stopped "
  + "— none of which produces an event of its own."]]));
children.push(bullet([["Agent-dependent. ", { bold: true }], ["Synthesis fires only once two or more distinct "
  + "perception agents have flagged something in the same window, so correlation is attempted when there is "
  + "something to correlate rather than on every raw event."]]));
children.push(p(
  "The loop closes back into the environment. Each daily cycle writes a decision row to inferred_events.json "
  + "and commits it to episodic memory, and the next day's cycle reads that decision back as the belief it "
  + "starts from. Agents read from the stream and act into a record the next tick can see — which is what "
  + "makes an inference persist rather than be recomputed from nothing each day."));

children.push(h2("3. Agent roster"));
children.push(p("Drawn in full in Appendices A and B.", { italics: true, color: MUTED }));
children.push(table([1600, 1450, 3300, 3010], [
  ["Component", "Layer", "Reads", "Trigger"],
  ["Usage Agent", "Perception (swarm)", "web_app_events", "Event-based and daily time-based"],
  ["Transaction Agent", "Perception (swarm)",
   "card_payments, core_banking_ledger, instant_payments, ach_wire, trading_brokerage",
   "Event-based and daily time-based"],
  ["Support Agent", "Perception (swarm)", "support_logs", "Event-based"],
  ["Life-Signal Agent", "Perception (swarm)", "loan_kyc, social_signal_consented (consent-gated)",
   "Event-based"],
  ["Synthesis Agent", "Correlation", "The state board — structured findings only, never raw events",
   "Agent-dependent: ≥ 2 distinct perception agents flagged in the window"],
  ["Action Proposer", "Decision", "Synthesis output, retrieved bank policy, prior decisions",
   "Downstream of Synthesis"],
  ["Guardrail", "Safety (code, not an agent)", "The proposed action and its corroboration",
   "Every proposal, before the Critique Agent"],
  ["Critique Agent", "Quality gate", "The proposal, the diagnosis and the guardrail verdict",
   "Only when an action other than no_action is proposed"],
  ["HITL stub", "Human checkpoint", "The final action and the full reasoning trace",
   "Any action ≠ no_action"],
]));

children.push(h2("4. MAS topology, by stage"));
children.push(table([1900, 1900, 5560], [
  ["Stage", "Pattern", "Justification"],
  ["Signal gathering (Layer 1)", "Swarm — parallel, independent",
   "Four perception agents read disjoint source_systems with no interdependency, so they run in parallel at "
   + "no cost to accuracy. Each is constructed without access to the state board, which is what makes their "
   + "findings independent rather than merely concurrent."],
  ["Correlation (Layer 2)", "Handoff to a single orchestrator",
   "Synthesis gains nothing from running in parallel with itself; it needs the swarm's combined output, so "
   + "it sits downstream of Layer 1 on an agent-dependent trigger."],
  ["Action quality gate (Layer 3)", "Critique-Refiner",
   "Targets the false-positive risk that is explicitly graded in all three scenarios. One bounded retry, "
   + "not an open loop: on a free tier an unbounded critique exhausts quota early in a long replay and "
   + "leaves the remainder running with no model at all."],
  ["Debate", "Considered and rejected",
   "Debate earns its place where genuine disagreement between agents needs surfacing rather than averaging "
   + "away. Here that disagreement is already surfaced and settled by the affinity and hysteresis rule in §5, "
   + "at no latency cost. A wider action set, or several customers in scope at once, would change the answer."],
]));

children.push(h2("5. Memory architecture"));
children.push(table([1450, 1600, 1900, 4410], [
  ["Tier", "Scope", "Storage", "Contents"],
  ["Working", "One in-flight decision", "In-process", "The tick currently being handled"],
  ["Episodic", "Per customer, persistent", "SQLite, keyed by customer_id",
   "Every event, finding and committed decision, timestamped; read only through a query carrying a "
   + "mandatory as_of, filtered on event_time ≤ as_of AND release_time ≤ as_of"],
  ["Semantic", "Cross-customer, shared", "ChromaDB",
   "Narrative pattern descriptions per inferred_state, and bank policy text used to ground the proposed "
   + "action and its subtype"],
]));
children.push(spacer(90));
children.push(p(
  "As the problem statement requires, an inferred life-event state persists across checkpoints rather than "
  + "being recomputed — it must still be available, and correctly weighted, weeks later."));
children.push(p(
  "Rolling statistics — login frequency, spend baselines — are maintained against the stream as events "
  + "arrive rather than recomputed at query time, and the semantic store is updated by content-hashed upsert "
  + "rather than a full re-index, so editing one policy re-embeds one document."));

children.push(h2("6. Guardrails, traceability and HITL"));
children.push(bullet([["The hard rule is code-level, not prompt-level. ", { bold: true }],
  ["No compliance_fraud_hold, relationship_manager_escalation or personalized_offer may fire unless at least "
  + "two independent source_systems corroborate the finding within the window, across at least two distinct "
  + "events. Every check is arithmetic and set membership; no model is consulted, so nothing can be talked "
  + "out of its answer."]]));
children.push(bullet([["The bar matches the cost of being wrong. ", { bold: true }],
  ["support_intervention and proactive_retention_outreach are deliberately unguarded. A support call to a "
  + "customer who did not need one costs almost nothing; a frozen account, a mistimed sales offer or an "
  + "unnecessary escalation are real harms. One threshold for everything would either block the cheap "
  + "interventions for no benefit or set the bar too low for the expensive ones."]]));
children.push(bullet([["Disagreement is resolved by a stated rule, never silently. ", { bold: true }],
  ["Perception agents routinely support different states from the same window. Candidate states are scored by "
  + "weighted signal affinity, and an established belief is displaced only if a challenger beats it by a "
  + "margin that scales with how well-evidenced that belief was. When the top two candidates fall within 25% "
  + "of each other the tie goes to the Synthesis model to adjudicate; with no model available the "
  + "deterministic winner stands."]]));
children.push(bullet([["The guardrail runs before the critique pass, ", { bold: true }],
  ["not after. The cheap deterministic check that can definitively reject should precede the expensive "
  + "judgement call that might not."]]));
children.push(bullet([["Any action other than no_action defaults to escalated, ", { bold: true }],
  ["matching the pattern in all three scenarios' ground truth. A rejection by a reviewer turns the action "
  + "into no_action in the output rather than merely relabelling it."]]));
children.push(bullet([["Traceability: ", { bold: true }],
  ["one structured trace record per checkpoint carrying the findings, the affinity scores, the guardrail "
  + "verdict, the critique verdict and the HITL routing, tagged with customer_id and as_of."]]));
children.push(bullet([["Explainability is enforced structurally: ", { bold: true }],
  ["an output row proposing an action cannot be constructed at all unless its notes cite at least one "
  + "event id. A code check, not an instruction in a prompt."]]));
children.push(bullet([["PII is tokenised rather than deleted ", { bold: true }],
  ["before any prompt or log line, so that a self-transfer to another bank still reads as a self-transfer "
  + "once the name is masked."]]));

children.push(h2("7. Design questions, and where they landed"));
children.push(table([3100, 6260], [
  ["Question", "Resolution"],
  ["Episodic-memory relevance and decay (§4.1): how far back should Synthesis look before an old flag is "
   + "stale, and when is a past memory relevant rather than noise?",
   "A 30-day correlation window for findings, chosen against the spacing of the practice narratives; a "
   + "60-day confidence decay for a belief with no supporting evidence at all, set longer than any gap "
   + "inside a graded window so it exists for the hidden set rather than being fitted to this one; and a "
   + "monotonic rule so an established belief does not drop a band because one quiet fortnight thinned "
   + "the window. Relevance is handled separately from recency: confidence and corroboration are computed "
   + "only over findings that actually support the state being claimed, so an unrelated event inside the "
   + "window cannot strengthen a narrative it says nothing about."],
  ["Memory bleed-through between customers (§4.1): how is one customer's history kept out of another's "
   + "inference?",
   "The store is keyed by customer_id and every read carries it, so isolation is a property of the query "
   + "rather than a convention. At this scale each replay runs against its own store; the same predicate is "
   + "what would scope a shared one."],
  ["Precise confidence-band thresholds — what quantifies low, medium and high?",
   "Two distinct strong signals AND either three independent source systems, or two source systems plus one "
   + "signal that is decisive — meaning the customer did something deliberate (filed a change, clicked "
   + "cancel, moved money to their own account elsewhere) rather than something we merely measured about "
   + "them. These were calibrated against the eight graded practice checkpoints, so scoring well on those "
   + "shows the calibration was applied, not that it generalises."],
  ["Escalation for ambiguity rather than cost (§6.3): should low confidence, or an unresolved disagreement, "
   + "reach a human even when no action is proposed?",
   "Yes, through a second channel. The confidence gate turns low confidence into no_action, which is "
   + "auto-approved — right for the graded field, but it meant the system was silent exactly where it was "
   + "least sure. A review queue now raises a checkpoint when several independent source systems agree but "
   + "not strongly enough to act, or when the top two candidate states score within 25% of each other. It "
   + "is written beside the graded file and never alters it, because hitl_status is itself graded. It "
   + "raises on change rather than persistence: a first version that flagged every qualifying day produced "
   + "38 items across scenario_01's 74 checkpoints, which is alert fatigue rather than oversight. It now "
   + "yields 2, 4 and 1."],
  ["Whether to build a local scoring harness for the brownie-point credit in §8.1",
   "Built. It scores the inferred-events output against each scenario's ground_truth.json on state, "
   + "confidence, action, subtype and HITL status. It also checks lead time against the ideal action date, "
   + "and that no action fires inside a red-herring window."],
]));

// ============================================================ DOC
const plate = (heading, caption, file, w, h, pageBreakBefore) => [
  new Paragraph({
    pageBreakBefore,
    spacing: { after: 50 },
    children: [new TextRun({ text: heading, size: 24, bold: true, color: ACCENT, font: "Calibri" })],
  }),
  p(caption, { italics: true, size: 16, color: MUTED, after: 90 }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h } })],
  }),
];

const diagram = [
  ...plate(
    "Appendix A — System Flow",
    "Every component, and how work passes between them. Solid lines are synchronous, taken the instant an "
    + "event arrives; dashed lines are asynchronous, on the daily 00:00Z cycle. Full resolution: docs/architecture.svg",
    "architecture.png", 720, 548, false),
  ...plate(
    "Appendix B — Agent & Tool Detail",
    "Each agent's toolset, and the data source every tool reads from or writes to. "
    + "Full resolution: docs/architecture_agents.svg",
    "architecture_agents.png", 720, 552, true),
];

const doc = new Document({
  styles: { default: { document: { run: { font: "Calibri", size: BODY } } } },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED, font: "Calibri" })],
        })],
      }),
    },
    children,
  }, {
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.LANDSCAPE },
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
      },
    },
    children: diagram,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2] || "Customer360_MidTerm_Submission.docx", buf);
  console.log("written");
});
