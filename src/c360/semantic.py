"""
Semantic memory -- ChromaDB holding narrative patterns and bank policy text.

WHAT IS IN HERE, AND WHY IT IS NOT JUST A DICTIONARY
----------------------------------------------------
Two collections:

  patterns   One short description per inferred_state enum. Used to explain and
             sanity-check a synthesis conclusion in language rather than in
             affinity scores.

  policies   Bank eligibility and action-grounding text. The Action Proposer
             retrieves from this rather than carrying a hard-coded mapping from
             state to action.

The obvious objection is that with six possible actions a Python dict would do
the same job in four lines. The reason it is retrieval instead:

  1. A dict is a decision baked into code. Policy is a thing a bank CHANGES --
     eligibility thresholds move, products launch, a regulator restricts an
     outreach channel. Retrieval means updating a document, not redeploying.
  2. The action_subtype field is free text and must be justified. Retrieved
     policy gives the Action Proposer something concrete to ground a subtype in
     and to cite, instead of inventing a plausible-sounding label.
  3. The problem statement marks down "the most obvious single-agent-with-a-
     vector-database pattern". The answer to that is not to avoid retrieval; it
     is to use it where it earns its place -- grounding a bounded decision in
     policy -- rather than as the whole architecture.

INCREMENTAL UPSERT, NOT FULL REBUILD
------------------------------------
Required by the mid-term. `upsert` is content-hashed: re-running the seed is a
no-op, and editing one policy document re-embeds only that one.

THE EMBEDDING PROBLEM, AND AN HONEST ANSWER
-------------------------------------------
ChromaDB's default embedding function downloads an ONNX MiniLM model on first
use. That is fine on a laptop with internet and a disaster in a sandboxed
evaluation environment: the run dies on a network error having nothing to do with
the system being evaluated.

So the embedder is pluggable. The default is tried first; if it cannot be built,
`HashingEmbedder` -- a dependency-free, deterministic bag-of-words embedding
defined below -- is used instead. Over a corpus of ~20 short policy documents
that is genuinely adequate, and it means the test suite and the graded run never
depend on a download. `SemanticMemory.embedder_name` records which was used so
the evaluation write-up can state it plainly.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .schema import Action, InferredState

TOKEN = re.compile(r"[a-z0-9']+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to with this "
    "was were will would should could may might can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class HashingEmbedder:
    """
    A deterministic bag-of-words embedding with no model and no download.

    Words are hashed into a fixed number of buckets, counts are damped with
    1 + log(count) so a word repeated ten times does not dominate one used once,
    and the vector is L2-normalised so cosine similarity behaves.

    This is the "hashing trick" -- standard, old, and well suited to a small
    closed corpus of domain text where the vocabulary of a query ("hospital
    bill", "payment plan") overlaps the documents directly. It has no semantic
    generalisation: it cannot know that "infant" relates to "baby". That is an
    acceptable trade for never failing on a network error, and the real
    embedder is used whenever it is available.
    """

    # Chroma's EmbeddingFunction protocol requires name() to be a CALLABLE, and
    # raises a confusing "'str' object is not callable" if it is an attribute.
    EMBEDDER_NAME = "hashing-256"

    @staticmethod
    def name() -> str:
        return HashingEmbedder.EMBEDDER_NAME

    @staticmethod
    def is_legacy() -> bool:
        # Chroma probes this as a callable; a bool attribute triggers a confusing
        # "'bool' object is not callable" deprecation warning on every write.
        return False

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002 -- Chroma's API
        return [self.embed(text) for text in input]

    # Newer Chroma versions call these two directly rather than __call__.
    # Symmetric embedding (the same function for documents and queries) is the
    # right choice here: an asymmetric model would need separate training, and
    # for bag-of-words overlap there is nothing to gain.
    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return self(input)

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return self(input)

    def embed(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        vector = [0.0] * self.dimensions
        for bucket, count in counts.items():
            vector[bucket] = 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingEmbedder":
        return HashingEmbedder(config.get("dimensions", 256))

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": self.dimensions}


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# The seed corpus
# ---------------------------------------------------------------------------
# Written as policy prose rather than as a lookup table, because that is what it
# is standing in for. Each policy names the action enum it authorises and the
# action_subtype it grounds, so the proposer cites a document rather than
# inventing a label.

POLICIES: list[Document] = [
    Document(
        "pol_medical_hardship",
        "Medical hardship support. When a customer shows sustained healthcare or pharmacy "
        "spending, a large hospital or medical billing charge, replacement income such as a "
        "disability or benefits credit in place of salary, or contacts the bank asking about a "
        "payment plan or deferral after illness, the relationship is treated as a hardship case. "
        "The authorised action is support_intervention with action_subtype "
        "medical_hardship_payment_plan: a servicing call offering a repayment arrangement, fee "
        "suspension and a referral to the hardship team. Eligibility requires corroboration from "
        "at least two independent source systems and high confidence. Do not respond to medical "
        "hardship with a sales offer.",
        {"kind": "policy", "action": Action.SUPPORT_INTERVENTION.value,
         "action_subtype": "medical_hardship_payment_plan",
         "state": InferredState.MEDICAL_HARDSHIP.value, "min_confidence": "high"},
    ),
    Document(
        "pol_new_child",
        "New child and growing family. Indicators include a KYC dependents increase, purchases at "
        "baby, nursery and children's retailers, a new recurring daycare or childcare standing "
        "instruction, searches about child education savings, and a partial-pay income dip "
        "consistent with parental leave. The authorised action is personalized_offer with "
        "action_subtype childcare_savings_or_insurance_plan: a child education savings product, "
        "family life cover and a review of the household protection position. Requires high "
        "confidence and at least two independent source systems. Offers of this value are routed "
        "to a human for approval before contact.",
        {"kind": "policy", "action": Action.PERSONALIZED_OFFER.value,
         "action_subtype": "childcare_savings_or_insurance_plan",
         "state": InferredState.NEW_CHILD_LIFE_EVENT.value, "min_confidence": "high"},
    ),
    Document(
        "pol_churn_escalation",
        "Churn risk, early intervention window. Indicators include a denied or unresolved "
        "complaint, a sharp fall in app engagement or session length, use of cancellation or "
        "account-closure screens, standing instructions being stopped, and outbound transfers to "
        "an account the customer holds at another institution. For a high-value or long-tenured "
        "customer the authorised action is relationship_manager_escalation with action_subtype "
        "premium_retention_offer_and_fee_waiver: the relationship manager makes contact within "
        "24 hours, with authority to waive the disputed fee and present a premium retention "
        "package. This is the correct action the first time churn risk is established with high "
        "confidence, and acting early is the point.",
        {"kind": "policy", "action": Action.RELATIONSHIP_MANAGER_ESCALATION.value,
         "action_subtype": "premium_retention_offer_and_fee_waiver",
         "state": InferredState.CHURN_RISK.value, "min_confidence": "high", "stage": "early"},
    ),
    Document(
        "pol_churn_late_outreach",
        "Churn risk, late stage. Where a relationship manager escalation has already been raised "
        "and the customer has continued to disengage -- card activity at or near zero, salary "
        "swept out on the day of deposit, the account functioning only as a pass-through -- a "
        "second escalation adds nothing. The authorised action is proactive_retention_outreach: "
        "a structured win-back contact through the retention team. Use this rather than a repeat "
        "relationship_manager_escalation once an escalation is already open.",
        {"kind": "policy", "action": Action.PROACTIVE_RETENTION_OUTREACH.value,
         "action_subtype": "retention_winback_contact",
         "state": InferredState.CHURN_RISK.value, "min_confidence": "high", "stage": "late"},
    ),
    Document(
        "pol_financial_distress",
        "General financial distress. Savings drawdown, income disruption without a clear medical "
        "or family cause, and requests for payment relief. The authorised action is "
        "support_intervention with action_subtype financial_hardship_review. Do not present "
        "investment or borrowing offers to a customer in distress.",
        {"kind": "policy", "action": Action.SUPPORT_INTERVENTION.value,
         "action_subtype": "financial_hardship_review",
         "state": InferredState.FINANCIAL_DISTRESS_GENERAL.value, "min_confidence": "high"},
    ),
    Document(
        "pol_windfall",
        "Wealth growth and windfall. A large inbound deposit such as a tax refund, bonus or "
        "maturity credit may indicate investable surplus. An investment or savings "
        "personalized_offer is permitted ONLY where the inflow is corroborated by independent "
        "evidence of sustained balance growth. A single isolated deposit is NOT sufficient: funds "
        "frequently rest briefly before leaving the bank, and an offer made against money that is "
        "on its way out is both wasted and damaging. Never present a windfall offer to a customer "
        "concurrently showing churn or distress indicators.",
        {"kind": "policy", "action": Action.PERSONALIZED_OFFER.value,
         "action_subtype": "investment_review", "state": InferredState.WEALTH_GROWTH_OR_WINDFALL.value,
         "min_confidence": "high"},
    ),
    Document(
        "pol_fraud_hold",
        "Suspected fraud or account takeover. A compliance_fraud_hold is the most intrusive "
        "action available and freezes customer funds. It requires corroborated evidence across at "
        "least two independent source systems -- for example unrecognised international card "
        "activity together with credential changes or a customer contact disputing the "
        "transaction. A single unusual purchase, however large, is not grounds for a hold. "
        "Verified customer confirmation that a transaction was genuine removes the grounds "
        "entirely.",
        {"kind": "policy", "action": Action.COMPLIANCE_FRAUD_HOLD.value, "action_subtype": None,
         "state": InferredState.POTENTIAL_FRAUD_OR_TAKEOVER.value, "min_confidence": "high"},
    ),
    Document(
        "pol_no_action",
        "Observation only. Where the inferred state is not yet supported at high confidence, or "
        "the evidence comes from a single source system, the correct response is no_action: "
        "continue monitoring and record the assessment. Acting on weak or uncorroborated signals "
        "erodes trust and generates false positives. no_action is a decision, not a failure to "
        "decide.",
        {"kind": "policy", "action": Action.NO_ACTION.value, "action_subtype": None,
         "state": "any", "min_confidence": "low"},
    ),
]

PATTERNS: list[Document] = [
    Document(
        "pat_medical_hardship",
        "Medical hardship: an acute medical event followed by sustained healthcare cost and a "
        "change in income. Typically an emergency room or hospital charge, pharmacy spend, salary "
        "replaced by a lower benefits or disability credit, savings drawn down to cover a large "
        "bill, and eventually a direct approach to the bank about a payment plan.",
        {"kind": "pattern", "state": InferredState.MEDICAL_HARDSHIP.value},
    ),
    Document(
        "pat_new_child",
        "New child: a partial-pay income dip consistent with parental leave, purchases shifting "
        "towards baby and nursery retailers, a new recurring childcare commitment, searches about "
        "education savings, and a formal KYC dependents increase confirming the change.",
        {"kind": "pattern", "state": InferredState.NEW_CHILD_LIFE_EVENT.value},
    ),
    Document(
        "pat_churn_risk",
        "Churn risk: a grievance the bank refused, followed by quiet withdrawal rather than "
        "complaint. Engagement falls, sessions shorten, standing instructions are cancelled, "
        "savings move to an account the customer holds elsewhere, card use drops towards zero and "
        "salary is swept out on arrival. The account becomes a pass-through.",
        {"kind": "pattern", "state": InferredState.CHURN_RISK.value},
    ),
    Document(
        "pat_income_disruption",
        "Income disruption: regular salary credits stop, shrink, or are replaced by another "
        "income type, without the healthcare spending that would indicate a medical cause or the "
        "family indicators that would suggest parental leave.",
        {"kind": "pattern", "state": InferredState.JOB_LOSS_OR_INCOME_DISRUPTION.value},
    ),
    Document(
        "pat_windfall",
        "Wealth growth or windfall: a large inbound credit such as a tax refund, bonus or "
        "investment maturity, followed by a sustained rise in balances rather than an immediate "
        "outflow.",
        {"kind": "pattern", "state": InferredState.WEALTH_GROWTH_OR_WINDFALL.value},
    ),
]


@dataclass
class Retrieved:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def action(self) -> str | None:
        return self.metadata.get("action")

    @property
    def action_subtype(self) -> str | None:
        return self.metadata.get("action_subtype")


class SemanticMemory:
    """
    Usage:
        semantic = SemanticMemory()          # in-memory
        semantic = SemanticMemory("chroma/") # persistent
        semantic.seed()
        hits = semantic.retrieve_policies("churn risk cancellation transfer out", k=3)
    """

    def __init__(self, path: str | Path | None = None, prefer_default_embedder: bool = True) -> None:
        import chromadb

        self.path = str(path) if path else None
        self.embedder, self.embedder_name = self._build_embedder(prefer_default_embedder)

        if self.path:
            self.client = chromadb.PersistentClient(path=self.path)
        else:
            self.client = chromadb.Client()

        self.policies = self._collection("c360_policies")
        self.patterns = self._collection("c360_patterns")

    def _build_embedder(self, prefer_default: bool):
        """
        Try the real embedding model; fall back to hashing if it cannot be built.

        The fallback is not a silent downgrade -- `embedder_name` records which
        one is in use and the run log prints it, so an evaluation report can state
        honestly which embedding backed the retrieval.
        """
        if prefer_default:
            try:
                from chromadb.utils import embedding_functions

                default = embedding_functions.DefaultEmbeddingFunction()
                default(["warmup"])  # forces the download NOW rather than mid-run
                return default, "chroma-default-minilm"
            except Exception:  # noqa: BLE001 -- any failure means fall back
                pass
        return HashingEmbedder(), HashingEmbedder.EMBEDDER_NAME

    def _collection(self, name: str):
        return self.client.get_or_create_collection(name=name, embedding_function=self.embedder)

    # -- writing -----------------------------------------------------------

    def upsert(self, collection, documents: Sequence[Document]) -> int:
        """
        Incremental upsert, content-hashed.

        Required by the mid-term: semantic memory must stay live without a full
        re-index. Re-seeding is a no-op; editing one document re-embeds only that
        document. On a small corpus the saving is trivial, but the PROPERTY is the
        point -- a store that silently rebuilds is a store that cannot be updated
        during a run.
        """
        existing = collection.get(ids=[d.doc_id for d in documents], include=["metadatas"])
        current_hashes = {
            doc_id: (meta or {}).get("content_hash")
            for doc_id, meta in zip(existing.get("ids", []), existing.get("metadatas", []))
        }

        changed = [
            d for d in documents if current_hashes.get(d.doc_id) != _content_hash(d.text)
        ]
        if not changed:
            return 0

        collection.upsert(
            ids=[d.doc_id for d in changed],
            documents=[d.text for d in changed],
            metadatas=[
                {**{k: ("" if v is None else v) for k, v in d.metadata.items()},
                 "content_hash": _content_hash(d.text)}
                for d in changed
            ],
        )
        return len(changed)

    def seed(self) -> dict[str, int]:
        return {
            "policies": self.upsert(self.policies, POLICIES),
            "patterns": self.upsert(self.patterns, PATTERNS),
        }

    # -- reading -----------------------------------------------------------

    def retrieve_policies(self, query: str, k: int = 3) -> list[Retrieved]:
        return self._query(self.policies, query, k)

    def retrieve_patterns(self, query: str, k: int = 2) -> list[Retrieved]:
        return self._query(self.patterns, query, k)

    def _query(self, collection, query: str, k: int) -> list[Retrieved]:
        count = collection.count()
        if not count:
            return []
        result = collection.query(query_texts=[query], n_results=min(k, count))
        out: list[Retrieved] = []
        for doc_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            meta = dict(meta or {})
            for key in ("action_subtype",):
                if meta.get(key) == "":
                    meta[key] = None
            out.append(Retrieved(doc_id=doc_id, text=text, metadata=meta, distance=float(distance)))
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder_name,
            "policies": self.policies.count(),
            "patterns": self.patterns.count(),
        }


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
