"""
Episodic memory -- the per-customer SQLite store.

WHAT THIS IS FOR
----------------
The replay engine hands out events one at a time and then forgets them. Episodic
memory is where they go, along with every finding the perception swarm publishes
and every decision the system commits to. It is the thing that makes this an
"ambient" system rather than a stateless classifier: an inference made on 8 March
is still on the record, correctly timestamped, on 10 April.

WHY SQLITE AND NOT REDIS
------------------------
The brief allows either. At ~500 events per customer the entire dataset fits in a
few hundred kilobytes, so throughput is irrelevant and the thing that actually
matters is being able to express one query correctly:

    WHERE event_time <= :as_of AND release_time <= :as_of

SQL expresses that precisely, the index makes it exact rather than approximate,
and the database file is a durable artifact you can open with any SQLite browser
during a viva and show the examiner. Redis would be a key-value store we would
then have to filter in Python, which is strictly more code for strictly less
guarantee.

THE ONE RULE THIS FILE EXISTS TO ENFORCE
----------------------------------------
`as_of` is the FIRST POSITIONAL ARGUMENT of every read method, with no default.
Not a keyword argument, not optional, not inferred from a clock the class holds
privately. A caller physically cannot ask this store a question without stating
what "now" is.

That is a deliberate API choice. The alternative -- letting memory hold a
reference to the clock and read it internally -- looks tidier and is far more
dangerous: the leak would then be invisible at every call site, and the hard rule
of the whole project would depend on one object's internal state being correct.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .findings import Finding, SignalStrength
from .output import Checkpoint
from .schema import Action, ConfidenceBand, Event, HitlStatus, InferredState

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    event_time      TEXT NOT NULL,
    ingestion_time  TEXT NOT NULL,
    release_time    TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    account_id      TEXT,
    source_system   TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL
);
-- The composite index matches the visibility predicate exactly, so the filter
-- that protects the hard rule is also the fast path.
CREATE INDEX IF NOT EXISTS idx_events_visibility
    ON events (customer_id, event_time, release_time);
CREATE INDEX IF NOT EXISTS idx_events_source
    ON events (customer_id, source_system, event_time);

CREATE TABLE IF NOT EXISTS findings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT NOT NULL,
    agent           TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    signal          TEXT NOT NULL,
    strength        TEXT NOT NULL,
    event_ids       TEXT NOT NULL,
    source_systems  TEXT NOT NULL,
    detail          TEXT,
    window_start    TEXT,
    metrics         TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_time ON findings (customer_id, as_of);

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT NOT NULL,
    as_of_time      TEXT NOT NULL,
    inferred_state  TEXT NOT NULL,
    confidence_band TEXT NOT NULL,
    action          TEXT NOT NULL,
    action_subtype  TEXT,
    hitl_status     TEXT NOT NULL,
    notes           TEXT,
    citations       TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions (customer_id, as_of_time);
"""


class TemporalLeakError(RuntimeError):
    """Raised when a query would expose information from the future."""


def _iso(moment: datetime) -> str:
    """
    Serialise a datetime for storage.

    Always normalised to UTC first. SQLite compares TEXT lexicographically, and
    ISO-8601 UTC strings sort correctly that way -- but only if every row uses the
    same offset. One row stored as "+05:30" would sort into the wrong place and
    silently corrupt every temporal filter in the system. Normalising here makes
    that impossible.
    """
    if moment.tzinfo is None:
        raise ValueError("refusing to store a naive datetime")
    return moment.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _require_as_of(as_of: Any) -> datetime:
    if not isinstance(as_of, datetime):
        raise TypeError(f"as_of must be a datetime, got {type(as_of).__name__}")
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware -- a naive 'now' is a leak waiting to happen")
    return as_of.astimezone(timezone.utc)


class EpisodicMemory:
    """
    Per-customer event, finding and decision history.

    Usage:
        memory = EpisodicMemory(customer_id="CUST_00184")     # in-memory
        memory = EpisodicMemory("run.sqlite", "CUST_00184")   # on disk
        memory.record_events(engine.history)
        recent = memory.events_as_of(now, since_days=30)
    """

    def __init__(self, path: str | Path = ":memory:", customer_id: str | None = None) -> None:
        self.path = str(path)
        self.customer_id = customer_id
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EpisodicMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # =====================================================================
    # Writing
    # =====================================================================

    def record_event(self, event: Event) -> None:
        self.record_events([event])

    def record_events(self, events: Iterable[Event]) -> int:
        """
        Store events. Re-recording the same event_id is a harmless no-op.

        INSERT OR IGNORE rather than REPLACE: if the same event arrives twice, the
        first copy is authoritative. Replacing would let a duplicate with a
        different timestamp silently rewrite history, which is exactly the kind of
        thing that would break a temporal filter without any error appearing.
        """
        rows = [
            (
                e.event_id,
                _iso(e.event_time),
                _iso(e.ingestion_time),
                _iso(e.release_time),
                e.customer_id,
                e.account_id,
                e.source_system,
                e.event_type,
                json.dumps(e.payload),
            )
            for e in events
        ]
        if not rows:
            return 0
        cursor = self.conn.executemany(
            "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self.conn.commit()
        return cursor.rowcount

    def record_finding(self, finding: Finding) -> None:
        row = finding.to_row()
        self.conn.execute(
            "INSERT INTO findings (customer_id, agent, as_of, signal, strength, event_ids, "
            "source_systems, detail, window_start, metrics) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.customer_id,
                row["agent"],
                row["as_of"],
                row["signal"],
                row["strength"],
                row["event_ids"],
                row["source_systems"],
                row["detail"],
                row["window_start"],
                row["metrics"],
            ),
        )
        self.conn.commit()

    def record_findings(self, findings: Iterable[Finding]) -> None:
        for finding in findings:
            self.record_finding(finding)

    def record_decision(self, checkpoint: Checkpoint) -> None:
        self.conn.execute(
            "INSERT INTO decisions (customer_id, as_of_time, inferred_state, confidence_band, "
            "action, action_subtype, hitl_status, notes, citations) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                self.customer_id,
                _iso(checkpoint.as_of_time),
                checkpoint.inferred_state.value,
                checkpoint.confidence_band.value,
                checkpoint.action.value,
                checkpoint.action_subtype,
                checkpoint.hitl_status.value,
                checkpoint.notes,
                json.dumps(list(checkpoint.citations)),
            ),
        )
        self.conn.commit()

    # =====================================================================
    # Reading -- every method below takes as_of FIRST and REQUIRED
    # =====================================================================

    def events_as_of(
        self,
        as_of: datetime,
        *,
        since_days: float | None = None,
        source_systems: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """
        Every event visible at `as_of`, oldest first.

        THE TWO FILTERS, AND WHY BOTH:

          event_time   <= as_of   Temporal leakage. Stops an agent reasoning about
                                  something that has not happened yet. This is the
                                  project's stated hard rule.

          release_time <= as_of   Availability leakage. Stops an agent reasoning
                                  about something that HAS happened but has not
                                  reached the bank yet. Without it, the $450
                                  diagnostics charge in scenario_01 (occurred
                                  1 March, received 3 March) would appear to have
                                  been known two days before it arrived -- and
                                  every lead-time figure computed from it would be
                                  wrong.

        This is deliberately the same predicate as ReplayEngine.visible_events().
        A test asserts the two agree on every checkpoint of every scenario; they
        are independent implementations, so agreement is evidence, not tautology.
        """
        as_of = _require_as_of(as_of)
        clauses = ["event_time <= :as_of", "release_time <= :as_of"]
        params: dict[str, Any] = {"as_of": _iso(as_of)}

        if self.customer_id:
            clauses.append("customer_id = :customer_id")
            params["customer_id"] = self.customer_id

        if since_days is not None:
            # A lookback window, not an escape hatch: it can only ever narrow the
            # set, never extend it past as_of.
            clauses.append("event_time >= :since")
            params["since"] = _iso(as_of - timedelta(days=since_days))

        if source_systems:
            placeholders = ",".join(f":ss{i}" for i in range(len(source_systems)))
            clauses.append(f"source_system IN ({placeholders})")
            params.update({f"ss{i}": s for i, s in enumerate(source_systems)})

        if event_types:
            placeholders = ",".join(f":et{i}" for i in range(len(event_types)))
            clauses.append(f"event_type IN ({placeholders})")
            params.update({f"et{i}": t for i, t in enumerate(event_types)})

        sql = f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY event_time, event_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self.conn.execute(sql, params).fetchall()
        events = [self._row_to_event(r) for r in rows]

        # Belt and braces. The SQL above cannot return a future event, but this
        # loop costs microseconds and converts "we believe the query is right"
        # into "the process would crash if it were not".
        for event in events:
            if event.event_time > as_of or event.release_time > as_of:
                raise TemporalLeakError(
                    f"{event.event_id} leaked: event_time={event.event_time.isoformat()} "
                    f"release_time={event.release_time.isoformat()} as_of={as_of.isoformat()}"
                )
        return events

    def count_as_of(self, as_of: datetime, **kwargs: Any) -> int:
        return len(self.events_as_of(as_of, **kwargs))

    def findings_as_of(
        self,
        as_of: datetime,
        *,
        since_days: float | None = None,
        agents: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[Finding]:
        """
        Findings published at or before `as_of`.

        Only one filter here, on `as_of`. A finding has no separate "arrival"
        time: it is produced by our own agents from data that was already visible,
        so the moment it was created is the moment it became knowable.
        """
        as_of = _require_as_of(as_of)
        clauses = ["as_of <= :as_of"]
        params: dict[str, Any] = {"as_of": _iso(as_of)}

        if self.customer_id:
            clauses.append("customer_id = :customer_id")
            params["customer_id"] = self.customer_id
        if since_days is not None:
            clauses.append("as_of >= :since")
            params["since"] = _iso(as_of - timedelta(days=since_days))
        if agents:
            placeholders = ",".join(f":a{i}" for i in range(len(agents)))
            clauses.append(f"agent IN ({placeholders})")
            params.update({f"a{i}": a for i, a in enumerate(agents)})
        if signals:
            placeholders = ",".join(f":s{i}" for i in range(len(signals)))
            clauses.append(f"signal IN ({placeholders})")
            params.update({f"s{i}": s for i, s in enumerate(signals)})

        rows = self.conn.execute(
            f"SELECT * FROM findings WHERE {' AND '.join(clauses)} ORDER BY as_of, id", params
        ).fetchall()
        return [self._row_to_finding(r) for r in rows]

    def decisions_as_of(self, as_of: datetime) -> list[dict[str, Any]]:
        as_of = _require_as_of(as_of)
        clauses = ["as_of_time <= :as_of"]
        params: dict[str, Any] = {"as_of": _iso(as_of)}
        if self.customer_id:
            clauses.append("customer_id = :customer_id")
            params["customer_id"] = self.customer_id
        rows = self.conn.execute(
            f"SELECT * FROM decisions WHERE {' AND '.join(clauses)} ORDER BY as_of_time, id", params
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_decision_as_of(self, as_of: datetime) -> dict[str, Any] | None:
        """
        The most recent committed decision at or before `as_of`.

        This is what makes an inference PERSIST rather than be recomputed from
        scratch. The problem statement is explicit about it: a life-event inferred
        on day one must still be available and correctly weighted weeks later,
        rather than each agent re-deriving it. Carrying the previous state forward
        on quiet days is also what keeps ~70 of 74 daily checkpoints free of an
        LLM call.
        """
        decisions = self.decisions_as_of(as_of)
        return decisions[-1] if decisions else None

    # =====================================================================
    # Derived views the perception agents lean on
    # =====================================================================

    def source_systems_active(
        self, as_of: datetime, *, since_days: float = 14
    ) -> dict[str, int]:
        """How many events per source system in the trailing window."""
        counts: dict[str, int] = {}
        for event in self.events_as_of(as_of, since_days=since_days):
            counts[event.source_system] = counts.get(event.source_system, 0) + 1
        return counts

    def daily_rate(
        self,
        as_of: datetime,
        *,
        source_systems: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        window_days: float = 14,
    ) -> float:
        """
        Events per day over the trailing window.

        The backbone of every "this dropped off" detection. Comparing a recent
        rate against a baseline rate from the history seed is how usage collapse
        and card-spend collapse get spotted -- neither of which produces an event
        of its own to react to.
        """
        events = self.events_as_of(
            as_of,
            since_days=window_days,
            source_systems=source_systems,
            event_types=event_types,
        )
        return len(events) / window_days if window_days else 0.0

    def baseline_daily_rate(
        self,
        as_of: datetime,
        *,
        source_systems: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        baseline_days: float = 90,
        exclude_recent_days: float = 14,
    ) -> float:
        """
        The customer's OWN normal rate, measured before the recent window.

        Per-customer baselines rather than a global threshold: David Chen logs in
        far more than Marcus Vance does, so "two logins this fortnight" means
        something very different for each. Excluding the recent window stops the
        very collapse we are trying to detect from dragging the baseline down with
        it.
        """
        as_of = _require_as_of(as_of)
        cutoff = as_of - timedelta(days=exclude_recent_days)
        window = baseline_days - exclude_recent_days
        if window <= 0:
            return 0.0
        events = [
            e
            for e in self.events_as_of(
                cutoff, since_days=window, source_systems=source_systems, event_types=event_types
            )
        ]
        return len(events) / window

    # =====================================================================

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            event_time=_parse(row["event_time"]),
            ingestion_time=_parse(row["ingestion_time"]),
            customer_id=row["customer_id"],
            account_id=row["account_id"],
            source_system=row["source_system"],
            event_type=row["event_type"],
            schema_version="1.0",
            payload=json.loads(row["payload"]),
        )

    @staticmethod
    def _row_to_finding(row: sqlite3.Row) -> Finding:
        return Finding(
            agent=row["agent"],
            as_of=_parse(row["as_of"]),
            signal=row["signal"],
            strength=SignalStrength(row["strength"]),
            event_ids=tuple(json.loads(row["event_ids"])),
            source_systems=tuple(json.loads(row["source_systems"])),
            detail=row["detail"] or "",
            window_start=_parse(row["window_start"]) if row["window_start"] else None,
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
        )

    def stats(self) -> dict[str, int]:
        return {
            "events": self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "findings": self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
            "decisions": self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
        }
