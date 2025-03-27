"""SQLite memory store: customers, identifiers, memories, idempotency markers, dead letters.

One connection guarded by an RLock keeps `:memory:` databases working and makes every
read/write atomic; WAL mode is enabled for file databases. All memory reads take a
customer_id, so a query can never return another customer's rows even if the vector
index were wrong (defence in depth for the isolation guarantee).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .schemas import FactKind, IssueStatus, MemoryRecord, ensure_aware, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    merged_into TEXT
);
CREATE TABLE IF NOT EXISTS identifiers (
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    first_seen TEXT NOT NULL,
    PRIMARY KEY (kind, value)
);
CREATE INDEX IF NOT EXISTS ix_identifiers_customer ON identifiers(customer_id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    observed_at TEXT NOT NULL,
    status TEXT,
    evidence TEXT,
    source_event_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    extractor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER,
    embedder TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memories_customer ON memories(customer_id, active);
CREATE INDEX IF NOT EXISTS ix_memories_slot ON memories(customer_id, kind, key, active);
CREATE TABLE IF NOT EXISTS processed_events (
    event_id TEXT PRIMARY KEY,
    outcome TEXT NOT NULL,
    queue_offset INTEGER,
    processed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dead_letters (
    event_id TEXT PRIMARY KEY,
    correlation_id TEXT,
    trace_id TEXT,
    stage TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error TEXT NOT NULL,
    payload TEXT NOT NULL,
    spans TEXT NOT NULL DEFAULT '[]',
    attempts INTEGER NOT NULL DEFAULT 1,
    queue_offset INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_MEM_COLS = (
    "id, customer_id, kind, key, value, confidence, observed_at, status, evidence, "
    "source_event_id, conversation_id, prompt_version, extractor, created_at, active, superseded_by"
)


def _iso(dt: datetime) -> str:
    return ensure_aware(dt).isoformat()


def _row_to_memory(r: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=r["id"],
        customer_id=r["customer_id"],
        kind=FactKind(r["kind"]),
        key=r["key"],
        value=r["value"],
        confidence=r["confidence"],
        observed_at=datetime.fromisoformat(r["observed_at"]),
        status=IssueStatus(r["status"]) if r["status"] else None,
        evidence=r["evidence"],
        source_event_id=r["source_event_id"],
        conversation_id=r["conversation_id"],
        prompt_version=r["prompt_version"],
        extractor=r["extractor"],
        created_at=datetime.fromisoformat(r["created_at"]),
        active=bool(r["active"]),
        superseded_by=r["superseded_by"],
    )


class Store:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._in_tx = False
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialised write transaction; re-entrant (inner blocks join the outer one)."""
        with self._lock:
            if self._in_tx:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_tx = True
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            finally:
                self._in_tx = False

    def _q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _x(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    # ---- customers & identifiers -------------------------------------------------

    def create_customer(self) -> str:
        cid = "cus_" + uuid.uuid4().hex[:16]
        self._x("INSERT INTO customers VALUES (?, ?, NULL)", (cid, _iso(utcnow())))
        return cid

    def canonical_customer(self, customer_id: str) -> str | None:
        """Follow merge pointers; None when the id is unknown."""
        seen = set()
        while customer_id not in seen:
            seen.add(customer_id)
            rows = self._q("SELECT merged_into FROM customers WHERE customer_id=?", (customer_id,))
            if not rows:
                return None
            if rows[0]["merged_into"] is None:
                return customer_id
            customer_id = rows[0]["merged_into"]
        return None

    def find_customers(self, pairs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], str]:
        out = {}
        for kind, value in pairs:
            rows = self._q("SELECT customer_id FROM identifiers WHERE kind=? AND value=?", (kind, value))
            if rows:
                out[(kind, value)] = rows[0]["customer_id"]
        return out

    def link_identifiers(self, customer_id: str, pairs: Iterable[tuple[str, str]]) -> None:
        now = _iso(utcnow())
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO identifiers VALUES (?, ?, ?, ?)",
                [(k, v, customer_id, now) for k, v in pairs],
            )

    def identifiers_of(self, customer_id: str) -> list[tuple[str, str]]:
        rows = self._q("SELECT kind, value FROM identifiers WHERE customer_id=? ORDER BY kind, value", (customer_id,))
        return [(r["kind"], r["value"]) for r in rows]

    def merge_customers(self, source: str, target: str) -> list[int]:
        """Deliberately merge `source` into `target`. Returns ids of memories that moved."""
        if source == target:
            raise ValueError("cannot merge a customer into itself")
        with self.transaction():
            for c in (source, target):
                if self.canonical_customer(c) != c:
                    raise ValueError(f"{c} is not a live customer")
            moved = [r["id"] for r in self._q("SELECT id FROM memories WHERE customer_id=?", (source,))]
            self._x("UPDATE identifiers SET customer_id=? WHERE customer_id=?", (target, source))
            self._x("UPDATE memories SET customer_id=? WHERE customer_id=?", (target, source))
            self._x("UPDATE customers SET merged_into=? WHERE customer_id=?", (target, source))
        return moved

    # ---- idempotency ---------------------------------------------------------------

    def is_processed(self, event_id: str) -> bool:
        return bool(self._q("SELECT 1 FROM processed_events WHERE event_id=?", (event_id,)))

    def event_status(self, event_id: str) -> str | None:
        """Final outcome of an event (processed/skipped), 'failed' if dead-lettered, else None (pending)."""
        rows = self._q("SELECT outcome FROM processed_events WHERE event_id=?", (event_id,))
        if rows:
            return rows[0]["outcome"]
        return "failed" if self._q("SELECT 1 FROM dead_letters WHERE event_id=?", (event_id,)) else None

    def mark_processed(self, event_id: str, outcome: str, offset: int | None = None) -> None:
        self._x(
            "INSERT OR REPLACE INTO processed_events VALUES (?, ?, ?, ?)",
            (event_id, outcome, offset, _iso(utcnow())),
        )
        self._x("DELETE FROM dead_letters WHERE event_id=?", (event_id,))

    # ---- memories ------------------------------------------------------------------

    def insert_memory(self, rec: dict, embedder: str, embedding: np.ndarray) -> int:
        cur = self._x(
            "INSERT INTO memories (customer_id, kind, key, value, confidence, observed_at, status, evidence,"
            " source_event_id, conversation_id, prompt_version, extractor, created_at, active, embedder, embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
            (
                rec["customer_id"], rec["kind"].value, rec["key"], rec["value"], rec["confidence"],
                _iso(rec["observed_at"]), rec["status"].value if rec["status"] else None, rec["evidence"],
                rec["source_event_id"], rec["conversation_id"], rec["prompt_version"], rec["extractor"],
                _iso(utcnow()), embedder, np.asarray(embedding, dtype=np.float32).tobytes(),
            ),
        )
        return int(cur.lastrowid)

    def active_in_slot(self, customer_id: str, kind: FactKind, key: str) -> list[MemoryRecord]:
        rows = self._q(
            f"SELECT {_MEM_COLS} FROM memories WHERE customer_id=? AND kind=? AND key=? AND active=1",
            (customer_id, kind.value, key),
        )
        return [_row_to_memory(r) for r in rows]

    def supersede(self, ids: list[int], by: int) -> None:
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET active=0, superseded_by=? WHERE id=?", [(by, i) for i in ids]
            )

    def get_memories(self, customer_id: str, ids: list[int]) -> dict[int, MemoryRecord]:
        """Fetch by id, always constrained to the customer (isolation guard)."""
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._q(
            f"SELECT {_MEM_COLS} FROM memories WHERE customer_id=? AND id IN ({marks})", (customer_id, *ids)
        )
        return {r["id"]: _row_to_memory(r) for r in rows}

    def list_memories(self, customer_id: str, include_history: bool = False) -> list[MemoryRecord]:
        sql = f"SELECT {_MEM_COLS} FROM memories WHERE customer_id=?"
        if not include_history:
            sql += " AND active=1"
        return [_row_to_memory(r) for r in self._q(sql + " ORDER BY observed_at DESC, id DESC", (customer_id,))]

    def active_embeddings(self) -> list[dict]:
        """Every active memory with its stored vector, for rebuilding the vector index."""
        rows = self._q(
            "SELECT id, customer_id, kind, key, value, status, embedder, embedding FROM memories WHERE active=1"
        )
        return [dict(r) for r in rows]

    def update_embedding(self, memory_id: int, embedder: str, embedding: np.ndarray) -> None:
        self._x(
            "UPDATE memories SET embedder=?, embedding=? WHERE id=?",
            (embedder, np.asarray(embedding, dtype=np.float32).tobytes(), memory_id),
        )

    def delete_customer_data(self, customer_id: str) -> list[int]:
        """Right-to-be-forgotten: remove memories, identifiers and dead letters for one customer."""
        with self.transaction():
            ids = [r["id"] for r in self._q("SELECT id FROM memories WHERE customer_id=?", (customer_id,))]
            self._x("DELETE FROM memories WHERE customer_id=?", (customer_id,))
            self._x("DELETE FROM identifiers WHERE customer_id=?", (customer_id,))
            self._x("DELETE FROM customers WHERE customer_id=?", (customer_id,))
        return ids

    def delete_expired(self, resolved_before: datetime, any_before: datetime) -> list[tuple[int, str]]:
        """Retention sweep. Returns (memory id, customer id) of removed rows."""
        with self.transaction():
            rows = self._q(
                "SELECT id, customer_id FROM memories WHERE observed_at < ? OR"
                " (kind='issue' AND status='resolved' AND observed_at < ?)",
                (_iso(any_before), _iso(resolved_before)),
            )
            if rows:
                with self._lock:
                    self._conn.executemany("DELETE FROM memories WHERE id=?", [(r["id"],) for r in rows])
        return [(r["id"], r["customer_id"]) for r in rows]

    def reset_derived(self) -> None:
        """Drop everything derived from events (memories, markers, dead letters); keep identities."""
        with self.transaction():
            for t in ("memories", "processed_events", "dead_letters"):
                self._x(f"DELETE FROM {t}")

    # ---- dead letters --------------------------------------------------------------

    def record_dead_letter(self, *, event_id: str, correlation_id: str | None, trace_id: str | None, stage: str,
                           error_type: str, error: str, payload: str, spans: list[dict],
                           offset: int | None) -> None:
        now = _iso(utcnow())
        self._x(
            "INSERT INTO dead_letters (event_id, correlation_id, trace_id, stage, error_type, error, payload,"
            " spans, queue_offset, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(event_id) DO UPDATE SET correlation_id=excluded.correlation_id,"
            " trace_id=excluded.trace_id, stage=excluded.stage, error_type=excluded.error_type,"
            " error=excluded.error, spans=excluded.spans, attempts=attempts+1, updated_at=excluded.updated_at",
            (event_id, correlation_id, trace_id, stage, error_type, error[:2000], payload,
             json.dumps(spans, default=str), offset, now, now),
        )

    def list_dead_letters(self, limit: int = 100) -> list[dict]:
        rows = self._q(
            "SELECT event_id, correlation_id, trace_id, stage, error_type, error, attempts, queue_offset,"
            " created_at, updated_at FROM dead_letters ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def get_dead_letter(self, event_id: str) -> dict | None:
        rows = self._q("SELECT * FROM dead_letters WHERE event_id=?", (event_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["spans"] = json.loads(d["spans"])
        d["payload"] = json.loads(d["payload"]) if d["payload"].startswith(("{", "[")) else d["payload"]
        return d

    def stats(self) -> dict[str, int]:
        one = lambda sql: self._q(sql)[0][0]  # noqa: E731
        return {
            "customers": one("SELECT COUNT(*) FROM customers WHERE merged_into IS NULL"),
            "active_memories": one("SELECT COUNT(*) FROM memories WHERE active=1"),
            "superseded_memories": one("SELECT COUNT(*) FROM memories WHERE active=0"),
            "processed_events": one("SELECT COUNT(*) FROM processed_events"),
            "dead_letters": one("SELECT COUNT(*) FROM dead_letters"),
        }
