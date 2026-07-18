"""DurableRunAuthority durable store (Agent v0.2 Slice V2.4).

This is the shared地基 (foundation) for the nine durable-run semantics pinned in
``docs/durable-run-authority-contract.md``: submit-or-get by request identity,
persistent Run identity/status, stable event id + per-run monotonic cursor
replay, and approval challenge/TTL/single-use/CAS — all durable across process
restart.

Design mirrors the repo's proven SQLite discipline (hermes_state.py) and the
cron execution-ledger state machine (cron/executions.py):

* one shared connection per store instance, ``check_same_thread=False`` +
  ``isolation_level=None`` + ``row_factory=sqlite3.Row``, serialized by an
  instance ``threading.Lock``;
* WAL via ``apply_wal_with_fallback`` (NFS-safe fallback) then
  ``PRAGMA foreign_keys=ON``;
* manual transactions with ``BEGIN IMMEDIATE`` for cross-process-atomic claims;
* guarded exactly-once state transitions (``UPDATE ... WHERE status IN (...)``)
  so terminal rows can never be rewritten;
* canonical JSON (``sort_keys`` + tight separators) for the request digest —
  never ``repr()`` — so semantically identical requests hash identically.

Red line (contract §7): zero live effect. This module only reads/writes the
caller-supplied ``db_path``; tests always use a throwaway tmp_path file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from hermes_state import apply_wal_with_fallback

# ---------------------------------------------------------------------------
# contract state machine (contract matrix §1)
# ---------------------------------------------------------------------------


class RunState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"  # terminal
    FAILED = "failed"  # terminal
    STOPPED = "stopped"  # terminal


TERMINAL_STATES = frozenset(
    {RunState.SUCCEEDED, RunState.FAILED, RunState.STOPPED}
)

# Allowed transitions. Terminal states have no outgoing edges (immutable).
_ALLOWED_TRANSITIONS = {
    RunState.QUEUED: frozenset(
        {RunState.RUNNING, RunState.STOPPED}
    ),  # stop-before-start allowed
    RunState.RUNNING: frozenset(
        {RunState.SUCCEEDED, RunState.FAILED, RunState.STOPPED}
    ),
    RunState.SUCCEEDED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.STOPPED: frozenset(),
}


class ConflictError(RuntimeError):
    """Same idempotency key presented with a different request digest (409)."""


class TerminalStateError(RuntimeError):
    """A transition was attempted out of an immutable terminal state."""


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubmitResult:
    run_id: str
    created: bool  # True if newly inserted, False if recovered by identity


@dataclass(frozen=True)
class RunEvent:
    event_id: str
    run_id: str
    seq: int
    event_type: str
    payload: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class ApprovalChallenge:
    challenge_id: str
    run_id: str
    action_digest: str
    expires_at: float


# ---------------------------------------------------------------------------
# DDL (declarative, CREATE IF NOT EXISTS — replay-safe)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    idempotency_key  TEXT NOT NULL,
    request_digest   TEXT NOT NULL,
    status           TEXT NOT NULL,
    session_id       TEXT,
    requested_policy TEXT,
    actual_policy    TEXT,
    fallback_reason  TEXT,
    usage_json       TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    UNIQUE (idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status, created_at);

CREATE TABLE IF NOT EXISTS run_events (
    event_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    seq          INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   REAL NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_run_events_run_seq ON run_events (run_id, seq);

CREATE TABLE IF NOT EXISTS approval_grants (
    challenge_id  TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    action_digest TEXT NOT NULL,
    expires_at    REAL NOT NULL,
    consumed      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_approval_grants_run ON approval_grants (run_id);
"""


def canonical_digest(request_body: dict[str, Any]) -> str:
    """sha256 over the request as canonical JSON (sort_keys, tight separators).

    Never ``repr()``: semantically identical requests (different key order) must
    hash identically, and ``repr()`` is not canonical across dict insertion order.
    """
    canonical = json.dumps(
        request_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,  # normalize non-JSON-native types deterministically
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DurableRunStore:
    """Append-only, restart-durable authority for Runs / events / approvals."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,  # manual transactions (BEGIN IMMEDIATE)
        )
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label=self.db_path.name)
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()

    # -- internal write helper: BEGIN IMMEDIATE + commit/rollback -----------

    def _write(self, fn):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(self._conn)
                self._conn.execute("COMMIT")
                return result
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # -- row 1/2: idempotency key + canonical digest + submit-or-get ---------

    def submit_or_get(
        self, *, idempotency_key: str, request_body: dict[str, Any]
    ) -> SubmitResult:
        digest = canonical_digest(request_body)

        def _op(conn: sqlite3.Connection) -> SubmitResult:
            row = conn.execute(
                "SELECT run_id, request_digest FROM runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row["request_digest"] != digest:
                    raise ConflictError(
                        "idempotency key already used with a different request"
                    )
                return SubmitResult(run_id=row["run_id"], created=False)

            run_id = f"run_{uuid.uuid4().hex}"
            now = time.time()
            try:
                conn.execute(
                    "INSERT INTO runs (run_id, idempotency_key, request_digest,"
                    " status, session_id, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        idempotency_key,
                        digest,
                        RunState.QUEUED.value,
                        request_body.get("session_id") or run_id,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                # Lost a cross-process race on UNIQUE(idempotency_key): the
                # winner's row is now visible; re-read and reconcile.
                row = conn.execute(
                    "SELECT run_id, request_digest FROM runs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is None:  # pragma: no cover - defensive
                    raise
                if row["request_digest"] != digest:
                    raise ConflictError(
                        "idempotency key already used with a different request"
                    )
                return SubmitResult(run_id=row["run_id"], created=False)
            return SubmitResult(run_id=run_id, created=True)

        return self._write(_op)

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def register_run(
        self,
        *,
        run_id: str,
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        request_body: Optional[dict[str, Any]] = None,
        requested_policy: Optional[str] = None,
    ) -> str:
        """Ensure a durable run row exists; return run_id.

        ``submit_or_get`` already persists keyed submissions. This registers
        server-minted runs (no Idempotency-Key) so their events/approvals have a
        parent row (contract row 9). Idempotent on ``run_id``: re-registering
        the same run is a no-op, so a later ``submit_or_get`` recovery or a
        duplicate registration never errors.
        """

        def _op(conn: sqlite3.Connection) -> str:
            if (
                conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is not None
            ):
                return run_id
            now = time.time()
            digest = (
                canonical_digest(request_body)
                if request_body is not None
                else f"server:{run_id}"
            )
            conn.execute(
                "INSERT INTO runs (run_id, idempotency_key, request_digest, status,"
                " session_id, requested_policy, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    idempotency_key or f"server:{run_id}",
                    digest,
                    RunState.QUEUED.value,
                    session_id or run_id,
                    requested_policy,
                    now,
                    now,
                ),
            )
            return run_id

        return self._write(_op)

    # -- row 5: requested vs actual provider/model/fallback/usage evidence ----

    def set_requested_policy(self, run_id: str, requested_policy: dict[str, Any]) -> bool:
        """Record the run's REQUESTED policy (the client's model + resolved route).

        Set once at submission; later actual-route recording never overwrites it
        (contract row 5: the requested route is never rewritten by the accounted
        route). Returns False for an unknown run (fail closed).
        """

        def _op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE runs SET requested_policy = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(requested_policy, ensure_ascii=False, default=str), time.time(), run_id),
            )
            return cur.rowcount == 1

        return self._write(_op)

    def record_run_outcome(
        self,
        run_id: str,
        *,
        actual_policy: Optional[dict[str, Any]] = None,
        fallback_reason: Optional[str] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record the run's ACTUAL outcome: provider/model, fallback reason, usage.

        Only touches actual_policy / fallback_reason / usage_json — never
        requested_policy. Returns False for an unknown run (fail closed).
        """

        def _op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE runs SET actual_policy = ?, fallback_reason = ?, usage_json = ?,"
                " updated_at = ? WHERE run_id = ?",
                (
                    json.dumps(actual_policy, ensure_ascii=False, default=str)
                    if actual_policy is not None
                    else None,
                    fallback_reason,
                    json.dumps(usage, ensure_ascii=False, default=str) if usage is not None else None,
                    time.time(),
                    run_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError(f"cannot record outcome for unknown run {run_id}")
            return True

        return self._write(_op)

    # -- row 1 state machine: guarded transitions + terminal immutability ----

    def transition(
        self, run_id: str, new_state: RunState, *, strict: bool = False
    ) -> bool:
        """Guarded exactly-once transition; terminal rows are immutable.

        Returns True iff the row moved to ``new_state``. Returns False for an
        unknown run, an illegal transition, or any transition out of a terminal
        state (fail closed). With ``strict=True``, raises TerminalStateError
        instead of returning False when the run is already terminal.
        """

        def _op(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return False
            current = RunState(row["status"])
            if current in TERMINAL_STATES:
                if strict:
                    raise TerminalStateError(
                        f"run {run_id} is terminal ({current.value}); cannot transition"
                    )
                return False
            if new_state not in _ALLOWED_TRANSITIONS[current]:
                return False
            cur = conn.execute(
                "UPDATE runs SET status = ?, updated_at = ?"
                " WHERE run_id = ? AND status = ?",
                (new_state.value, time.time(), run_id, current.value),
            )
            return cur.rowcount == 1

        return self._write(_op)

    # -- row 4: stable event id + per-run monotonic seq + replay --------------

    def append_event(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> RunEvent:
        def _op(conn: sqlite3.Connection) -> RunEvent:
            if (
                conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is None
            ):
                raise ValueError(f"cannot append event to unknown run {run_id}")
            next_seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            event = RunEvent(
                event_id=f"evt_{uuid.uuid4().hex}",
                run_id=run_id,
                seq=next_seq,
                event_type=event_type,
                payload=payload,
                created_at=time.time(),
            )
            conn.execute(
                "INSERT INTO run_events (event_id, run_id, seq, event_type,"
                " payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    run_id,
                    event.seq,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    event.created_at,
                ),
            )
            return event

        return self._write(_op)

    def replay_events(self, run_id: str, *, since_seq: int = 0) -> list[RunEvent]:
        """Events with seq > since_seq, ordered by seq (no gap, no dup)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_id, run_id, seq, event_type, payload_json, created_at"
                " FROM run_events WHERE run_id = ? AND seq > ? ORDER BY seq ASC",
                (run_id, since_seq),
            ).fetchall()
        return [
            RunEvent(
                event_id=r["event_id"],
                run_id=r["run_id"],
                seq=r["seq"],
                event_type=r["event_type"],
                payload=json.loads(r["payload_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- row 6: approval challenge + TTL + single-use + CAS -------------------

    def issue_approval_challenge(
        self, run_id: str, *, action_digest: str, ttl_seconds: float
    ) -> ApprovalChallenge:
        def _op(conn: sqlite3.Connection) -> ApprovalChallenge:
            if (
                conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is None
            ):
                raise ValueError(
                    f"cannot issue approval for unknown run {run_id}"
                )
            now = time.time()
            ch = ApprovalChallenge(
                challenge_id=f"chg_{uuid.uuid4().hex}",
                run_id=run_id,
                action_digest=action_digest,
                expires_at=now + ttl_seconds,
            )
            conn.execute(
                "INSERT INTO approval_grants (challenge_id, run_id, action_digest,"
                " expires_at, consumed, created_at) VALUES (?, ?, ?, ?, 0, ?)",
                (ch.challenge_id, run_id, action_digest, ch.expires_at, now),
            )
            return ch

        return self._write(_op)

    def consume_approval(self, challenge_id: str, *, action_digest: str) -> bool:
        """Single-use + TTL + CAS consume.

        The UPDATE only fires when the grant is unconsumed, unexpired, AND the
        presented action_digest matches the challenged one — so a stale, expired,
        or forged attempt never consumes the grant, and a failed attempt leaves
        the real grant intact (compare-and-swap via rowcount).
        """

        def _op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE approval_grants SET consumed = 1"
                " WHERE challenge_id = ? AND consumed = 0 AND expires_at > ?"
                " AND action_digest = ?",
                (challenge_id, time.time(), action_digest),
            )
            return cur.rowcount == 1

        return self._write(_op)
