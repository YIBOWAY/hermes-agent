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
import os
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
    approval_id: str
    action_digest: str
    expires_at: float


class DurableAuthorityLock:
    """Non-expiring, process-scoped ownership fence for one durable DB.

    The lock is held by an open file descriptor and is released by the OS on
    process exit/crash.  It deliberately has no lease timeout: a second API
    server can never reconcile or mutate a DB while the first process still
    owns authority, even when the two servers bind different HTTP ports.
    """

    def __init__(self, db_path: str | Path) -> None:
        canonical = Path(db_path).expanduser().resolve(strict=False)
        self.db_path = canonical
        self.lock_path = Path(f"{canonical}.authority.lock")
        self._handle: Optional[Any] = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_path, "a+b")  # noqa: SIM115 - held for lock lifetime
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
    conversation_session_id TEXT,
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
    approval_id   TEXT,
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

    def __init__(
        self,
        db_path: str | Path,
        *,
        authority_lock: Optional[DurableAuthorityLock] = None,
    ) -> None:
        self.db_path = Path(db_path)
        if authority_lock is not None and not authority_lock.acquired:
            raise RuntimeError("durable authority lock must be acquired before opening")
        self.authority_lock = authority_lock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Public methods are intentionally exercised inside the capability
        # probe's outer transaction.  RLock + savepoints make those nested
        # calls atomic without deadlocking or committing the probe rows.
        self._lock = threading.RLock()
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
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(approval_grants)"
                ).fetchall()
            }
            if "approval_id" not in columns:
                self._conn.execute(
                    "ALTER TABLE approval_grants ADD COLUMN approval_id TEXT"
                )
            run_columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(runs)"
                ).fetchall()
            }
            if "conversation_session_id" not in run_columns:
                self._conn.execute(
                    "ALTER TABLE runs ADD COLUMN conversation_session_id TEXT"
                )

    def close(self) -> None:
        try:
            with self._lock:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                self._conn.close()
        finally:
            # Production construction acquires this fence before SQLite is
            # opened/schema-initialized.  Keep it through the final checkpoint
            # and connection close, then release it last.
            if self.authority_lock is not None:
                self.authority_lock.release()

    # -- internal write helper: BEGIN IMMEDIATE + commit/rollback -----------

    def _write(self, fn):
        with self._lock:
            if self._conn.in_transaction:
                savepoint = f"durable_sp_{uuid.uuid4().hex}"
                self._conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    result = fn(self._conn)
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    return result
                except Exception:
                    self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(self._conn)
                self._conn.execute("COMMIT")
                return result
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # -- row 1/2: idempotency key + canonical digest + submit-or-get ---------

    def find_submission(
        self,
        *,
        idempotency_key: str,
        request_body: dict[str, Any],
    ) -> Optional[SubmitResult]:
        """Read an existing exact submission without allocating a new Run.

        This preflight lets HTTP idempotency recovery bypass provider
        concurrency admission: returning an already-existing Run is a read, not
        new agent work.  A reused key with different canonical request bytes is
        still an exact conflict.
        """
        digest = canonical_digest(request_body)
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, request_digest FROM runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        if row["request_digest"] != digest:
            raise ConflictError(
                "idempotency key already used with a different request"
            )
        return SubmitResult(run_id=row["run_id"], created=False)

    def submit_or_get(
        self,
        *,
        idempotency_key: str,
        request_body: dict[str, Any],
        session_id: Optional[str] = None,
        conversation_session_id: Optional[str] = None,
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
                    " status, session_id, conversation_session_id, created_at,"
                    " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        idempotency_key,
                        digest,
                        RunState.QUEUED.value,
                        session_id or request_body.get("session_id") or run_id,
                        conversation_session_id
                        or session_id
                        or request_body.get("session_id")
                        or run_id,
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

    def list_non_terminal_runs(self) -> list[dict[str, Any]]:
        """All runs in a non-terminal state (queued/running) — for reconcile."""
        non_terminal = (RunState.QUEUED.value, RunState.RUNNING.value)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE status IN (?, ?)", non_terminal
            ).fetchall()
        return [dict(r) for r in rows]

    def register_run(
        self,
        *,
        run_id: str,
        session_id: Optional[str] = None,
        conversation_session_id: Optional[str] = None,
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
                " session_id, conversation_session_id, requested_policy,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    idempotency_key or f"server:{run_id}",
                    digest,
                    RunState.QUEUED.value,
                    session_id or run_id,
                    conversation_session_id or session_id or run_id,
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

        encoded = json.dumps(
            requested_policy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

        def _op(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT requested_policy, status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return False
            if row["status"] != RunState.QUEUED.value:
                return False
            existing = row["requested_policy"]
            if existing is not None:
                try:
                    existing = json.dumps(
                        json.loads(existing),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        default=str,
                    )
                except (TypeError, ValueError):
                    pass
                return existing == encoded
            cur = conn.execute(
                "UPDATE runs SET requested_policy = ?, updated_at = ?"
                " WHERE run_id = ? AND requested_policy IS NULL",
                (encoded, time.time(), run_id),
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
            row = conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"cannot record outcome for unknown run {run_id}")
            if RunState(row["status"]) in TERMINAL_STATES:
                raise TerminalStateError(
                    f"cannot rewrite outcome evidence for terminal run {run_id}"
                )
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

    def finalize_run(
        self,
        run_id: str,
        *,
        terminal_state: RunState,
        event_type: str,
        event_payload: dict[str, Any],
        actual_policy: Optional[dict[str, Any]] = None,
        fallback_reason: Optional[str] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> Optional[RunEvent]:
        """Atomically commit terminal event, outcome evidence, and status.

        A process crash or injected failure at any of the three public seams
        rolls the whole transaction back.  Repeating the same terminal finalize
        is idempotent and never appends a second terminal event; a conflicting
        terminal fact fails closed.
        """
        if terminal_state not in TERMINAL_STATES:
            raise ValueError("finalize_run requires a terminal RunState")

        def _op(conn: sqlite3.Connection) -> Optional[RunEvent]:
            row = conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"cannot finalize unknown run {run_id}")
            current = RunState(row["status"])
            if current in TERMINAL_STATES:
                if current == terminal_state:
                    events = [
                        event
                        for event in self.replay_events(run_id, since_seq=0)
                        if event.event_type == event_type
                        and event.payload == event_payload
                    ]
                    completed = self.get_run(run_id)
                    try:
                        stored_actual = (
                            json.loads(completed["actual_policy"])
                            if completed and completed.get("actual_policy")
                            else None
                        )
                        stored_usage = (
                            json.loads(completed["usage_json"])
                            if completed and completed.get("usage_json")
                            else None
                        )
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"run {run_id} has corrupt terminal evidence"
                        ) from exc
                    if (
                        len(events) == 1
                        and stored_actual == actual_policy
                        and stored_usage == usage
                        and completed.get("fallback_reason") == fallback_reason
                    ):
                        return None
                    raise RuntimeError(
                        f"run {run_id} terminal fact is incomplete or conflicting"
                    )
                raise TerminalStateError(
                    f"run {run_id} already finalized as {current.value}"
                )

            event = self.append_event(run_id, event_type, event_payload)
            if not self.record_run_outcome(
                run_id,
                actual_policy=actual_policy,
                fallback_reason=fallback_reason,
                usage=usage,
            ):
                raise RuntimeError(f"outcome evidence rejected for {run_id}")
            if not self.transition(run_id, terminal_state):
                raise RuntimeError(
                    f"terminal transition rejected for {run_id}: {terminal_state.value}"
                )
            return event

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
            run = conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"cannot append event to unknown run {run_id}")
            if RunState(run["status"]) in TERMINAL_STATES:
                raise TerminalStateError(
                    f"cannot append event to terminal run {run_id}"
                )
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

    def probe_capabilities(self) -> dict[str, bool]:
        """Exercise durable semantics in one transaction and always roll back.

        A readable database is not sufficient evidence for write durability.
        This probe performs the same write/read/CAS/state-transition operations
        the advertised capabilities depend on, while ``ROLLBACK`` guarantees
        that a health request leaves no canonical rows behind.
        """
        probe_token = uuid.uuid4().hex
        idempotency_key = f"probe:{probe_token}"
        approval_id = f"apr_probe_{probe_token}"
        request_body = {"probe": probe_token}
        evidence = {
            "idempotency": False,
            "event_replay": False,
            "approval_cas": False,
            "idempotent_stop": False,
            "restart_reconcile": False,
            "run_evidence": False,
        }

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                first = self.submit_or_get(
                    idempotency_key=idempotency_key,
                    request_body=request_body,
                )
                second = self.submit_or_get(
                    idempotency_key=idempotency_key,
                    request_body=request_body,
                )
                identity = self.get_run(first.run_id)
                evidence["idempotency"] = bool(
                    identity
                    and first.created
                    and not second.created
                    and second.run_id == first.run_id
                    and identity["request_digest"] == canonical_digest(request_body)
                )

                requested = {"model": "probe-requested"}
                actual = {"model": "probe-actual"}
                usage = {"total_tokens": 1}
                policy_set = self.set_requested_policy(first.run_id, requested)

                self.append_event(first.run_id, "probe.event", {"seq": 1})
                self.append_event(first.run_id, "probe.event", {"seq": 2})
                replay = self.replay_events(first.run_id, since_seq=0)
                evidence["event_replay"] = (
                    [row.seq for row in replay] == [1, 2]
                    and len({row.event_id for row in replay}) == 2
                )

                started = self.transition(first.run_id, RunState.RUNNING)
                challenge = self.issue_approval_challenge(
                    first.run_id,
                    approval_id=approval_id,
                    action_digest="probe-action",
                    ttl_seconds=60,
                )
                challenge_row = self.get_approval_challenge(challenge.challenge_id)
                wrong = self.consume_approval_with_event(
                    challenge.challenge_id,
                    approval_id="apr_wrong",
                    action_digest="probe-action",
                    choice="deny",
                )
                correct = self.consume_approval_with_event(
                    challenge.challenge_id,
                    approval_id=approval_id,
                    action_digest="probe-action",
                    choice="once",
                )
                second_choice = self.consume_approval_with_event(
                    challenge.challenge_id,
                    approval_id=approval_id,
                    action_digest="probe-action",
                    choice="deny",
                )
                evidence["approval_cas"] = (
                    challenge_row is not None
                    and challenge_row["approval_id"] == approval_id
                    and wrong is None
                    and correct is not None
                    and second_choice is None
                    and correct.event_type == "approval.decision_recorded"
                )

                terminal_payload = {
                    "event": "run.stopped",
                    "run_id": first.run_id,
                    "reason": "probe-stop",
                }
                stopped = self.finalize_run(
                    first.run_id,
                    terminal_state=RunState.STOPPED,
                    event_type="run.stopped",
                    event_payload=terminal_payload,
                    actual_policy=actual,
                    fallback_reason="probe-fallback",
                    usage=usage,
                )
                stopped_again = self.finalize_run(
                    first.run_id,
                    terminal_state=RunState.STOPPED,
                    event_type="run.stopped",
                    event_payload=terminal_payload,
                    actual_policy=actual,
                    fallback_reason="probe-fallback",
                    usage=usage,
                )
                stopped_row = self.get_run(first.run_id)
                evidence["idempotent_stop"] = (
                    started
                    and stopped is not None
                    and stopped_again is None
                    and stopped_row is not None
                    and stopped_row["status"] == RunState.STOPPED.value
                )

                reconcile = self.submit_or_get(
                    idempotency_key=f"probe-reconcile:{probe_token}",
                    request_body={"reconcile": probe_token},
                )
                reconcile_started = self.transition(
                    reconcile.run_id, RunState.RUNNING
                )
                non_terminal = self.list_non_terminal_runs()
                reconcile_payload = {
                    "event": "run.stopped",
                    "run_id": reconcile.run_id,
                    "reason": "startup_reconcile",
                }
                reconciled = self.finalize_run(
                    reconcile.run_id,
                    terminal_state=RunState.STOPPED,
                    event_type="run.stopped",
                    event_payload=reconcile_payload,
                )
                evidence["restart_reconcile"] = (
                    reconcile_started
                    and any(row["run_id"] == reconcile.run_id for row in non_terminal)
                    and reconciled is not None
                    and self.get_run(reconcile.run_id)["status"]
                    == RunState.STOPPED.value
                )

                outcome = self.get_run(first.run_id)
                evidence["run_evidence"] = bool(
                    outcome
                    and policy_set
                    and json.loads(outcome["requested_policy"]) == requested
                    and json.loads(outcome["actual_policy"]) == actual
                    and outcome["fallback_reason"] == "probe-fallback"
                    and json.loads(outcome["usage_json"]) == usage
                )
                return evidence
            finally:
                self._conn.execute("ROLLBACK")

    # -- row 6: approval challenge + TTL + single-use + CAS -------------------

    def issue_approval_challenge(
        self,
        run_id: str,
        *,
        action_digest: str,
        ttl_seconds: float,
        approval_id: Optional[str] = None,
    ) -> ApprovalChallenge:
        def _op(conn: sqlite3.Connection) -> ApprovalChallenge:
            run = conn.execute(
                "SELECT status, NOT EXISTS (SELECT 1 FROM run_events"
                " WHERE run_events.run_id = runs.run_id"
                " AND event_type = 'run.stop_requested') AS accepts_approval"
                " FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError(
                    f"cannot issue approval for unknown run {run_id}"
                )
            if (
                run["status"] != RunState.RUNNING.value
                or not bool(run["accepts_approval"])
            ):
                raise RuntimeError(
                    f"cannot issue approval unless run {run_id} is running and not stopping"
                )
            now = time.time()
            exact_approval_id = approval_id or f"apr_{uuid.uuid4().hex}"
            ch = ApprovalChallenge(
                challenge_id=f"chg_{uuid.uuid4().hex}",
                run_id=run_id,
                approval_id=exact_approval_id,
                action_digest=action_digest,
                expires_at=now + ttl_seconds,
            )
            conn.execute(
                "INSERT INTO approval_grants (challenge_id, run_id, approval_id,"
                " action_digest, expires_at, consumed, created_at)"
                " VALUES (?, ?, ?, ?, ?, 0, ?)",
                (
                    ch.challenge_id,
                    run_id,
                    exact_approval_id,
                    action_digest,
                    ch.expires_at,
                    now,
                ),
            )
            return ch

        return self._write(_op)

    def consume_approval(
        self,
        challenge_id: str,
        *,
        action_digest: str,
        approval_id: Optional[str] = None,
    ) -> bool:
        """Single-use + TTL + CAS consume.

        The UPDATE only fires when the grant is unconsumed, unexpired, AND the
        presented action_digest matches the challenged one — so a stale, expired,
        or forged attempt never consumes the grant, and a failed attempt leaves
        the real grant intact (compare-and-swap via rowcount).
        """

        def _op(conn: sqlite3.Connection) -> bool:
            sql = (
                "UPDATE approval_grants SET consumed = 1"
                " WHERE challenge_id = ? AND consumed = 0 AND expires_at > ?"
                " AND action_digest = ?"
                " AND run_id IN (SELECT run_id FROM runs WHERE status = 'running')"
                " AND NOT EXISTS (SELECT 1 FROM run_events"
                " WHERE run_events.run_id = approval_grants.run_id"
                " AND event_type = 'run.stop_requested')"
            )
            params: list[Any] = [challenge_id, time.time(), action_digest]
            if approval_id is not None:
                sql += " AND approval_id = ?"
                params.append(approval_id)
            cur = conn.execute(sql, params)
            return cur.rowcount == 1

        return self._write(_op)

    def consume_approval_with_event(
        self,
        challenge_id: str,
        *,
        action_digest: str,
        approval_id: str,
        choice: str,
    ) -> Optional[RunEvent]:
        """Atomically consume one exact grant and persist its decision fact.

        The public ``append_event`` call intentionally runs inside this outer
        transaction.  Its nested savepoint lets fault injection prove that an
        event failure rolls the CAS back, so no live waiter can be released by
        an unaudited approval.
        """

        def _op(conn: sqlite3.Connection) -> Optional[RunEvent]:
            row = conn.execute(
                "SELECT approval_grants.run_id FROM approval_grants"
                " JOIN runs ON runs.run_id = approval_grants.run_id"
                " WHERE challenge_id = ? AND approval_id = ?"
                " AND action_digest = ? AND consumed = 0 AND expires_at > ?",
                (challenge_id, approval_id, action_digest, time.time()),
            ).fetchone()
            if row is None:
                return None
            if conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND status = ?"
                " AND NOT EXISTS (SELECT 1 FROM run_events"
                " WHERE run_id = ? AND event_type = ?)",
                (
                    row["run_id"],
                    RunState.RUNNING.value,
                    row["run_id"],
                    "run.stop_requested",
                ),
            ).fetchone() is None:
                return None
            cur = conn.execute(
                "UPDATE approval_grants SET consumed = 1"
                " WHERE challenge_id = ? AND approval_id = ?"
                " AND action_digest = ? AND consumed = 0 AND expires_at > ?",
                (challenge_id, approval_id, action_digest, time.time()),
            )
            if cur.rowcount != 1:
                return None
            run_id = str(row["run_id"])
            return self.append_event(
                run_id,
                "approval.decision_recorded",
                {
                    "event": "approval.decision_recorded",
                    "run_id": run_id,
                    "challenge_id": challenge_id,
                    "approval_id": approval_id,
                    "action_digest": action_digest,
                    "choice": choice,
                    "timestamp": time.time(),
                },
            )

        return self._write(_op)

    def release_approval_consume(self, challenge_id: str) -> bool:
        """Undo a successful consume when the live resolve did not take effect.

        Plan A6: a non-success approval path must not change the prior fact.
        If the adapter CAS-consumed the grant and then ``resolve_gateway_approval``
        returned 0 (empty queue / race / restart), restore ``consumed=0`` so the
        client can retry with the same challenge. Only un-burns a currently
        consumed row; never touches unconsumed/foreign rows.
        """

        def _op(conn: sqlite3.Connection) -> bool:
            challenge = conn.execute(
                "SELECT run_id FROM approval_grants WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()
            if challenge is None:
                return False
            decisions = conn.execute(
                "SELECT payload_json FROM run_events"
                " WHERE run_id = ? AND event_type = ?",
                (challenge["run_id"], "approval.decision_recorded"),
            ).fetchall()
            if any(
                json.loads(row["payload_json"]).get("challenge_id") == challenge_id
                for row in decisions
            ):
                # Once a decision fact exists, consumption is immutable.  A
                # recovery path may deliver that same choice, never reopen CAS.
                return False
            cur = conn.execute(
                "UPDATE approval_grants SET consumed = 0"
                " WHERE challenge_id = ? AND consumed = 1",
                (challenge_id,),
            )
            return cur.rowcount == 1

        return self._write(_op)

    def get_approval_challenge(self, challenge_id: str) -> Optional[dict[str, Any]]:
        """Fetch an approval grant row (for run-binding checks), or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT challenge_id, run_id, approval_id, action_digest, expires_at,"
                " consumed, created_at FROM approval_grants WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_approval_decision(self, challenge_id: str) -> Optional[RunEvent]:
        """Return the immutable decision fact for a challenge, if committed."""
        challenge = self.get_approval_challenge(challenge_id)
        if challenge is None:
            return None
        for event in self.replay_events(str(challenge["run_id"]), since_seq=0):
            if (
                event.event_type == "approval.decision_recorded"
                and event.payload.get("challenge_id") == challenge_id
            ):
                return event
        return None

    def get_approval_response(self, challenge_id: str) -> Optional[RunEvent]:
        """Return the exact committed live-response fact for a challenge."""
        challenge = self.get_approval_challenge(challenge_id)
        if challenge is None:
            return None
        for event in self.replay_events(str(challenge["run_id"]), since_seq=0):
            if (
                event.event_type == "approval.responded"
                and event.payload.get("challenge_id") == challenge_id
            ):
                return event
        return None

    def get_approval_release(self, challenge_id: str) -> Optional[RunEvent]:
        """Return the durable release commitment, which is not signal proof."""
        challenge = self.get_approval_challenge(challenge_id)
        if challenge is None:
            return None
        for event in self.replay_events(str(challenge["run_id"]), since_seq=0):
            if (
                event.event_type == "approval.release_committed"
                and event.payload.get("challenge_id") == challenge_id
            ):
                return event
        return None

    def get_approval_delivery(self, challenge_id: str) -> Optional[RunEvent]:
        """Return post-signal observation, never a pre-signal commitment."""
        challenge = self.get_approval_challenge(challenge_id)
        if challenge is None:
            return None
        for event in self.replay_events(str(challenge["run_id"]), since_seq=0):
            if (
                event.event_type == "approval.signalled"
                and event.payload.get("challenge_id") == challenge_id
            ):
                return event
        return None

    def approval_delivery_allowed(
        self,
        challenge_id: str,
        *,
        approval_id: str,
        action_digest: str,
        choice: str,
    ) -> bool:
        """Re-fence recovery delivery against canonical run/stop truth.

        A consumed decision may be retried after a response-persistence fault,
        but it never grants a timeless capability: delivery remains legal only
        while the same run is running and has no durable stop intent.
        """
        with self._lock:
            grant = self._conn.execute(
                "SELECT approval_grants.run_id, approval_grants.approval_id,"
                " approval_grants.action_digest, approval_grants.consumed,"
                " runs.status FROM approval_grants JOIN runs"
                " ON runs.run_id = approval_grants.run_id"
                " WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()
            if (
                grant is None
                or not bool(grant["consumed"])
                or grant["approval_id"] != approval_id
                or grant["action_digest"] != action_digest
                or grant["status"] != RunState.RUNNING.value
            ):
                return False
            if self._conn.execute(
                "SELECT 1 FROM run_events WHERE run_id = ?"
                " AND event_type = 'run.stop_requested' LIMIT 1",
                (grant["run_id"],),
            ).fetchone() is not None:
                return False
            decisions = self._conn.execute(
                "SELECT payload_json FROM run_events WHERE run_id = ?"
                " AND event_type = 'approval.decision_recorded'",
                (grant["run_id"],),
            ).fetchall()
            return any(
                (payload := json.loads(row["payload_json"])).get("challenge_id")
                == challenge_id
                and payload.get("approval_id") == approval_id
                and payload.get("action_digest") == action_digest
                and payload.get("choice") == choice
                for row in decisions
            )
