"""V2.4 DurableRunAuthority durable-store tests (contract matrix §2 rows 2,3,4,6,9).

These pin the store contract BEFORE the implementation exists (TDD red). The
store is the shared地基 for submit-or-get, persistent Run identity/status,
stable event id + monotonic cursor replay, and approval challenge/TTL/single-use/
CAS — all durable across process restart (modeled here by reopening the DB).

No live effect: every test uses a throwaway tmp_path DB file.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from gateway.durable_runs import (
    ConflictError,
    DurableRunStore,
    RunState,
    TerminalStateError,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_BODY_A = {"input": "research AAPL momentum", "model": "test-model", "stream": False}
_BODY_B = {"input": "research MSFT value", "model": "test-model", "stream": False}


@pytest.fixture()
def store(tmp_path):
    s = DurableRunStore(db_path=tmp_path / "durable_runs.db")
    yield s
    s.close()


def _reopen(tmp_path, name="durable_runs.db"):
    return DurableRunStore(db_path=tmp_path / name)


# ---------------------------------------------------------------------------
# row 1/2: idempotency key + canonical digest + submit-or-get
# ---------------------------------------------------------------------------


def test_submit_or_get_returns_same_run_for_same_identity(store) -> None:
    first = store.submit_or_get(idempotency_key="k-1", request_body=_BODY_A)
    second = store.submit_or_get(idempotency_key="k-1", request_body=_BODY_A)

    assert first.run_id == second.run_id
    assert first.created is True
    assert second.created is False  # recovered, not re-created


def test_find_submission_is_read_only_exact_recovery(store) -> None:
    assert (
        store.find_submission(
            idempotency_key="k-find",
            request_body=_BODY_A,
        )
        is None
    )
    created = store.submit_or_get(
        idempotency_key="k-find",
        request_body=_BODY_A,
    )

    found = store.find_submission(
        idempotency_key="k-find",
        request_body=_BODY_A,
    )

    assert found is not None
    assert found.run_id == created.run_id
    assert found.created is False
    assert store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_find_submission_rejects_reused_key_with_different_body(store) -> None:
    store.submit_or_get(
        idempotency_key="k-find-conflict",
        request_body=_BODY_A,
    )

    with pytest.raises(ConflictError):
        store.find_submission(
            idempotency_key="k-find-conflict",
            request_body=_BODY_B,
        )


def test_submit_or_get_distinct_bodies_create_distinct_runs(store) -> None:
    a = store.submit_or_get(idempotency_key="k-a", request_body=_BODY_A)
    b = store.submit_or_get(idempotency_key="k-b", request_body=_BODY_B)
    assert a.run_id != b.run_id


def test_same_key_different_digest_conflicts(store) -> None:
    store.submit_or_get(idempotency_key="k-1", request_body=_BODY_A)
    with pytest.raises(ConflictError):
        store.submit_or_get(idempotency_key="k-1", request_body=_BODY_B)


def test_canonical_digest_is_key_order_independent(store) -> None:
    # Same semantic request, different key order in the body -> SAME run.
    body1 = {"a": 1, "b": {"x": 1, "y": 2}}
    body2 = {"b": {"y": 2, "x": 1}, "a": 1}
    r1 = store.submit_or_get(idempotency_key="k-canon", request_body=body1)
    r2 = store.submit_or_get(idempotency_key="k-canon", request_body=body2)
    assert r1.run_id == r2.run_id


def test_new_run_starts_queued(store) -> None:
    r = store.submit_or_get(idempotency_key="k-q", request_body=_BODY_A)
    got = store.get_run(r.run_id)
    assert got is not None
    assert got["status"] == RunState.QUEUED.value
    assert got["idempotency_key"] == "k-q"
    assert len(got["request_digest"]) == 64  # sha256 hex


def test_submit_or_get_persists_resolved_session_identity(store) -> None:
    result = store.submit_or_get(
        idempotency_key="k-session",
        request_body={"input": "continue", "previous_response_id": "resp_1"},
        session_id="session_from_previous_response",
    )

    assert store.get_run(result.run_id)["session_id"] == "session_from_previous_response"


def test_conversation_root_and_resolved_tip_survive_reopen(store, tmp_path) -> None:
    result = store.submit_or_get(
        idempotency_key="k-compressed-session",
        request_body={"input": "continue", "session_id": "conversation-root"},
        session_id="resolved-tip",
        conversation_session_id="conversation-root",
    )
    store.close()

    reopened = _reopen(tmp_path)
    try:
        run = reopened.get_run(result.run_id)
        assert run is not None
        assert run["conversation_session_id"] == "conversation-root"
        assert run["session_id"] == "resolved-tip"
        recovered = reopened.submit_or_get(
            idempotency_key="k-compressed-session",
            request_body={"input": "continue", "session_id": "conversation-root"},
            session_id="resolved-tip",
            conversation_session_id="conversation-root",
        )
        assert recovered.run_id == result.run_id
        assert recovered.created is False
    finally:
        reopened.close()


def test_requested_policy_is_set_once_and_same_value_is_idempotent(store) -> None:
    result = store.submit_or_get(idempotency_key="k-policy", request_body=_BODY_A)

    assert store.set_requested_policy(result.run_id, {"model": "requested-a"}) is True
    assert store.set_requested_policy(result.run_id, {"model": "requested-a"}) is True
    assert store.set_requested_policy(result.run_id, {"model": "requested-b"}) is False
    assert store.get_run(result.run_id)["requested_policy"] == '{"model":"requested-a"}'


# ---------------------------------------------------------------------------
# row 3/9: persistent Run identity/status across restart (reopen)
# ---------------------------------------------------------------------------


def test_run_identity_and_status_survive_reopen(store, tmp_path) -> None:
    r = store.submit_or_get(idempotency_key="k-persist", request_body=_BODY_A)
    store.transition(r.run_id, RunState.RUNNING)
    run_id = r.run_id
    store.close()

    reopened = _reopen(tmp_path)
    try:
        got = reopened.get_run(run_id)
        assert got is not None
        assert got["status"] == RunState.RUNNING.value
        assert got["idempotency_key"] == "k-persist"
        # submit-or-get after restart recovers the SAME run, not a new one.
        again = reopened.submit_or_get(idempotency_key="k-persist", request_body=_BODY_A)
        assert again.run_id == run_id
        assert again.created is False
    finally:
        reopened.close()


def test_get_run_unknown_returns_none(store) -> None:
    assert store.get_run("run_does_not_exist") is None


# ---------------------------------------------------------------------------
# row 1 state machine: guarded transitions + terminal immutability
# ---------------------------------------------------------------------------


def test_valid_transition_succeeds(store) -> None:
    r = store.submit_or_get(idempotency_key="k-t", request_body=_BODY_A)
    assert store.transition(r.run_id, RunState.RUNNING) is True
    assert store.transition(r.run_id, RunState.SUCCEEDED) is True
    assert store.get_run(r.run_id)["status"] == RunState.SUCCEEDED.value


def test_invalid_transition_rejected(store) -> None:
    r = store.submit_or_get(idempotency_key="k-bad", request_body=_BODY_A)
    # queued -> succeeded is not an allowed transition.
    assert store.transition(r.run_id, RunState.SUCCEEDED) is False
    assert store.get_run(r.run_id)["status"] == RunState.QUEUED.value


def test_terminal_state_is_immutable(store) -> None:
    r = store.submit_or_get(idempotency_key="k-term", request_body=_BODY_A)
    store.transition(r.run_id, RunState.RUNNING)
    store.transition(r.run_id, RunState.SUCCEEDED)

    # Any further transition out of a terminal state must fail closed.
    assert store.transition(r.run_id, RunState.RUNNING) is False
    assert store.transition(r.run_id, RunState.FAILED) is False
    with pytest.raises(TerminalStateError):
        store.transition(r.run_id, RunState.SUCCEEDED, strict=True)
    assert store.get_run(r.run_id)["status"] == RunState.SUCCEEDED.value


def test_transition_unknown_run_returns_false(store) -> None:
    assert store.transition("run_nope", RunState.RUNNING) is False


# ---------------------------------------------------------------------------
# row 4: stable event id + per-run monotonic seq + replay (no gap/dup)
# ---------------------------------------------------------------------------


def test_events_have_monotonic_per_run_seq(store) -> None:
    r = store.submit_or_get(idempotency_key="k-ev", request_body=_BODY_A)
    e1 = store.append_event(r.run_id, "run.started", {"msg": "go"})
    e2 = store.append_event(r.run_id, "message.delta", {"delta": "hi"})
    e3 = store.append_event(r.run_id, "run.completed", {})

    assert e1.seq == 1
    assert e2.seq == 2
    assert e3.seq == 3
    # stable, distinct event ids
    assert len({e1.event_id, e2.event_id, e3.event_id}) == 3
    assert all(e.event_id for e in (e1, e2, e3))


def test_replay_from_cursor_has_no_gap_no_dup(store) -> None:
    r = store.submit_or_get(idempotency_key="k-replay", request_body=_BODY_A)
    for i in range(5):
        store.append_event(r.run_id, "message.delta", {"i": i})

    full = store.replay_events(r.run_id, since_seq=0)
    assert [e.seq for e in full] == [1, 2, 3, 4, 5]

    tail = store.replay_events(r.run_id, since_seq=2)
    assert [e.seq for e in tail] == [3, 4, 5]
    # no duplicates within a replay
    seqs = [e.seq for e in tail]
    assert len(seqs) == len(set(seqs))


def test_events_survive_reopen(store, tmp_path) -> None:
    r = store.submit_or_get(idempotency_key="k-evp", request_body=_BODY_A)
    store.append_event(r.run_id, "run.started", {})
    store.append_event(r.run_id, "run.completed", {})
    run_id = r.run_id
    store.close()

    reopened = _reopen(tmp_path)
    try:
        events = reopened.replay_events(run_id, since_seq=0)
        assert [e.seq for e in events] == [1, 2]
        assert [e.event_type for e in events] == ["run.started", "run.completed"]
    finally:
        reopened.close()


def test_append_event_to_unknown_run_rejected(store) -> None:
    with pytest.raises((KeyError, ValueError)):
        store.append_event("run_nope", "run.started", {})


def test_capability_probe_exercises_writes_and_rolls_back(store) -> None:
    tables = ("runs", "run_events", "approval_grants")
    before_counts = {
        table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }

    evidence = store.probe_capabilities()

    assert all(evidence.values())
    after_counts = {
        table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }
    assert after_counts == before_counts


def test_capability_probe_uses_public_store_contract_and_outer_rollback(
    store, monkeypatch
) -> None:
    """A probe cannot pass by bypassing a broken public store operation."""
    assert isinstance(store._lock, type(threading.RLock()))
    before = {
        table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("runs", "run_events", "approval_grants")
    }

    def _fail(*args, **kwargs):
        raise OSError("public append contract unavailable")

    monkeypatch.setattr(store, "append_event", _fail)
    with pytest.raises(OSError, match="public append"):
        store.probe_capabilities()

    after = {
        table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("runs", "run_events", "approval_grants")
    }
    assert after == before


def test_atomic_approval_decision_rolls_back_consume_when_event_append_fails(
    store, monkeypatch
) -> None:
    run = store.submit_or_get(idempotency_key="k-atomic-approval", request_body=_BODY_A)
    assert store.transition(run.run_id, RunState.RUNNING)
    challenge = store.issue_approval_challenge(
        run.run_id,
        approval_id="apr_atomic",
        action_digest="digest-atomic",
        ttl_seconds=60,
    )

    def _fail(*args, **kwargs):
        raise OSError("event append unavailable")

    monkeypatch.setattr(store, "append_event", _fail)
    with pytest.raises(OSError, match="event append"):
        store.consume_approval_with_event(
            challenge.challenge_id,
            approval_id=challenge.approval_id,
            action_digest=challenge.action_digest,
            choice="once",
        )

    row = store.get_approval_challenge(challenge.challenge_id)
    assert row is not None
    assert row["consumed"] == 0
    assert store.replay_events(run.run_id) == []


def test_atomic_approval_decision_is_immutable_and_single_choice(store) -> None:
    run = store.submit_or_get(idempotency_key="k-immutable-decision", request_body=_BODY_A)
    assert store.transition(run.run_id, RunState.RUNNING)
    challenge = store.issue_approval_challenge(
        run.run_id,
        approval_id="apr_immutable",
        action_digest="digest-immutable",
        ttl_seconds=60,
    )

    first = store.consume_approval_with_event(
        challenge.challenge_id,
        approval_id=challenge.approval_id,
        action_digest=challenge.action_digest,
        choice="once",
    )
    second = store.consume_approval_with_event(
        challenge.challenge_id,
        approval_id=challenge.approval_id,
        action_digest=challenge.action_digest,
        choice="deny",
    )

    assert first is not None
    assert second is None
    assert store.get_approval_challenge(challenge.challenge_id)["consumed"] == 1
    decisions = [
        event
        for event in store.replay_events(run.run_id)
        if event.event_type == "approval.decision_recorded"
    ]
    assert len(decisions) == 1
    assert decisions[0].payload["choice"] == "once"


@pytest.mark.parametrize(
    "failed_method",
    ["append_event", "record_run_outcome", "transition"],
)
def test_finalize_run_rolls_back_every_canonical_fact_on_failure(
    store, monkeypatch, failed_method
) -> None:
    run = store.submit_or_get(
        idempotency_key=f"k-finalize-{failed_method}", request_body=_BODY_A
    )
    assert store.transition(run.run_id, RunState.RUNNING)

    def _fail(*args, **kwargs):
        raise OSError(f"{failed_method} failed")

    monkeypatch.setattr(store, failed_method, _fail)
    with pytest.raises(OSError, match="failed"):
        store.finalize_run(
            run.run_id,
            terminal_state=RunState.SUCCEEDED,
            event_type="run.completed",
            event_payload={"event": "run.completed", "run_id": run.run_id},
            actual_policy={"model": "actual"},
            usage={"total_tokens": 3},
        )

    row = store.get_run(run.run_id)
    assert row["status"] == RunState.RUNNING.value
    assert row["actual_policy"] is None
    assert row["usage_json"] is None
    assert store.replay_events(run.run_id) == []


def test_finalize_run_is_idempotent_without_duplicate_terminal_event(store) -> None:
    run = store.submit_or_get(idempotency_key="k-finalize-once", request_body=_BODY_A)
    assert store.transition(run.run_id, RunState.RUNNING)
    kwargs = {
        "terminal_state": RunState.SUCCEEDED,
        "event_type": "run.completed",
        "event_payload": {"event": "run.completed", "run_id": run.run_id},
        "actual_policy": {"model": "actual"},
        "usage": {"total_tokens": 3},
    }

    first = store.finalize_run(run.run_id, **kwargs)
    second = store.finalize_run(run.run_id, **kwargs)

    assert first is not None
    assert second is None
    assert [event.event_type for event in store.replay_events(run.run_id)] == [
        "run.completed"
    ]


def test_terminal_run_rejects_events_before_and_after_reopen(store, tmp_path) -> None:
    run = store.submit_or_get(idempotency_key="k-terminal-fence", request_body=_BODY_A)
    assert store.transition(run.run_id, RunState.RUNNING)
    store.finalize_run(
        run.run_id,
        terminal_state=RunState.SUCCEEDED,
        event_type="run.completed",
        event_payload={"event": "run.completed", "run_id": run.run_id},
    )
    with pytest.raises(TerminalStateError):
        store.append_event(run.run_id, "message.delta", {"delta": "late"})
    store.close()

    reopened = _reopen(tmp_path)
    try:
        with pytest.raises(TerminalStateError):
            reopened.append_event(run.run_id, "message.delta", {"delta": "later"})
    finally:
        reopened.close()


def test_terminal_run_rolls_back_approval_consume_when_decision_cannot_append(store) -> None:
    run = store.submit_or_get(idempotency_key="k-terminal-approval", request_body=_BODY_A)
    assert store.transition(run.run_id, RunState.RUNNING)
    challenge = store.issue_approval_challenge(
        run.run_id,
        approval_id="apr_terminal",
        action_digest="digest-terminal",
        ttl_seconds=60,
    )
    assert store.transition(run.run_id, RunState.STOPPED)

    assert (
        store.consume_approval_with_event(
            challenge.challenge_id,
            approval_id=challenge.approval_id,
            action_digest=challenge.action_digest,
            choice="once",
        )
        is None
    )

    assert store.get_approval_challenge(challenge.challenge_id)["consumed"] == 0


def test_stop_intent_blocks_new_approval_challenge(store) -> None:
    run = store.submit_or_get(
        idempotency_key="k-stop-before-challenge", request_body=_BODY_A
    )
    assert store.transition(run.run_id, RunState.RUNNING)
    store.append_event(
        run.run_id,
        "run.stop_requested",
        {"event": "run.stop_requested", "run_id": run.run_id},
    )

    with pytest.raises(RuntimeError, match="not stopping"):
        store.issue_approval_challenge(
            run.run_id,
            approval_id="apr_too_late",
            action_digest="digest-too-late",
            ttl_seconds=60,
        )

    assert store._conn.execute(
        "SELECT COUNT(*) FROM approval_grants WHERE run_id = ?", (run.run_id,)
    ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# row 6: approval challenge + TTL + single-use + CAS
# ---------------------------------------------------------------------------


def test_existing_approval_schema_adds_exact_binding_column(tmp_path) -> None:
    db_path = tmp_path / "pre_exact_approval.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE approval_grants ("
        " challenge_id TEXT PRIMARY KEY,"
        " run_id TEXT NOT NULL,"
        " action_digest TEXT NOT NULL,"
        " expires_at REAL NOT NULL,"
        " consumed INTEGER NOT NULL DEFAULT 0,"
        " created_at REAL NOT NULL)"
    )
    connection.close()

    migrated = DurableRunStore(db_path)
    try:
        columns = {
            row["name"]
            for row in migrated._conn.execute(
                "PRAGMA table_info(approval_grants)"
            ).fetchall()
        }
        assert "approval_id" in columns
    finally:
        migrated.close()


def test_approval_challenge_roundtrip(store) -> None:
    r = store.submit_or_get(idempotency_key="k-appr", request_body=_BODY_A)
    assert store.transition(r.run_id, RunState.RUNNING)
    ch = store.issue_approval_challenge(
        r.run_id, action_digest="sha256:rm-rf", ttl_seconds=60
    )
    assert ch.challenge_id
    assert ch.action_digest == "sha256:rm-rf"

    # correct digest resolves (single-use consume)
    ok = store.consume_approval(ch.challenge_id, action_digest="sha256:rm-rf")
    assert ok is True


def test_approval_challenge_is_bound_to_exact_pending_entry(store) -> None:
    run = store.submit_or_get(idempotency_key="k-exact-appr", request_body=_BODY_A)
    assert store.transition(run.run_id, RunState.RUNNING)
    challenge = store.issue_approval_challenge(
        run.run_id,
        approval_id="apr_exact",
        action_digest="digest-exact",
        ttl_seconds=60,
    )

    assert challenge.approval_id == "apr_exact"
    assert store.consume_approval(
        challenge.challenge_id,
        approval_id="apr_other",
        action_digest="digest-exact",
    ) is False
    assert store.consume_approval(
        challenge.challenge_id,
        approval_id="apr_exact",
        action_digest="digest-exact",
    ) is True


def test_approval_is_single_use(store) -> None:
    r = store.submit_or_get(idempotency_key="k-single", request_body=_BODY_A)
    assert store.transition(r.run_id, RunState.RUNNING)
    ch = store.issue_approval_challenge(r.run_id, action_digest="d1", ttl_seconds=60)
    assert store.consume_approval(ch.challenge_id, action_digest="d1") is True
    # second consume of the same grant must fail (single-use).
    assert store.consume_approval(ch.challenge_id, action_digest="d1") is False


def test_approval_digest_mismatch_rejected(store) -> None:
    r = store.submit_or_get(idempotency_key="k-mism", request_body=_BODY_A)
    assert store.transition(r.run_id, RunState.RUNNING)
    ch = store.issue_approval_challenge(r.run_id, action_digest="real", ttl_seconds=60)
    # a forged/different action digest must not consume the grant.
    assert store.consume_approval(ch.challenge_id, action_digest="forged") is False
    # and the grant is NOT consumed by the failed attempt (CAS), so the real
    # digest still works afterwards.
    assert store.consume_approval(ch.challenge_id, action_digest="real") is True


def test_approval_ttl_expiry(store) -> None:
    r = store.submit_or_get(idempotency_key="k-ttl", request_body=_BODY_A)
    assert store.transition(r.run_id, RunState.RUNNING)
    ch = store.issue_approval_challenge(r.run_id, action_digest="d", ttl_seconds=0)
    # ttl_seconds=0 => already expired.
    time.sleep(0.01)
    assert store.consume_approval(ch.challenge_id, action_digest="d") is False


def test_approval_grants_survive_reopen(store, tmp_path) -> None:
    r = store.submit_or_get(idempotency_key="k-ap", request_body=_BODY_A)
    assert store.transition(r.run_id, RunState.RUNNING)
    ch = store.issue_approval_challenge(r.run_id, action_digest="dx", ttl_seconds=3600)
    cid = ch.challenge_id
    store.close()

    reopened = _reopen(tmp_path)
    try:
        # an unconsumed, unexpired grant is still consumable after restart.
        assert reopened.consume_approval(cid, action_digest="dx") is True
    finally:
        reopened.close()
