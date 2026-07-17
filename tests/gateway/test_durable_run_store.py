"""V2.4 DurableRunAuthority durable-store tests (contract matrix §2 rows 2,3,4,6,9).

These pin the store contract BEFORE the implementation exists (TDD red). The
store is the shared地基 for submit-or-get, persistent Run identity/status,
stable event id + monotonic cursor replay, and approval challenge/TTL/single-use/
CAS — all durable across process restart (modeled here by reopening the DB).

No live effect: every test uses a throwaway tmp_path DB file.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# row 6: approval challenge + TTL + single-use + CAS
# ---------------------------------------------------------------------------


def test_approval_challenge_roundtrip(store) -> None:
    r = store.submit_or_get(idempotency_key="k-appr", request_body=_BODY_A)
    ch = store.issue_approval_challenge(
        r.run_id, action_digest="sha256:rm-rf", ttl_seconds=60
    )
    assert ch.challenge_id
    assert ch.action_digest == "sha256:rm-rf"

    # correct digest resolves (single-use consume)
    ok = store.consume_approval(ch.challenge_id, action_digest="sha256:rm-rf")
    assert ok is True


def test_approval_is_single_use(store) -> None:
    r = store.submit_or_get(idempotency_key="k-single", request_body=_BODY_A)
    ch = store.issue_approval_challenge(r.run_id, action_digest="d1", ttl_seconds=60)
    assert store.consume_approval(ch.challenge_id, action_digest="d1") is True
    # second consume of the same grant must fail (single-use).
    assert store.consume_approval(ch.challenge_id, action_digest="d1") is False


def test_approval_digest_mismatch_rejected(store) -> None:
    r = store.submit_or_get(idempotency_key="k-mism", request_body=_BODY_A)
    ch = store.issue_approval_challenge(r.run_id, action_digest="real", ttl_seconds=60)
    # a forged/different action digest must not consume the grant.
    assert store.consume_approval(ch.challenge_id, action_digest="forged") is False
    # and the grant is NOT consumed by the failed attempt (CAS), so the real
    # digest still works afterwards.
    assert store.consume_approval(ch.challenge_id, action_digest="real") is True


def test_approval_ttl_expiry(store) -> None:
    r = store.submit_or_get(idempotency_key="k-ttl", request_body=_BODY_A)
    ch = store.issue_approval_challenge(r.run_id, action_digest="d", ttl_seconds=0)
    # ttl_seconds=0 => already expired.
    time.sleep(0.01)
    assert store.consume_approval(ch.challenge_id, action_digest="d") is False


def test_approval_grants_survive_reopen(store, tmp_path) -> None:
    r = store.submit_or_get(idempotency_key="k-ap", request_body=_BODY_A)
    ch = store.issue_approval_challenge(r.run_id, action_digest="dx", ttl_seconds=3600)
    cid = ch.challenge_id
    store.close()

    reopened = _reopen(tmp_path)
    try:
        # an unconsumed, unexpired grant is still consumable after restart.
        assert reopened.consume_approval(cid, action_digest="dx") is True
    finally:
        reopened.close()
