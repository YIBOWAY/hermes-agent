"""V2.7 — approval challenge + TTL + single-use + CAS (contract matrix §2 row 6).

A gated action must issue an **exact challenge** (bound to the run + a digest of
the action), the grant is **time-boxed (TTL)** and **consume-once**, and the
consume is **compare-and-swap** — so a stale / expired / digest-mismatched /
replayed grant can never resolve the queue, and a failed attempt leaves the real
grant intact.

Broker-only (durable store configured); the legacy in-memory approval path is
byte-identical when no store is present.

Red line: hermetic only — durable store injected with a throwaway ``tmp_path``
DB; no live state, no network. Fails until ``_approval_notify`` issues a durable
challenge and ``_handle_run_approval`` consumes it via CAS.
"""

import asyncio
import threading

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod

_TIMEOUT = aiohttp.ClientTimeout(total=20, sock_connect=5, sock_read=15)


def _make_adapter(durable_store=None) -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={})
    return APIServerAdapter(config, durable_store=durable_store)


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


@pytest.fixture()
def store(tmp_path):
    s = DurableRunStore(db_path=tmp_path / "durable_runs.db")
    yield s
    s.close()


def _slow_agent():
    """Agent that blocks until interrupted (so a run stays live for approval)."""
    ready = threading.Event()
    interrupted = threading.Event()
    mock_agent = MagicMock()
    mock_agent.interrupt = MagicMock(side_effect=lambda message=None: interrupted.set())

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        interrupted.wait(timeout=10)
        return {"final_response": "done"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    return mock_agent, ready, interrupted


def _pending_entry(command="bash -c rm -rf /tmp/x", description="danger"):
    return approval_mod._ApprovalEntry({
        "command": command,
        "description": description,
        "pattern_keys": ["shell-c"],
    })


async def _start_live_run(adapter, cli):
    """Start a run whose agent blocks; return (run_id, mock_agent, interrupted)."""
    mock_agent, ready, interrupted = _slow_agent()
    with patch.object(adapter, "_create_agent") as mock_create:
        mock_create.return_value = mock_agent
        resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 202
        run_id = (await resp.json())["run_id"]
        assert ready.wait(timeout=3.0)
    return run_id, mock_agent, interrupted


class TestChallengeIssued:
    @pytest.mark.asyncio
    async def test_approval_request_carries_challenge(self, store):
        """approval.request event carries challenge_id + action_digest + expires_at."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]

                # Trigger the gateway notify so an approval.request is issued.
                notify = approval_mod._gateway_notify_cbs.get(run_id)
                if notify is not None:
                    notify(dict(entry.data))
                await asyncio.sleep(0)

                # The store must hold a challenge for this run bound to the action.
                challenges = [
                    row for row in store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchall()
                ]
                assert challenges, "an approval.request must issue a durable challenge"
                ch = dict(challenges[0])
                assert ch["action_digest"]
                assert ch["expires_at"] > 0
                request_event = next(
                    event
                    for event in store.replay_events(run_id)
                    if event.event_type == "approval.request"
                )
                assert "pattern_key" not in request_event.payload
                assert "pattern_keys" not in request_event.payload
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_challenge_store_failure_never_publishes_unanswerable_request(
        self, store, monkeypatch
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                notify = approval_mod._gateway_notify_cbs[run_id]

                def _fail(*args, **kwargs):
                    raise OSError("approval store unavailable")

                monkeypatch.setattr(store, "issue_approval_challenge", _fail)
                with pytest.raises(RuntimeError, match="challenge unavailable"):
                    notify(dict(_pending_entry().data))

                await asyncio.sleep(0)
                assert store.get_run(run_id)["status"] == "running"
                assert not any(
                    event.event_type == "approval.request"
                    for event in store.replay_events(run_id)
                )
            finally:
                interrupted.set()


class TestApprovalCAS:
    @pytest.mark.asyncio
    async def test_challenge_resolves_only_its_exact_concurrent_entry(self, store):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                first = _pending_entry(command="first dangerous command")
                second = _pending_entry(command="second dangerous command")
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [first, second]

                notify = approval_mod._gateway_notify_cbs.get(run_id)
                assert notify is not None
                notify(dict(first.data))
                notify(dict(second.data))
                challenges = {
                    row["approval_id"]: dict(row)
                    for row in store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchall()
                }
                challenge = challenges[second.approval_id]

                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": challenge["challenge_id"],
                        "action_digest": challenge["action_digest"],
                        "resolve_all": True,
                    },
                )

                assert response.status == 200
                response_body = await response.json()
                assert response_body["decision_status"] == "committed"
                assert response_body["waiter_signal_status"] == "confirmed"
                assert "resolved" not in response_body
                assert second.event.is_set()
                assert not first.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_missing_challenge_id_rejected(self, store):
        """POST without challenge_id ⇒ 409, queue NOT resolved (fail closed)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]

                resp = await cli.post(f"/v1/runs/{run_id}/approval", json={"choice": "once"})
                assert resp.status == 409
                assert not entry.event.is_set()  # never resolved
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_digest_mismatch_rejected_and_grant_intact(self, store):
        """Forged digest ⇒ 409, grant NOT consumed, real digest still works (CAS)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]

                # Issue the real challenge (as the gateway would).
                notify = approval_mod._gateway_notify_cbs.get(run_id)
                if notify is not None:
                    notify(dict(entry.data))
                ch = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )

                # Forged digest ⇒ rejected, grant intact.
                bad = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "challenge_id": ch["challenge_id"],
                          "action_digest": "forged"},
                )
                assert bad.status == 409
                assert not entry.event.is_set()

                # Real digest ⇒ resolves.
                good = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "challenge_id": ch["challenge_id"],
                          "action_digest": ch["action_digest"]},
                )
                assert good.status == 200
                assert entry.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_exact_completed_replay_is_idempotent_without_second_decision(self, store):
        """Lost HTTP ACK can replay the same immutable decision, never rewrite it."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                notify = approval_mod._gateway_notify_cbs.get(run_id)
                if notify is not None:
                    notify(dict(entry.data))
                ch = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                payload = {"choice": "once", "challenge_id": ch["challenge_id"],
                           "action_digest": ch["action_digest"]}
                first = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                assert first.status == 200
                before = list(store.replay_events(run_id))
                replay = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                assert replay.status == 200
                assert (await replay.json())["idempotent_replay"] is True
                assert len(store.replay_events(run_id)) == len(before)
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_expired_grant_rejected(self, store):
        """An expired grant ⇒ 409 (TTL enforced), queue NOT resolved."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                # Issue an already-expired challenge directly (ttl=0).
                ch = store.issue_approval_challenge(
                    run_id, action_digest="d-exp", ttl_seconds=0
                )
                await asyncio.sleep(0.01)
                resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "challenge_id": ch.challenge_id,
                          "action_digest": "d-exp"},
                )
                assert resp.status == 409
                assert not entry.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()


class TestLegacyApprovalUnchanged:
    @pytest.mark.asyncio
    async def test_no_store_legacy_resolve_still_works(self):
        """With no durable store, approval resolves without any challenge (legacy)."""
        adapter = _make_adapter(durable_store=None)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                resp = await cli.post(f"/v1/runs/{run_id}/approval", json={"choice": "once"})
                assert resp.status == 200
                assert entry.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()


class TestApprovalConsumeDoesNotBurnOnNonSuccess:
    """A6: non-success approval paths must not change the prior grant fact."""

    @pytest.mark.asyncio
    async def test_not_pending_leaves_grant_intact_for_retry(self, store):
        """No queue entry → 409 approval_not_pending AND grant still consumable."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                # Issue a real challenge via notify path, but do NOT enqueue a
                # pending approval entry — simulates drained/raced queue.
                entry = _pending_entry()
                notify = approval_mod._gateway_notify_cbs.get(run_id)
                assert notify is not None
                notify(dict(entry.data))
                ch = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                assert ch["consumed"] == 0

                # Empty the queue deliberately (session key exists, nothing pending).
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)

                bad = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": ch["challenge_id"],
                        "action_digest": ch["action_digest"],
                    },
                )
                assert bad.status == 409
                body = await bad.json()
                err = body.get("error") or body
                code = err.get("code") if isinstance(err, dict) else None
                assert code in {"approval_not_pending", "approval_not_active"} or (
                    "pending" in str(body).lower() or "active" in str(body).lower()
                )

                row = dict(
                    store._conn.execute(
                        "SELECT consumed FROM approval_grants WHERE challenge_id = ?",
                        (ch["challenge_id"],),
                    ).fetchone()
                )
                assert row["consumed"] == 0, "grant must not burn on non-success"

                # Retry with the exact pending entry must still succeed (grant intact).
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                good = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": ch["challenge_id"],
                        "action_digest": ch["action_digest"],
                    },
                )
                assert good.status == 200
                assert entry.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_decision_append_failure_does_not_signal_or_consume(self, store, monkeypatch):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                notify = approval_mod._gateway_notify_cbs.get(run_id)
                assert notify is not None
                notify(dict(entry.data))
                ch = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )

                await asyncio.sleep(0)
                original_append = store.append_event

                def _fail_decision(run_id_arg, event_type, payload):
                    if event_type == "approval.decision_recorded":
                        assert entry.claimed is True
                        with approval_mod._lock:
                            assert entry not in approval_mod._gateway_queues.get(
                                run_id, []
                            )
                        raise OSError("decision event unavailable")
                    return original_append(run_id_arg, event_type, payload)

                monkeypatch.setattr(store, "append_event", _fail_decision)
                resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": ch["challenge_id"],
                        "action_digest": ch["action_digest"],
                    },
                )
                assert resp.status == 503
                row = dict(
                    store._conn.execute(
                        "SELECT consumed FROM approval_grants WHERE challenge_id = ?",
                        (ch["challenge_id"],),
                    ).fetchone()
                )
                assert row["consumed"] == 0
                assert not entry.event.is_set()
                assert store.get_approval_decision(ch["challenge_id"]) is None
                assert entry.claimed is False
                with approval_mod._lock:
                    assert entry in approval_mod._gateway_queues[run_id]
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_stop_intent_rejects_exact_approval_without_signalling(
        self, store
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, _mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                approval_mod._gateway_notify_cbs[run_id](dict(entry.data))
                await asyncio.sleep(0)
                challenge = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                store.append_event(
                    run_id,
                    "run.stop_requested",
                    {"event": "run.stop_requested", "run_id": run_id},
                )

                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": challenge["challenge_id"],
                        "action_digest": challenge["action_digest"],
                    },
                )

                assert response.status == 409
                assert not entry.event.is_set()
                assert entry.claimed is False
                assert store.get_approval_challenge(challenge["challenge_id"])[
                    "consumed"
                ] == 0
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_live_delivery_exception_is_recoverable_not_false_success(
        self, store, monkeypatch
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, _mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                approval_mod._gateway_notify_cbs[run_id](dict(entry.data))
                await asyncio.sleep(0)
                challenge = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                original_finalize = approval_mod.finalize_gateway_approval_claim
                monkeypatch.setattr(
                    approval_mod,
                    "finalize_gateway_approval_claim",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("delivery failed")
                    ),
                )

                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": challenge["challenge_id"],
                        "action_digest": challenge["action_digest"],
                    },
                )

                assert response.status == 503
                assert not entry.event.is_set()
                assert entry.result is None
                assert entry.claimed is False
                assert store.get_approval_response(challenge["challenge_id"]) is not None
                assert store.get_approval_delivery(challenge["challenge_id"]) is None
                with approval_mod._lock:
                    assert entry in approval_mod._gateway_queues[run_id]

                monkeypatch.setattr(
                    approval_mod,
                    "finalize_gateway_approval_claim",
                    original_finalize,
                )
                recovered = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": challenge["challenge_id"],
                        "action_digest": challenge["action_digest"],
                    },
                )
                assert recovered.status == 200
                assert entry.event.is_set()
                assert store.get_approval_delivery(challenge["challenge_id"]) is not None
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_release_commit_before_signal_reports_unknown_then_recovers_exact_waiter(
        self, store, monkeypatch
    ):
        """A crash gap after durable release commit is not delivery evidence."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, _mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                approval_mod._gateway_notify_cbs[run_id](dict(entry.data))
                await asyncio.sleep(0)
                challenge = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                original_finalize = approval_mod.finalize_gateway_approval_claim

                def _commit_then_crash(_entry, _choice, *, before_signal=None, **_kwargs):
                    assert before_signal is not None
                    before_signal()
                    raise RuntimeError("crash after release commit before Event.set")

                monkeypatch.setattr(
                    approval_mod,
                    "finalize_gateway_approval_claim",
                    _commit_then_crash,
                )
                payload = {
                    "choice": "once",
                    "challenge_id": challenge["challenge_id"],
                    "action_digest": challenge["action_digest"],
                }

                first = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                first_body = await first.json()

                assert first.status == 200
                assert first_body["decision_status"] == "committed"
                assert first_body["waiter_signal_status"] == "unknown"
                assert "resolved" not in first_body
                assert not entry.event.is_set()
                events = store.replay_events(run_id)
                assert len(
                    [e for e in events if e.event_type == "approval.release_committed"]
                ) == 1
                assert not any(e.event_type == "approval.signalled" for e in events)

                monkeypatch.setattr(
                    approval_mod,
                    "finalize_gateway_approval_claim",
                    original_finalize,
                )
                recovered = await cli.post(
                    f"/v1/runs/{run_id}/approval", json=payload
                )
                recovered_body = await recovered.json()

                assert recovered.status == 200
                assert recovered_body["decision_status"] == "committed"
                assert recovered_body["waiter_signal_status"] == "confirmed"
                assert "resolved" not in recovered_body
                assert entry.event.is_set()
                events = store.replay_events(run_id)
                assert len(
                    [e for e in events if e.event_type == "approval.release_committed"]
                ) == 1
                assert len(
                    [e for e in events if e.event_type == "approval.signalled"]
                ) == 1
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_stop_in_release_commit_signal_gap_never_signals_waiter(
        self, store, monkeypatch
    ):
        """A stop committed in the crash gap wins over same-choice recovery."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, _mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                approval_mod._gateway_notify_cbs[run_id](dict(entry.data))
                await asyncio.sleep(0)
                challenge = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )

                def _commit_then_crash(_entry, _choice, *, before_signal=None, **_kwargs):
                    assert before_signal is not None
                    before_signal()
                    raise RuntimeError("crash after release commit before Event.set")

                monkeypatch.setattr(
                    approval_mod,
                    "finalize_gateway_approval_claim",
                    _commit_then_crash,
                )
                payload = {
                    "choice": "once",
                    "challenge_id": challenge["challenge_id"],
                    "action_digest": challenge["action_digest"],
                }
                first = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                assert first.status == 200
                assert (await first.json())["waiter_signal_status"] == "unknown"
                assert not entry.event.is_set()

                store.append_event(
                    run_id,
                    "run.stop_requested",
                    {"event": "run.stop_requested", "run_id": run_id},
                )
                replay = await cli.post(
                    f"/v1/runs/{run_id}/approval", json=payload
                )
                replay_body = await replay.json()

                assert replay.status == 200
                assert replay_body["decision_status"] == "committed"
                assert replay_body["waiter_signal_status"] == "unknown"
                assert "resolved" not in replay_body
                assert not entry.event.is_set()
                assert not any(
                    event.event_type == "approval.signalled"
                    for event in store.replay_events(run_id)
                )
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_response_append_failure_recovers_only_same_choice(
        self, store, monkeypatch
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                notify = approval_mod._gateway_notify_cbs[run_id]
                notify(dict(entry.data))
                await asyncio.sleep(0)
                ch = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                original_append = store.append_event

                def _fail_response(run_id_arg, event_type, payload):
                    if event_type == "approval.responded":
                        raise OSError("response event unavailable")
                    return original_append(run_id_arg, event_type, payload)

                monkeypatch.setattr(store, "append_event", _fail_response)
                payload = {
                    "choice": "once",
                    "challenge_id": ch["challenge_id"],
                    "action_digest": ch["action_digest"],
                }
                failed = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                assert failed.status == 503
                assert not entry.event.is_set()
                assert store.get_approval_challenge(ch["challenge_id"])["consumed"] == 1
                decision = store.get_approval_decision(ch["challenge_id"])
                assert decision is not None
                assert decision.payload["choice"] == "once"

                changed = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={**payload, "choice": "deny"},
                )
                assert changed.status == 409
                assert not entry.event.is_set()

                monkeypatch.setattr(store, "append_event", original_append)
                recovered = await cli.post(
                    f"/v1/runs/{run_id}/approval", json=payload
                )
                assert recovered.status == 200
                recovered_body = await recovered.json()
                assert recovered_body["approval_id"] == ch["approval_id"]
                assert entry.event.is_set()

                replay = store.replay_events(run_id)
                decisions = [
                    event for event in replay
                    if event.event_type == "approval.decision_recorded"
                ]
                responses = [
                    event for event in replay
                    if event.event_type == "approval.responded"
                ]
                deliveries = [
                    event for event in replay
                    if event.event_type == "approval.signalled"
                ]
                releases = [
                    event for event in replay
                    if event.event_type == "approval.release_committed"
                ]
                assert len(decisions) == len(responses) == 1
                assert len(releases) == 1
                assert len(deliveries) == 1
                assert responses[0].payload["decision_status"] == "committed"
                assert responses[0].payload["waiter_signal_status"] == "unknown"
                assert "resolved" not in responses[0].payload
                assert responses[0].seq < releases[0].seq < deliveries[0].seq
                assert responses[0].payload["challenge_id"] == ch["challenge_id"]
                assert responses[0].payload["approval_id"] == ch["approval_id"]
                assert responses[0].payload["action_digest"] == ch["action_digest"]

                # Lost HTTP ACK: exact retry is a 200 idempotent replay even
                # though the live queue entry has already been released.
                replayed = await cli.post(
                    f"/v1/runs/{run_id}/approval", json=payload
                )
                assert replayed.status == 200
                assert (await replayed.json())["idempotent_replay"] is True
                assert len(store.replay_events(run_id)) == len(replay)
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_response_failure_then_stop_blocks_same_choice_delivery(
        self, store, monkeypatch
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, _mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                approval_mod._gateway_notify_cbs[run_id](dict(entry.data))
                await asyncio.sleep(0)
                challenge = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                original_append = store.append_event

                def _fail_response(run_id_arg, event_type, payload):
                    if event_type == "approval.responded":
                        raise OSError("response unavailable")
                    return original_append(run_id_arg, event_type, payload)

                monkeypatch.setattr(store, "append_event", _fail_response)
                payload = {
                    "choice": "once",
                    "challenge_id": challenge["challenge_id"],
                    "action_digest": challenge["action_digest"],
                }
                first = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                assert first.status == 503
                assert store.get_approval_decision(challenge["challenge_id"]) is not None
                assert not entry.event.is_set()

                monkeypatch.setattr(store, "append_event", original_append)
                store.append_event(
                    run_id,
                    "run.stop_requested",
                    {"event": "run.stop_requested", "run_id": run_id},
                )
                replay = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)

                assert replay.status == 200
                replay_body = await replay.json()
                assert replay_body["decision_status"] == "committed"
                assert replay_body["waiter_signal_status"] == "unknown"
                assert "resolved" not in replay_body
                assert not entry.event.is_set()
                assert store.get_approval_response(challenge["challenge_id"]) is None
                assert store.get_approval_delivery(challenge["challenge_id"]) is None
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_finalizer_false_never_commits_release_or_signals(
        self, store, monkeypatch
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id, _mock_agent, interrupted = await _start_live_run(adapter, cli)
            try:
                entry = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]
                approval_mod._gateway_notify_cbs[run_id](dict(entry.data))
                await asyncio.sleep(0)
                challenge = dict(
                    store._conn.execute(
                        "SELECT * FROM approval_grants WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
                monkeypatch.setattr(
                    approval_mod,
                    "finalize_gateway_approval_claim",
                    lambda *_args, **_kwargs: False,
                )
                payload = {
                    "choice": "once",
                    "challenge_id": challenge["challenge_id"],
                    "action_digest": challenge["action_digest"],
                }

                first = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                replay = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)

                assert first.status == 503
                assert replay.status == 503
                assert not entry.event.is_set()
                assert entry.result is None
                assert store.get_approval_response(challenge["challenge_id"]) is not None
                assert store.get_approval_delivery(challenge["challenge_id"]) is None
                assert store.get_approval_release(challenge["challenge_id"]) is None
                assert "resolved" not in store.get_approval_response(
                    challenge["challenge_id"]
                ).payload
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()
