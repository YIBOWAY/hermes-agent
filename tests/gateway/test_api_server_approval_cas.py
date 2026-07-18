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
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()


class TestApprovalCAS:
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
    async def test_replay_consume_rejected(self, store):
        """A consumed grant cannot be replayed (single-use): 2nd POST ⇒ 409."""
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
                # Replay the exact same grant ⇒ rejected (already consumed).
                replay = await cli.post(f"/v1/runs/{run_id}/approval", json=payload)
                assert replay.status == 409
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

                # Retry with a real pending entry must still succeed (grant intact).
                entry2 = _pending_entry()
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry2]
                good = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "challenge_id": ch["challenge_id"],
                        "action_digest": ch["action_digest"],
                    },
                )
                assert good.status == 200
                assert entry2.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()

    @pytest.mark.asyncio
    async def test_resolve_zero_restores_grant(self, store):
        """If resolve returns 0 after consume (TOCTOU), grant is restored."""
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

                # has_blocking_approval True, but resolve forced to 0.
                with patch(
                    "tools.approval.resolve_gateway_approval", return_value=0
                ), patch(
                    "tools.approval.has_blocking_approval", return_value=True
                ):
                    resp = await cli.post(
                        f"/v1/runs/{run_id}/approval",
                        json={
                            "choice": "once",
                            "challenge_id": ch["challenge_id"],
                            "action_digest": ch["action_digest"],
                        },
                    )
                assert resp.status == 409
                row = dict(
                    store._conn.execute(
                        "SELECT consumed FROM approval_grants WHERE challenge_id = ?",
                        (ch["challenge_id"],),
                    ).fetchone()
                )
                assert row["consumed"] == 0
                assert not entry.event.is_set()
            finally:
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(run_id, None)
                interrupted.set()
