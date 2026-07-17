"""V2.2 — Idempotency-Key + canonical digest on POST /v1/runs (TDD red).

Contract matrix §2 rows 1 & 2: ``POST /v1/runs`` must honor the caller's
``Idempotency-Key`` and compute+persist the *canonical* request digest as part
of run identity. Two submissions share identity iff both key and digest match.

* Same identity (same key + same semantic body) ⇒ returns the **same** Run and
  does NOT spawn a duplicate (submit-or-get).
* Same key with a **different** body ⇒ **409 Conflict** (fail closed).
* Key-order-only differences in the body ⇒ **same** Run (canonical digest, not
  ``repr()``).
* Durability: a fresh adapter on the **same** DB path recovers the **same** Run
  after "restart" (contract row 3 underlies row 2's by-identity recovery).
* No ``Idempotency-Key`` ⇒ legacy behavior is unchanged (fresh run each time).

Red line: hermetic only — the durable store is injected with a throwaway
``tmp_path`` DB; no live state, no network. These tests fail until
``APIServerAdapter`` accepts an optional ``durable_store`` and wires
submit-or-get into ``_handle_runs``.
"""

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


def _make_adapter(durable_store=None, api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
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


def _completed_agent_mock():
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": "done"}
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    return mock_agent


# ---------------------------------------------------------------------------
# row 1/2: submit-or-get by request identity
# ---------------------------------------------------------------------------


class TestIdempotentSubmit:
    @pytest.mark.asyncio
    async def test_same_identity_returns_same_run_no_duplicate(self, store):
        """Same Idempotency-Key + same body ⇒ same run_id, agent created once."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                headers = {"Idempotency-Key": "idem-1"}
                r1 = await cli.post("/v1/runs", json={"input": "hello"}, headers=headers)
                r2 = await cli.post("/v1/runs", json={"input": "hello"}, headers=headers)

                assert r1.status == 202
                assert r2.status == 202
                d1 = await r1.json()
                d2 = await r2.json()
                assert d1["run_id"] == d2["run_id"]
                # The replay must not start a second agent run.
                assert mock_create.call_count == 1
                # Replay is explicitly marked so callers can tell it apart.
                assert d2.get("idempotent_replay") is True

    @pytest.mark.asyncio
    async def test_same_key_different_body_conflicts_409(self, store):
        """Same key, different digest ⇒ 409 (fail closed), no run spawned."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                r1 = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers={"Idempotency-Key": "idem-c"}
                )
                r2 = await cli.post(
                    "/v1/runs", json={"input": "different"}, headers={"Idempotency-Key": "idem-c"}
                )
                assert r1.status == 202
                assert r2.status == 409
                # The conflicting request must not spawn a second run.
                assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_canonical_digest_key_order_independent(self, store):
        """Body differing only in key order ⇒ SAME run (canonical digest, not repr())."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        body1 = {"input": "hi", "model": "m", "stream": False}
        body2 = {"stream": False, "model": "m", "input": "hi"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                headers = {"Idempotency-Key": "idem-canon"}
                r1 = await cli.post("/v1/runs", json=body1, headers=headers)
                r2 = await cli.post("/v1/runs", json=body2, headers=headers)
                d1 = await r1.json()
                d2 = await r2.json()
                assert d1["run_id"] == d2["run_id"]
                assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_recovered_run_is_not_reexecuted(self, store):
        """A replayed submission must never re-register or re-spawn the run."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                headers = {"Idempotency-Key": "idem-rec"}
                r1 = await cli.post("/v1/runs", json={"input": "hello"}, headers=headers)
                run_id = (await r1.json())["run_id"]
                registered = dict(adapter._run_statuses)

                r2 = await cli.post("/v1/runs", json={"input": "hello"}, headers=headers)
                assert r2.status == 202
                assert (await r2.json())["run_id"] == run_id
                # No new run registered; no extra agent built.
                assert adapter._run_statuses.keys() == registered.keys()
                assert mock_create.call_count == 1
    @pytest.mark.asyncio
    async def test_oversized_idempotency_key_rejected_400(self, store):
        """An oversized Idempotency-Key is rejected (fail closed), not persisted."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "k" * 500},
                )
                assert resp.status == 400
                assert mock_create.call_count == 0


# ---------------------------------------------------------------------------
# row 3 (underlies row 2): durability across restart
# ---------------------------------------------------------------------------


class TestIdempotencyDurability:
    @pytest.mark.asyncio
    async def test_same_identity_recovered_after_restart(self, tmp_path):
        """A fresh adapter on the same DB recovers the SAME Run (no new spawn)."""
        db = tmp_path / "durable_runs.db"
        headers = {"Idempotency-Key": "idem-persist"}
        body = {"input": "hello"}

        store1 = DurableRunStore(db_path=db)
        adapter1 = _make_adapter(durable_store=store1)
        app1 = _create_runs_app(adapter1)
        async with TestClient(TestServer(app1)) as cli1:
            with patch.object(adapter1, "_create_agent") as mock_create1:
                mock_create1.return_value = _completed_agent_mock()
                r1 = await cli1.post("/v1/runs", json=body, headers=headers)
                assert r1.status == 202
                run_id = (await r1.json())["run_id"]
                assert mock_create1.call_count == 1
        store1.close()

        # "Restart": brand-new adapter + store on the SAME DB file.
        store2 = DurableRunStore(db_path=db)
        adapter2 = _make_adapter(durable_store=store2)
        app2 = _create_runs_app(adapter2)
        async with TestClient(TestServer(app2)) as cli2:
            with patch.object(adapter2, "_create_agent") as mock_create2:
                mock_create2.return_value = _completed_agent_mock()
                r2 = await cli2.post("/v1/runs", json=body, headers=headers)
                assert r2.status == 202
                d2 = await r2.json()
                assert d2["run_id"] == run_id  # recovered, not a new run
                assert d2.get("idempotent_replay") is True
                assert mock_create2.call_count == 0  # never re-executed
        store2.close()


# ---------------------------------------------------------------------------
# legacy behavior preserved when no Idempotency-Key is present
# ---------------------------------------------------------------------------


class TestLegacySubmitUnchanged:
    @pytest.mark.asyncio
    async def test_no_key_still_creates_fresh_run_each_time(self, store):
        """Without Idempotency-Key, each POST mints a NEW run (legacy behavior)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                r1 = await cli.post("/v1/runs", json={"input": "hello"})
                r2 = await cli.post("/v1/runs", json={"input": "hello"})
                d1 = await r1.json()
                d2 = await r2.json()
                assert r1.status == 202 and r2.status == 202
                assert d1["run_id"] != d2["run_id"]
                assert d1["run_id"].startswith("run_")
                assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_no_store_no_key_behaves_legacy(self):
        """No durable store configured at all ⇒ legacy in-memory-only behavior."""
        adapter = _make_adapter(durable_store=None)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")
                assert mock_create.call_count == 1
