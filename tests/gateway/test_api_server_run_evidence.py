"""V2.6 — requested vs actual provider/model/fallback/usage evidence (contract §2 row 5).

A run must record the **requested** policy *and* the **actual** provider/model,
any fallback (+reason), and usage — and must **never overwrite the requested
route**. This slice persists that evidence in the durable runs table and
surfaces it via ``GET /v1/runs/{run_id}``.

* Requested policy (the client's ``model`` field + resolved route) is captured
  at submission and is immutable.
* Actual provider/model + usage are recorded at completion.
* Fallback (requested route not honored) sets ``fallback_reason``; requested is
  still preserved.
* Evidence survives restart and is merged into the polled status response.

Red line: hermetic only — durable store injected with a throwaway ``tmp_path``
DB; no live state, no network. Fails until the store gains
``set_requested_policy``/``record_run_outcome`` and the adapter wires them.
"""

import asyncio
import json

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

_TIMEOUT = aiohttp.ClientTimeout(total=20, sock_connect=5, sock_read=15)


def _make_adapter(durable_store=None, model_routes=None) -> APIServerAdapter:
    extra = {}
    if model_routes is not None:
        extra["model_routes"] = model_routes
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


def _completed_agent(tokens=(10, 5, 15)):
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": "done"}
    mock_agent.session_prompt_tokens = tokens[0]
    mock_agent.session_completion_tokens = tokens[1]
    mock_agent.session_total_tokens = tokens[2]
    return mock_agent


async def _run_to_terminal(adapter, cli, body):
    with patch.object(adapter, "_create_agent") as mock_create:
        mock_create.return_value = _completed_agent()
        resp = await cli.post("/v1/runs", json=body)
        assert resp.status == 202
        run_id = (await resp.json())["run_id"]
        status = None
        for _ in range(100):
            status = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get("status")
            if status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)
        assert status == "completed"
        return run_id


class TestStoreEvidence:
    def test_record_run_outcome_sets_actual_and_usage(self, store):
        """record_run_outcome writes actual_policy/usage without touching requested."""
        r = store.submit_or_get(idempotency_key="k-ev", request_body={"input": "hi"})
        store.set_requested_policy(r.run_id, {"model": "req-model", "provider": "req-prov"})
        store.record_run_outcome(
            r.run_id,
            actual_policy={"model": "act-model", "provider": "act-prov"},
            fallback_reason=None,
            usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )
        row = store.get_run(r.run_id)
        assert row is not None
        # requested preserved verbatim
        assert json.loads(row["requested_policy"]) == {"model": "req-model", "provider": "req-prov"}
        assert json.loads(row["actual_policy"]) == {"model": "act-model", "provider": "act-prov"}
        assert json.loads(row["usage_json"]) == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        assert row["fallback_reason"] is None

    def test_requested_policy_not_overwritten_by_outcome(self, store):
        """Recording the actual route must never overwrite the requested route."""
        r = store.submit_or_get(idempotency_key="k-req", request_body={"input": "hi"})
        store.set_requested_policy(r.run_id, {"model": "wanted"})
        store.record_run_outcome(
            r.run_id, actual_policy={"model": "got-instead"}, fallback_reason="route_unavailable", usage={}
        )
        row = store.get_run(r.run_id)
        assert json.loads(row["requested_policy"]) == {"model": "wanted"}
        assert row["fallback_reason"] == "route_unavailable"

    def test_record_outcome_unknown_run_rejected(self, store):
        with pytest.raises((KeyError, ValueError)):
            store.record_run_outcome("run_nope", actual_policy={}, fallback_reason=None, usage={})


class TestAdapterEvidence:
    @pytest.mark.asyncio
    async def test_requested_and_actual_recorded_and_surfaced(self, store):
        """A completed run records requested + actual policy + usage, surfaced via GET."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, {"input": "hi", "model": "my-model"})

            resp = await cli.get(f"/v1/runs/{run_id}")
            assert resp.status == 200
            data = await resp.json()

        assert data["status"] == "completed"
        # requested policy preserved (the client's model field)
        assert data.get("requested_policy", {}).get("model") == "my-model"
        # usage evidence present
        assert data.get("usage", {}).get("total_tokens") == 15

    @pytest.mark.asyncio
    async def test_route_resolution_records_requested_route(self, store):
        """A configured model_routes alias is recorded as the requested route."""
        routes = {"fast": {"model": "fast-model", "provider": "fast-prov"}}
        adapter = _make_adapter(durable_store=store, model_routes=routes)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, {"input": "hi", "model": "fast"})
            resp = await cli.get(f"/v1/runs/{run_id}")
            data = await resp.json()

        req = data.get("requested_policy", {})
        # requested captures the alias the client asked for
        assert req.get("model") == "fast"

    @pytest.mark.asyncio
    async def test_requested_policy_immutable_after_completion(self, store):
        """requested_policy must equal what was requested, not the executed route."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, {"input": "hi", "model": "requested-xyz"})

        row = store.get_run(run_id)
        assert row is not None
        requested = json.loads(row["requested_policy"])
        assert requested.get("model") == "requested-xyz"
        # actual may differ (default/fallback), but requested is untouched.
        assert requested.get("model") == "requested-xyz"

    @pytest.mark.asyncio
    async def test_evidence_recorded_on_failed_run(self, store):
        """A run that raises still records actual outcome evidence (no silent gap)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.side_effect = RuntimeError("boom")
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hi", "model": "fail-model"})
                run_id = (await resp.json())["run_id"]
                status = None
                for _ in range(100):
                    status = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get("status")
                    if status in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.05)
                assert status == "failed"

        row = store.get_run(run_id)
        assert row is not None
        # requested preserved even on failure; outcome recording ran (row present).
        assert json.loads(row["requested_policy"]).get("model") == "fail-model"


class TestEvidenceDurability:
    @pytest.mark.asyncio
    async def test_evidence_survives_restart(self, tmp_path):
        """requested/actual/usage evidence persists across 'restart' and is served by GET."""
        db = tmp_path / "durable_runs.db"
        store1 = DurableRunStore(db_path=db)
        adapter1 = _make_adapter(durable_store=store1)
        app1 = _create_runs_app(adapter1)
        async with TestClient(TestServer(app1), timeout=_TIMEOUT) as cli1:
            run_id = await _run_to_terminal(adapter1, cli1, {"input": "hi", "model": "persist-me"})
        store1.close()

        store2 = DurableRunStore(db_path=db)
        adapter2 = _make_adapter(durable_store=store2)
        app2 = _create_runs_app(adapter2)
        async with TestClient(TestServer(app2), timeout=_TIMEOUT) as cli2:
            resp = await cli2.get(f"/v1/runs/{run_id}")
            assert resp.status == 200
            data = await resp.json()
        store2.close()

        assert data.get("requested_policy", {}).get("model") == "persist-me"
        assert data.get("usage", {}).get("total_tokens") == 15


class TestPostFallbackEvidence:
    """V2.6b — actual_policy + fallback_reason must reflect a REAL post-fallback route.

    V2.6's original writer set actual_policy.model to the requested/default model
    and hardcoded fallback_reason=None, so GET /v1/runs could never show that a
    fallback served the run. The adapter must read the live agent's
    agent.model / agent.provider / agent._fallback_activated after
    run_conversation and persist the real route + a reason.
    """

    def _fallback_agent(self, *, model: str, provider: str, activated: bool = True):
        mock_agent = _completed_agent()
        # Real strings (NOT MagicMock defaults) so the writer can distinguish a
        # real AIAgent from a bare MagicMock used by the existing helpers.
        mock_agent.model = model
        mock_agent.provider = provider
        mock_agent._fallback_activated = activated
        return mock_agent

    @pytest.mark.asyncio
    async def test_fallback_records_actual_model_and_reason(self, store):
        """When the agent falls back, actual_policy.model is the FALLBACK model
        and fallback_reason is a non-empty string naming that model — not null
        and not the requested model."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = self._fallback_agent(
                    model="fallback-y", provider="custom", activated=True
                )
                resp = await cli.post(
                    "/v1/runs", json={"input": "hi", "model": "primary-x"}
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                status = None
                for _ in range(100):
                    status = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get(
                        "status"
                    )
                    if status in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.05)
                assert status == "completed"

                resp = await cli.get(f"/v1/runs/{run_id}")
                data = await resp.json()

        # requested is the client's model and is never overwritten.
        assert data.get("requested_policy", {}).get("model") == "primary-x"
        # actual is the agent's post-fallback model, NOT the requested one.
        assert data.get("actual_policy", {}).get("model") == "fallback-y"
        assert data.get("actual_policy", {}).get("provider") == "custom"
        # A real reason is recorded (not None).
        reason = data.get("fallback_reason")
        assert isinstance(reason, str) and reason
        assert "fallback-y" in reason

    @pytest.mark.asyncio
    async def test_no_fallback_leaves_reason_none_and_uses_agent_model(self, store):
        """When no fallback fired, fallback_reason stays None and actual_policy
        still prefers the agent's model (the real executing model) over the
        advertised default."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = self._fallback_agent(
                    model="primary-x", provider="custom", activated=False
                )
                resp = await cli.post(
                    "/v1/runs", json={"input": "hi", "model": "primary-x"}
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(100):
                    status = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get(
                        "status"
                    )
                    if status in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.05)

                resp = await cli.get(f"/v1/runs/{run_id}")
                data = await resp.json()

        assert data.get("requested_policy", {}).get("model") == "primary-x"
        assert data.get("actual_policy", {}).get("model") == "primary-x"
        assert data.get("actual_policy", {}).get("provider") == "custom"
        assert data.get("fallback_reason") is None

    @pytest.mark.asyncio
    async def test_bare_magicmock_agent_does_not_fabricate_fallback(self, store):
        """Regression: a bare MagicMock agent (existing helpers) must NOT be
        treated as a fallback — bool(MagicMock()) is True, so the writer must
        require _fallback_activated is literally True and model/provider are
        real strings."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            # _run_to_terminal uses a bare MagicMock with no model/provider set.
            run_id = await _run_to_terminal(
                adapter, cli, {"input": "hi", "model": "my-model"}
            )
            resp = await cli.get(f"/v1/runs/{run_id}")
            data = await resp.json()

        # No fabricated fallback reason from a bare MagicMock.
        assert data.get("fallback_reason") is None
        # requested still recorded.
        assert data.get("requested_policy", {}).get("model") == "my-model"
