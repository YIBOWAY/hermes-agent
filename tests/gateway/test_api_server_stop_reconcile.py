"""V2.8 — idempotent stop + restart reconcile (contract matrix §2 row 7).

* **Idempotent stop**: repeating ``POST /v1/runs/{id}/stop`` is a no-op that
  returns the SAME result — even after the run is terminal. A known terminal run
  never 404s; it reports its (already-stopped/terminal) state.
* **Durable status tracking**: the store's ``runs.status`` mirrors the live
  lifecycle (queued→running→succeeded/failed/stopped), with upstream names
  mapped to contract names (completed→succeeded, cancelled→stopped).
* **Restart reconcile**: on adapter start, runs left in ``queued``/``running``
  in the store (phantom after a restart — no live task exists) reconcile to the
  deterministic terminal state ``stopped``.

Broker-only (durable store configured); legacy stop behavior is byte-identical
when no store is present.

Red line: hermetic only — durable store injected with a throwaway ``tmp_path``
DB; no live state, no network. Fails until stop is idempotent, the store status
tracks the lifecycle, and reconcile-on-start marks phantom runs stopped.
"""

import asyncio
import threading

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore, RunState
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

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


def _completed_agent():
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": "done"}
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    return mock_agent


def _slow_agent():
    ready = threading.Event()
    interrupted = threading.Event()
    mock_agent = MagicMock()
    mock_agent.interrupt = MagicMock(side_effect=lambda message=None: interrupted.set())

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    return mock_agent, ready, interrupted


async def _run_to_terminal(adapter, cli, body=None):
    with patch.object(adapter, "_create_agent") as mock_create:
        mock_create.return_value = _completed_agent()
        resp = await cli.post("/v1/runs", json=body or {"input": "hello"})
        run_id = (await resp.json())["run_id"]
        status = None
        for _ in range(100):
            status = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get("status")
            if status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)
        assert status == "completed"
        return run_id


class TestDurableStatusTracking:
    @pytest.mark.asyncio
    async def test_store_status_tracks_lifecycle_to_succeeded(self, store):
        """Store runs.status mirrors live: ends at succeeded for a completed run."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli)

        row = store.get_run(run_id)
        assert row is not None
        assert row["status"] == RunState.SUCCEEDED.value

    @pytest.mark.asyncio
    async def test_store_status_stopped_after_stop(self, store):
        """A stopped run's durable status becomes stopped (contract name)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            mock_agent, ready, interrupted = _slow_agent()
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = mock_agent
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert ready.wait(timeout=3.0)
                stop = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop.status == 200
                for _ in range(100):
                    st = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get("status")
                    if st == "cancelled":
                        break
                    await asyncio.sleep(0.05)

        row = store.get_run(run_id)
        assert row is not None
        assert row["status"] == RunState.STOPPED.value


class TestIdempotentStop:
    @pytest.mark.asyncio
    async def test_stop_terminal_run_is_idempotent_not_404(self, store):
        """Stopping an already-terminal run returns the same result, not 404."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli)
            # Run is completed; refs are cleaned up. Stop must NOT 404.
            first = await cli.post(f"/v1/runs/{run_id}/stop")
            assert first.status == 200
            first_body = await first.json()
            # Repeat stop returns the SAME result (idempotent).
            second = await cli.post(f"/v1/runs/{run_id}/stop")
            assert second.status == 200
            second_body = await second.json()
            assert first_body == second_body

    @pytest.mark.asyncio
    async def test_stop_unknown_run_still_404(self, store):
        """A run that never existed still 404s (idempotency is not a blanket 200)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            resp = await cli.post("/v1/runs/run_never_existed/stop")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_live_run_still_interrupts(self, store):
        """A live stop still interrupts the agent (idempotency doesn't break it)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            mock_agent, ready, interrupted = _slow_agent()
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = mock_agent
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert ready.wait(timeout=3.0)

                stop = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop.status == 200
                mock_agent.interrupt.assert_called_once()
                interrupted.set()


class TestRestartReconcile:
    def test_reconcile_marks_phantom_running_runs_stopped(self, store):
        """Runs left running/queued in the store reconcile to stopped on startup."""
        # Simulate a crashed process: a run recorded as running with no live task.
        store.register_run(run_id="run_phantom1", session_id="s1")
        store.transition("run_phantom1", RunState.RUNNING)
        store.register_run(run_id="run_phantom2", session_id="s2")  # stays queued

        adapter = _make_adapter(durable_store=store)
        adapter.reconcile_durable_runs()  # the startup hook

        assert store.get_run("run_phantom1")["status"] == RunState.STOPPED.value
        assert store.get_run("run_phantom2")["status"] == RunState.STOPPED.value

    def test_reconcile_leaves_terminal_runs_untouched(self, store):
        """Reconcile must not rewrite terminal runs (terminal immutability)."""
        store.register_run(run_id="run_done", session_id="s")
        store.transition("run_done", RunState.RUNNING)
        store.transition("run_done", RunState.SUCCEEDED)

        adapter = _make_adapter(durable_store=store)
        adapter.reconcile_durable_runs()

        assert store.get_run("run_done")["status"] == RunState.SUCCEEDED.value

    def test_reconcile_is_idempotent(self, store):
        """Running reconcile twice is a no-op the second time."""
        store.register_run(run_id="run_p", session_id="s")
        store.transition("run_p", RunState.RUNNING)

        adapter = _make_adapter(durable_store=store)
        adapter.reconcile_durable_runs()
        adapter.reconcile_durable_runs()  # second pass: no error, no change

        assert store.get_run("run_p")["status"] == RunState.STOPPED.value


class TestIdempotentStopPreservesTerminalFact:
    @pytest.mark.asyncio
    async def test_stop_on_succeeded_returns_succeeded_not_stopped(self, store):
        """Idempotent stop must report actual terminal status, not coerce to stopped."""
        from gateway.durable_runs import RunState

        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli)
            row = store.get_run(run_id)
            assert row["status"] == RunState.SUCCEEDED.value

            first = await cli.post(f"/v1/runs/{run_id}/stop")
            assert first.status == 200
            first_body = await first.json()
            assert first_body["status"] == "succeeded"
            assert first_body.get("idempotent_replay") is True

            second = await cli.post(f"/v1/runs/{run_id}/stop")
            assert second.status == 200
            second_body = await second.json()
            assert second_body == first_body

            # Store fact unchanged.
            assert store.get_run(run_id)["status"] == RunState.SUCCEEDED.value
