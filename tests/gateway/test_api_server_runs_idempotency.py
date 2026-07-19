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
* No ``Idempotency-Key`` ⇒ fresh-run identity behavior is unchanged.

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
    async def test_initial_status_failure_cleans_live_state_and_returns_503(
        self, store
    ):
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        with patch.object(
            adapter,
            "_set_run_status",
            side_effect=OSError("durable status unavailable"),
        ), patch.object(adapter, "_create_agent") as mock_create:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                payload = await response.json()

        assert response.status == 503
        assert payload["error"]["code"] == "durable_unavailable"
        row = dict(store._conn.execute("SELECT * FROM runs").fetchone())
        run_id = row["run_id"]
        assert row["status"] == "stopped"
        for mapping in (
            adapter._run_streams,
            adapter._run_streams_created,
            adapter._run_approval_sessions,
            adapter._run_statuses,
            adapter._run_event_seq,
            adapter._run_event_subscribers,
            adapter._active_run_agents,
            adapter._active_run_tasks,
        ):
            assert run_id not in mapping
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_previous_response_session_is_persisted_as_run_identity(self, store):
        adapter = _make_adapter(durable_store=store)
        adapter._response_store.put(
            "resp_previous",
            {
                "session_id": "session_original",
                "conversation_history": [{"role": "user", "content": "before"}],
            },
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _completed_agent_mock()
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "continue", "previous_response_id": "resp_previous"},
                    headers={"Idempotency-Key": "idem-previous"},
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]

        assert store.get_run(run_id)["session_id"] == "session_original"

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
                assert d2["status"] in {
                    "queued",
                    "running",
                    "succeeded",
                    "failed",
                    "stopped",
                }
                assert d2["status"] not in {
                    "started",
                    "recovered",
                    "completed",
                    "cancelled",
                    "waiting_for_approval",
                    "stopping",
                }

    @pytest.mark.asyncio
    async def test_duplicate_with_unknown_stored_state_fails_closed(self, store):
        body = {"input": "hello"}
        submit = store.submit_or_get(
            idempotency_key="idem-invalid-state", request_body=body
        )
        store._conn.execute(
            "UPDATE runs SET status = ? WHERE run_id = ?",
            ("future_unreviewed_state", submit.run_id),
        )
        store._conn.commit()
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": "idem-invalid-state"},
                )
                response_body = await response.json()

        assert response.status == 503
        assert response_body["error"]["code"] == "run_state_invalid"
        assert "future_unreviewed_state" not in str(response_body)
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_lookup_failure_returns_durable_503(
        self, store, monkeypatch
    ):
        body = {"input": "hello"}
        store.submit_or_get(idempotency_key="idem-read-fails", request_body=body)
        monkeypatch.setattr(
            store,
            "get_run",
            lambda run_id: (_ for _ in ()).throw(OSError("database unavailable")),
        )
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": "idem-read-fails"},
                )
                response_body = await response.json()

        assert response.status == 503
        assert response_body["error"]["code"] == "durable_unavailable"
        mock_create.assert_not_called()

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
# fresh-run identity behavior preserved when no Idempotency-Key is present
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
                assert data["status"] == "queued"
                assert data["run_id"].startswith("run_")
                assert mock_create.call_count == 1


class TestDurableAdmissionFailsClosed:
    @pytest.mark.asyncio
    async def test_broker_seed_failure_stops_unadmitted_run(self, store, monkeypatch):
        adapter = _make_adapter(durable_store=store)
        monkeypatch.setattr(
            adapter,
            "_broker_seed",
            lambda run_id: (_ for _ in ()).throw(RuntimeError("broker seed failed")),
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post(
                    "/v1/runs", json={"input": "must seed broker"}
                )
                response_body = await response.json()

        assert response.status == 503
        assert response_body["error"]["code"] == "durable_unavailable"
        mock_create.assert_not_called()
        assert not adapter._active_run_tasks
        assert not adapter._run_streams
        rows = store._conn.execute("SELECT status FROM runs").fetchall()
        assert [row["status"] for row in rows] == ["stopped"]

    @pytest.mark.parametrize("failure_mode", ["false", "exception"])
    @pytest.mark.asyncio
    async def test_requested_policy_failure_rejects_before_agent_creation(
        self, store, monkeypatch, failure_mode
    ):
        adapter = _make_adapter(durable_store=store)

        def _fail_requested_policy(run_id, policy):
            if failure_mode == "exception":
                raise OSError("requested-policy evidence unavailable")
            return False

        monkeypatch.setattr(store, "set_requested_policy", _fail_requested_policy)
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post(
                    "/v1/runs", json={"input": "must preserve requested policy"}
                )
                response_body = await response.json()

        assert response.status == 503
        assert response_body["error"]["code"] == "durable_unavailable"
        mock_create.assert_not_called()
        assert not adapter._active_run_tasks
        assert not adapter._run_streams
        assert not adapter._run_statuses
        assert not adapter._run_event_seq
        rows = store._conn.execute("SELECT status FROM runs").fetchall()
        assert [row["status"] for row in rows] == ["stopped"]

    @pytest.mark.asyncio
    async def test_register_failure_rejects_before_agent_creation(self, store, monkeypatch):
        adapter = _make_adapter(durable_store=store)
        monkeypatch.setattr(
            store,
            "register_run",
            lambda **kwargs: (_ for _ in ()).throw(OSError("durable DB unavailable")),
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post("/v1/runs", json={"input": "must persist"})

        assert response.status == 503
        assert mock_create.call_count == 0

    @pytest.mark.asyncio
    async def test_submit_failure_rejects_before_agent_creation(self, store, monkeypatch):
        adapter = _make_adapter(durable_store=store)
        monkeypatch.setattr(
            store,
            "submit_or_get",
            lambda **kwargs: (_ for _ in ()).throw(OSError("durable DB unavailable")),
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "must persist"},
                    headers={"Idempotency-Key": "fail-closed"},
                )

        assert response.status == 503
        assert mock_create.call_count == 0


class TestDurableReadFailsClosed:
    @pytest.mark.parametrize("live_status_present", [False, True])
    @pytest.mark.asyncio
    async def test_get_run_lookup_failure_returns_durable_503(
        self, store, monkeypatch, live_status_present
    ):
        run_id = "run_lookup_failure"
        adapter = _make_adapter(durable_store=store)
        if live_status_present:
            adapter._run_statuses[run_id] = {
                "run_id": run_id,
                "status": "running",
            }
        monkeypatch.setattr(
            store,
            "get_run",
            lambda requested_id: (_ for _ in ()).throw(
                OSError("database unavailable")
            ),
        )
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(f"/v1/runs/{run_id}")
            response_body = await response.json()

        assert response.status == 503
        assert response_body["error"]["code"] == "durable_unavailable"
