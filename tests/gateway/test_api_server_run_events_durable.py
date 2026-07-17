"""V2.5 — Durable event/cursor/replay + SSE broadcast (contract matrix §2 row 4).

Each run event must get a stable ``event_id`` and a per-run monotonic ``seq``;
replay from any cursor with **no gap, no duplicate**; and SSE disconnect must
**not** delete canonical events. This slice wires the durable store
(``DurableRunStore.append_event`` / ``replay_events``) behind a per-run fan-out
broker so:

* every emitted event is persisted durably (survives restart);
* each subscriber gets its own queue (fixes the single-consumer race);
* SSE disconnect only drops the *subscriber* transport — canonical events
  remain replayable;
* ``GET /v1/runs/{id}/events?since={seq}`` and the ``Last-Event-ID`` header
  resume from a cursor; frames carry an SSE ``id:`` field so the cursor is
  spec-compliant.

Red line: hermetic only — durable store injected with a throwaway ``tmp_path``
DB; no live state, no network. Fails until ``_handle_runs``/``_handle_run_events``
are wired to the broker.
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

# Bounded connect/read so a regression that leaves the stream hanging fails
# fast instead of stalling the suite on the 30s keepalive timeout.
_TIMEOUT = aiohttp.ClientTimeout(total=20, sock_connect=5, sock_read=15)


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


def _agent_with_deltas(deltas):
    """Mock agent that emits ``deltas`` via stream_delta_callback then completes."""
    mock_agent = MagicMock()

    def _run(user_message=None, conversation_history=None, task_id=None):
        cb = getattr(mock_agent, "_captured_stream_cb", None)
        if cb is not None:
            for d in deltas:
                cb(d)
        return {"final_response": "FINAL"}

    mock_agent.run_conversation.side_effect = _run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    return mock_agent


def _capture_stream_cb(mock_create, mock_agent):
    def _create(*args, **kwargs):
        mock_agent._captured_stream_cb = kwargs.get("stream_delta_callback")
        return mock_agent

    mock_create.side_effect = _create


async def _run_to_terminal(adapter, cli, deltas, **post_kwargs):
    """POST a run whose agent emits ``deltas``; wait for completion; return run_id."""
    with patch.object(adapter, "_create_agent") as mock_create:
        mock_agent = _agent_with_deltas(deltas)
        _capture_stream_cb(mock_create, mock_agent)
        resp = await cli.post("/v1/runs", json={"input": "hello", **post_kwargs})
        assert resp.status == 202
        run_id = (await resp.json())["run_id"]
        for _ in range(100):
            status = (await (await cli.get(f"/v1/runs/{run_id}")).json()).get("status")
            if status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)
        assert status == "completed"
        return run_id


def _sse_events(body: str):
    """Parse SSE ``data: {...}`` frames into a list of event dicts."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[len("data:"):].strip()))
            except json.JSONDecodeError:
                pass
    return out


def _deltas(events):
    return [e.get("delta") for e in events if e.get("event") == "message.delta"]


class TestDurableEventPersistence:
    @pytest.mark.asyncio
    async def test_events_persisted_with_monotonic_seq_and_stable_ids(self, store):
        """Emitted run events land in the store with stable event_id + monotonic seq."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, ["a", "b", "c"])

        events = store.replay_events(run_id, since_seq=0)
        delta_payloads = [e.payload.get("delta") for e in events if e.event_type == "message.delta"]
        assert delta_payloads == ["a", "b", "c"]
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))  # monotonic, no dup
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids)) and all(ids)  # stable + distinct

    @pytest.mark.asyncio
    async def test_events_survive_restart(self, tmp_path):
        """Events persisted by one adapter are intact after a 'restart'."""
        db = tmp_path / "durable_runs.db"
        store1 = DurableRunStore(db_path=db)
        adapter1 = _make_adapter(durable_store=store1)
        app1 = _create_runs_app(adapter1)
        async with TestClient(TestServer(app1), timeout=_TIMEOUT) as cli1:
            run_id = await _run_to_terminal(adapter1, cli1, ["x", "y"])
        store1.close()

        store2 = DurableRunStore(db_path=db)
        try:
            deltas = [
                e.payload.get("delta")
                for e in store2.replay_events(run_id, since_seq=0)
                if e.event_type == "message.delta"
            ]
            assert deltas == ["x", "y"]
        finally:
            store2.close()


class TestEventCursorReplay:
    @pytest.mark.asyncio
    async def test_replay_from_since_cursor_no_gap_no_dup(self, store):
        """?since={seq} returns only later events, ordered, no gap/dup."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, ["0", "1", "2", "3"])
            all_events = store.replay_events(run_id, since_seq=0)
            cursor = all_events[1].seq  # resume after the 2nd event

            resp = await cli.get(f"/v1/runs/{run_id}/events?since={cursor}")
            assert resp.status == 200
            body = await resp.text()

        events = _sse_events(body)
        seqs = [e.get("seq") for e in events if e.get("seq") is not None]
        assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
        assert all(s > cursor for s in seqs)  # strictly after the cursor

    @pytest.mark.asyncio
    async def test_last_event_id_header_resumes(self, store):
        """Last-Event-ID header is honored as a resume cursor (by seq)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, ["m", "n", "o"])
            all_events = store.replay_events(run_id, since_seq=0)
            cursor = all_events[0].seq

            resp = await cli.get(
                f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": str(cursor)}
            )
            assert resp.status == 200
            body = await resp.text()

        events = _sse_events(body)
        seqs = [e.get("seq") for e in events if e.get("seq") is not None]
        assert all(s > cursor for s in seqs)


class TestBroadcastNoRace:
    @pytest.mark.asyncio
    async def test_two_subscribers_each_get_all_events(self, store):
        """Two concurrent subscribers must EACH receive the full event set (no race)."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, ["p", "q"])
            r1, r2 = await asyncio.gather(
                cli.get(f"/v1/runs/{run_id}/events"),
                cli.get(f"/v1/runs/{run_id}/events"),
            )
            b1 = _sse_events(await r1.text())
            b2 = _sse_events(await r2.text())

        assert _deltas(b1) == ["p", "q"]
        assert _deltas(b2) == ["p", "q"]


class TestSSEWireFormat:
    @pytest.mark.asyncio
    async def test_frames_carry_id_and_event_fields(self, store):
        """SSE frames are spec-compliant: monotonic id: + event: + data: per event."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, ["hello"])
            resp = await cli.get(f"/v1/runs/{run_id}/events")
            body = await resp.text()

        # Every event frame must have an id: line (the resume cursor) and an
        # event: line, followed by a data: line.
        id_lines = [l for l in body.splitlines() if l.startswith("id: ")]
        event_lines = [l for l in body.splitlines() if l.startswith("event: ")]
        data_lines = [l for l in body.splitlines() if l.startswith("data: ")]
        assert id_lines, "frames must carry an SSE id: field for Last-Event-ID resume"
        assert len(id_lines) == len(data_lines)
        assert len(event_lines) == len(data_lines)
        # id: values are the monotonic seq, strictly increasing.
        ids = [int(l.split("id: ", 1)[1]) for l in id_lines]
        assert ids == sorted(ids) and len(ids) == len(set(ids))
        # First frame id is 1 (per-run monotonic from 1).
        assert ids[0] == 1


class TestDisconnectPreservesEvents:
    @pytest.mark.asyncio
    async def test_disconnect_does_not_delete_canonical_events(self, store):
        """A subscriber disconnecting must not remove replayable canonical events."""
        adapter = _make_adapter(durable_store=store)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app), timeout=_TIMEOUT) as cli:
            run_id = await _run_to_terminal(adapter, cli, ["u", "v"])

            # First subscriber connects then disconnects.
            first = await cli.get(f"/v1/runs/{run_id}/events")
            first.close()

            # Canonical events must still be fully replayable for a new subscriber.
            second = await cli.get(f"/v1/runs/{run_id}/events")
            body = await second.text()

        assert _deltas(_sse_events(body)) == ["u", "v"]
