"""Managed Hermes Session contracts for the durable ``/v1/runs`` surface.

These tests exercise the real SessionDB and DurableRunStore against temporary
databases.  The platform supplies only a stable Hermes ``session_id`` and the
new user input; Hermes remains the sole owner of transcript recovery.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


def _make_adapter(
    *,
    session_db: SessionDB,
    durable_store: DurableRunStore,
) -> APIServerAdapter:
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True),
        durable_store=durable_store,
    )
    adapter._session_db = session_db
    return adapter


def _runs_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    return app


def _completed_agent(captured: dict[str, object]) -> MagicMock:
    agent = MagicMock()

    def _run_conversation(*, user_message, conversation_history, task_id):
        captured.update(
            {
                "user_message": user_message,
                "conversation_history": conversation_history,
                "task_id": task_id,
            }
        )
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _run_conversation
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


def _platform_metadata() -> dict[str, str]:
    return {
        "command_id": "command-7",
        "kind": "conversation.turn",
        "client_request_id": "request-9",
        "platform_session_id": "platform-session-5",
        "canonical_request_digest": "a" * 64,
        "payload_ref": "platform-payload://sha256/" + ("b" * 64),
        "source": "platform.hqa_hermes_run_port",
    }


@pytest.mark.parametrize(
    "payload_ref",
    [
        "payload:sha256:" + ("b" * 64),
        "hqa-payload:sha256:" + ("b" * 64),
        "platform-payload://sha256/" + ("b" * 64),
    ],
)
def test_platform_run_context_accepts_all_canonical_payload_ref_surfaces(
    payload_ref,
):
    metadata = _platform_metadata()
    metadata["payload_ref"] = payload_ref

    context, error = APIServerAdapter._parse_platform_run_context(
        {"metadata": metadata},
        managed_session=True,
    )

    assert error is None
    assert context == {
        "command_id": "command-7",
        "platform_session_id": "platform-session-5",
    }


async def _wait_for_run(adapter: APIServerAdapter, run_id: str) -> None:
    task = adapter._active_run_tasks.get(run_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=2)


@pytest.fixture
def managed_state(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    durable_store = DurableRunStore(tmp_path / "durable-runs.db")
    try:
        yield session_db, durable_store
    finally:
        durable_store.close()
        session_db.close()


@pytest.mark.asyncio
async def test_existing_session_recovers_history_without_client_history(managed_state):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    session_db.append_message("managed-1", "user", "CONTEXT_ALPHA=7319")
    session_db.append_message("managed-1", "assistant", "I will remember it.")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    captured: dict[str, object] = {}

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_completed_agent(captured),
    ):
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "What was the number?", "session_id": "managed-1"},
                headers={"Idempotency-Key": "turn-2"},
            )
            body = await response.json()
            await _wait_for_run(adapter, body["run_id"])

    assert response.status == 202
    assert body["session_id"] == "managed-1"
    assert body["created"] is True
    assert captured["user_message"] == "What was the number?"
    assert captured["task_id"] == "managed-1"
    assert [
        {"role": item["role"], "content": item["content"]}
        for item in captured["conversation_history"]
    ] == [
            {"role": "user", "content": "CONTEXT_ALPHA=7319"},
            {"role": "assistant", "content": "I will remember it."},
        ]


@pytest.mark.asyncio
async def test_platform_run_context_reaches_only_current_managed_run_environment(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("web_managed_1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    captured: dict[str, object] = {}
    agent = MagicMock()

    def _run_conversation(*, user_message, conversation_history, task_id):
        from gateway.session_context import get_session_env
        from tools.environments.local import _inject_session_context_env

        child_env: dict[str, str] = {}
        _inject_session_context_env(child_env)
        captured.update(
            {
                "user_message": user_message,
                "task_id": task_id,
                "command_id": get_session_env("HERMES_PLATFORM_COMMAND_ID"),
                "platform_session_id": get_session_env(
                    "HERMES_PLATFORM_SESSION_ID"
                ),
                "platform_run_id": get_session_env("HERMES_PLATFORM_RUN_ID"),
                "managed_session_id": get_session_env(
                    "HERMES_PLATFORM_MANAGED_SESSION_ID"
                ),
                "child_env": child_env,
            }
        )
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _run_conversation
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    with patch.object(adapter, "_create_agent", return_value=agent) as create_agent:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={
                    "input": "Keep the transcript natural.",
                    "session_id": "web_managed_1",
                    "metadata": _platform_metadata(),
                },
                headers={"Idempotency-Key": "platform-turn-1"},
            )
            body = await response.json()
            await _wait_for_run(adapter, body["run_id"])

    assert response.status == 202
    assert captured["user_message"] == "Keep the transcript natural."
    assert captured["task_id"] == "web_managed_1"
    assert captured["command_id"] == "command-7"
    assert captured["platform_session_id"] == "platform-session-5"
    assert captured["platform_run_id"] == body["run_id"]
    assert captured["managed_session_id"] == "web_managed_1"
    assert captured["child_env"] == {
        "HERMES_SESSION_PLATFORM": "api_server",
        "HERMES_SESSION_SOURCE": "",
        "HERMES_SESSION_CHAT_ID": "",
        "HERMES_SESSION_CHAT_NAME": "",
        "HERMES_SESSION_THREAD_ID": "",
        "HERMES_SESSION_USER_ID": "",
        "HERMES_SESSION_USER_NAME": "",
        "HERMES_SESSION_KEY": body["run_id"],
        "HERMES_SESSION_ID": "web_managed_1",
        "HERMES_UI_SESSION_ID": "",
        "HERMES_SESSION_MESSAGE_ID": "",
        "HERMES_SESSION_PROFILE": "",
        "HERMES_PLATFORM_COMMAND_ID": "command-7",
        "HERMES_PLATFORM_SESSION_ID": "platform-session-5",
        "HERMES_PLATFORM_RUN_ID": body["run_id"],
        "HERMES_PLATFORM_MANAGED_SESSION_ID": "web_managed_1",
    }
    assert create_agent.call_args.kwargs["ephemeral_system_prompt"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata.pop("command_id"),
        lambda metadata: metadata.__setitem__("extra", "not-allowed"),
        lambda metadata: metadata.__setitem__(
            "canonical_request_digest", "not-a-digest"
        ),
        lambda metadata: metadata.__setitem__("payload_ref", "payload:unbound"),
    ],
)
async def test_claimed_platform_run_context_is_strict_and_allocates_nothing(
    managed_state,
    mutate,
):
    session_db, durable_store = managed_state
    session_db.create_session("web_managed_1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    metadata = _platform_metadata()
    mutate(metadata)

    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={
                    "input": "hello",
                    "session_id": "web_managed_1",
                    "metadata": metadata,
                },
                headers={"Idempotency-Key": "invalid-platform-context"},
            )
            body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "invalid_platform_run_context"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_platform_run_context_requires_managed_session(managed_state):
    session_db, durable_store = managed_state
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )

    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "metadata": _platform_metadata()},
                headers={"Idempotency-Key": "unmanaged-platform-context"},
            )
            body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "platform_context_requires_managed_session"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_managed_session_id_must_exist_before_run_allocation(managed_state):
    session_db, durable_store = managed_state
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )

    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "missing"},
                headers={"Idempotency-Key": "missing-session"},
            )
            body = await response.json()

    assert response.status == 404
    assert body["error"]["code"] == "session_not_found"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_managed_session_history_failure_is_fail_closed(
    managed_state,
    monkeypatch,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    monkeypatch.setattr(
        session_db,
        "get_messages_as_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("state database unavailable")
        ),
    )

    with patch.object(adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "managed-1"},
                headers={"Idempotency-Key": "history-failure"},
            )
            body = await response.json()

    assert response.status == 503
    assert body["error"]["code"] == "session_history_unavailable"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_ended_managed_session_requires_fork(managed_state):
    session_db, durable_store = managed_state
    session_db.create_session("managed-ended", "api_server")
    session_db.end_session("managed-ended", "user_ended")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )

    async with TestClient(TestServer(_runs_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "continue", "session_id": "managed-ended"},
            headers={"Idempotency-Key": "ended-session"},
        )
        body = await response.json()

    assert response.status == 409
    assert body["error"]["code"] == "session_ended"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_managed_session_resolves_compression_tip_and_ancestor_history(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-root", "api_server")
    session_db.append_message("managed-root", "user", "before compression")
    session_db.end_session("managed-root", "compression")
    session_db.create_session(
        "managed-tip",
        "api_server",
        parent_session_id="managed-root",
    )
    session_db.append_message("managed-tip", "assistant", "after compression")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    captured: dict[str, object] = {}

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_completed_agent(captured),
    ) as create_agent:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "continue", "session_id": "managed-root"},
                headers={"Idempotency-Key": "after-compression"},
            )
            body = await response.json()
            await _wait_for_run(adapter, body["run_id"])

    assert response.status == 202
    assert body["session_id"] == "managed-tip"
    assert durable_store.get_run(body["run_id"])["session_id"] == "managed-tip"
    assert create_agent.call_args.kwargs["session_id"] == "managed-tip"
    assert [
        {"role": item["role"], "content": item["content"]}
        for item in captured["conversation_history"]
    ] == [
        {"role": "user", "content": "before compression"},
        {"role": "assistant", "content": "after compression"},
    ]


@pytest.mark.asyncio
async def test_managed_fork_keeps_source_and_child_run_history_independent(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-source", "api_server")
    session_db.append_message("managed-source", "user", "shared question")
    session_db.append_message("managed-source", "assistant", "shared answer")
    fork_point = session_db.get_messages("managed-source")[-1]["id"]
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    source_capture: dict[str, object] = {}
    child_capture: dict[str, object] = {}

    with patch.object(
        adapter,
        "_create_agent",
        side_effect=[
            _completed_agent(source_capture),
            _completed_agent(child_capture),
        ],
    ):
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            fork_response = await client.post(
                "/api/sessions/managed-source/fork",
                json={
                    "id": "managed-child",
                    "preserve_source": True,
                    "fork_point": f"message:{fork_point}",
                },
            )
            assert fork_response.status == 201

            session_db.append_message("managed-source", "user", "source-only context")
            session_db.append_message("managed-child", "user", "child-only context")

            source_response = await client.post(
                "/v1/runs",
                json={"input": "continue source", "session_id": "managed-source"},
                headers={"Idempotency-Key": "source-turn"},
            )
            source_body = await source_response.json()
            await _wait_for_run(adapter, source_body["run_id"])

            child_response = await client.post(
                "/v1/runs",
                json={"input": "continue child", "session_id": "managed-child"},
                headers={"Idempotency-Key": "child-turn"},
            )
            child_body = await child_response.json()
            await _wait_for_run(adapter, child_body["run_id"])
            child_replay = await client.post(
                "/v1/runs",
                json={"input": "continue child", "session_id": "managed-child"},
                headers={"Idempotency-Key": "child-turn"},
            )
            child_replay_body = await child_replay.json()

    assert source_response.status == child_response.status == child_replay.status == 202
    assert source_body["session_id"] == "managed-source"
    assert child_body["session_id"] == "managed-child"
    assert child_replay_body["created"] is False
    assert child_replay_body["run_id"] == child_body["run_id"]
    assert child_replay_body["session_id"] == "managed-child"
    assert [
        {"role": item["role"], "content": item["content"]}
        for item in source_capture["conversation_history"]
    ] == [
        {"role": "user", "content": "shared question"},
        {"role": "assistant", "content": "shared answer"},
        {"role": "user", "content": "source-only context"},
    ]
    assert [
        {"role": item["role"], "content": item["content"]}
        for item in child_capture["conversation_history"]
    ] == [
        {"role": "user", "content": "shared question"},
        {"role": "assistant", "content": "shared answer"},
        {"role": "user", "content": "child-only context"},
    ]


@pytest.mark.asyncio
async def test_managed_fork_compression_history_stops_at_branch_boundary(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-source", "api_server")
    session_db.append_message("managed-source", "user", "shared question")
    session_db.append_message("managed-source", "assistant", "shared answer")
    fork_point = session_db.get_messages("managed-source")[-1]["id"]
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    captured: dict[str, object] = {}

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_completed_agent(captured),
    ):
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            fork_response = await client.post(
                "/api/sessions/managed-source/fork",
                json={
                    "id": "managed-child",
                    "preserve_source": True,
                    "fork_point": f"message:{fork_point}",
                },
            )
            assert fork_response.status == 201

            session_db.end_session("managed-child", "compression")
            session_db.create_session(
                "managed-child-tip",
                "api_server",
                parent_session_id="managed-child",
            )
            session_db.append_message(
                "managed-child-tip",
                "user",
                "child context after compression",
            )

            response = await client.post(
                "/v1/runs",
                json={"input": "continue child", "session_id": "managed-child"},
                headers={"Idempotency-Key": "compressed-child-turn"},
            )
            body = await response.json()
            await _wait_for_run(adapter, body["run_id"])

    assert response.status == 202
    assert body["session_id"] == "managed-child-tip"
    assert [
        {"role": item["role"], "content": item["content"]}
        for item in captured["conversation_history"]
    ] == [
        {"role": "user", "content": "shared question"},
        {"role": "assistant", "content": "shared answer"},
        {"role": "user", "content": "child context after compression"},
    ]


@pytest.mark.asyncio
async def test_managed_session_rejects_client_supplied_history(managed_state):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )

    async with TestClient(TestServer(_runs_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={
                "input": "hello",
                "session_id": "managed-1",
                "conversation_history": [
                    {"role": "assistant", "content": "counterfeit history"}
                ],
            },
        )
        body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "managed_session_history_owned_by_hermes"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_durable_ack_replay_returns_same_run_session_and_created_false(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )

    with patch.object(adapter, "_create_agent") as create_agent:
        create_agent.return_value = _completed_agent({})
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            headers = {"Idempotency-Key": "ack-loss-turn"}
            first = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "managed-1"},
                headers=headers,
            )
            first_body = await first.json()
            replay = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "managed-1"},
                headers=headers,
            )
            replay_body = await replay.json()
            await _wait_for_run(adapter, first_body["run_id"])

    assert first.status == replay.status == 202
    assert first_body["created"] is True
    assert replay_body["created"] is False
    assert replay_body["run_id"] == first_body["run_id"]
    assert replay_body["session_id"] == "managed-1"
    assert create_agent.call_count == 1


@pytest.mark.asyncio
async def test_inflight_ack_replay_bypasses_new_run_concurrency_limit(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    adapter._max_concurrent_runs = 1
    entered = threading.Event()
    release = threading.Event()
    agent = MagicMock()

    def _block(**kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _block
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            headers = {"Idempotency-Key": "ack-replay-at-capacity"}
            request = {"input": "hello", "session_id": "managed-1"}
            first = await client.post(
                "/v1/runs",
                json=request,
                headers=headers,
            )
            first_body = await first.json()
            assert await asyncio.to_thread(entered.wait, 1)

            replay = await client.post(
                "/v1/runs",
                json=request,
                headers=headers,
            )
            replay_body = await replay.json()
            release.set()
            await _wait_for_run(adapter, first_body["run_id"])

    assert first.status == replay.status == 202
    assert replay_body["run_id"] == first_body["run_id"]
    assert replay_body["created"] is False


@pytest.mark.asyncio
async def test_completed_ack_replay_bypasses_other_session_concurrency(
    managed_state,
):
    session_db, durable_store = managed_state
    session_db.create_session("managed-a", "api_server")
    session_db.create_session("managed-b", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    adapter._max_concurrent_runs = 1
    entered = threading.Event()
    release = threading.Event()
    slow_agent = MagicMock()

    def _block(**kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"final_response": "done"}

    slow_agent.run_conversation.side_effect = _block
    slow_agent.session_prompt_tokens = 0
    slow_agent.session_completion_tokens = 0
    slow_agent.session_total_tokens = 0

    with patch.object(adapter, "_create_agent") as create_agent:
        create_agent.side_effect = [_completed_agent({}), slow_agent]
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            first_headers = {"Idempotency-Key": "completed-ack"}
            first_request = {"input": "first", "session_id": "managed-a"}
            first = await client.post(
                "/v1/runs",
                json=first_request,
                headers=first_headers,
            )
            first_body = await first.json()
            await _wait_for_run(adapter, first_body["run_id"])

            second = await client.post(
                "/v1/runs",
                json={"input": "busy", "session_id": "managed-b"},
                headers={"Idempotency-Key": "busy-other-session"},
            )
            second_body = await second.json()
            assert await asyncio.to_thread(entered.wait, 1)

            replay = await client.post(
                "/v1/runs",
                json=first_request,
                headers=first_headers,
            )
            replay_body = await replay.json()
            release.set()
            await _wait_for_run(adapter, second_body["run_id"])

    assert first.status == second.status == replay.status == 202
    assert replay_body["run_id"] == first_body["run_id"]
    assert replay_body["created"] is False
    assert create_agent.call_count == 2


@pytest.mark.asyncio
async def test_same_client_id_is_scoped_by_managed_session(managed_state):
    session_db, durable_store = managed_state
    session_db.create_session("managed-a", "api_server")
    session_db.create_session("managed-b", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )

    with patch.object(adapter, "_create_agent") as create_agent:
        create_agent.return_value = _completed_agent({})
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            headers = {"Idempotency-Key": "client-action-42"}
            first = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "managed-a"},
                headers=headers,
            )
            second = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "managed-b"},
                headers=headers,
            )
            first_body = await first.json()
            second_body = await second.json()
            await _wait_for_run(adapter, first_body["run_id"])
            await _wait_for_run(adapter, second_body["run_id"])

    assert first.status == second.status == 202
    assert first_body["run_id"] != second_body["run_id"]
    assert first_body["session_id"] == "managed-a"
    assert second_body["session_id"] == "managed-b"
    assert create_agent.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_distinct_turn_on_same_session_is_busy(managed_state):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    entered = threading.Event()
    release = threading.Event()
    agent = MagicMock()

    def _block(**kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _block
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            first = await client.post(
                "/v1/runs",
                json={"input": "first", "session_id": "managed-1"},
                headers={"Idempotency-Key": "first"},
            )
            first_body = await first.json()
            assert await asyncio.to_thread(entered.wait, 1)

            second = await client.post(
                "/v1/runs",
                json={"input": "second", "session_id": "managed-1"},
                headers={"Idempotency-Key": "second"},
            )
            second_body = await second.json()
            release.set()
            await _wait_for_run(adapter, first_body["run_id"])

    assert first.status == 202
    assert second.status == 409
    assert second_body["error"]["code"] == "session_busy"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_inflight_same_client_id_with_different_body_is_conflict(managed_state):
    session_db, durable_store = managed_state
    session_db.create_session("managed-1", "api_server")
    adapter = _make_adapter(
        session_db=session_db,
        durable_store=durable_store,
    )
    entered = threading.Event()
    release = threading.Event()
    agent = MagicMock()

    def _block(**kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _block
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            headers = {"Idempotency-Key": "same-client-action"}
            first = await client.post(
                "/v1/runs",
                json={"input": "first", "session_id": "managed-1"},
                headers=headers,
            )
            first_body = await first.json()
            assert await asyncio.to_thread(entered.wait, 1)

            conflict = await client.post(
                "/v1/runs",
                json={"input": "different", "session_id": "managed-1"},
                headers=headers,
            )
            conflict_body = await conflict.json()
            release.set()
            await _wait_for_run(adapter, first_body["run_id"])

    assert conflict.status == 409
    assert conflict_body["error"]["code"] == "idempotency_conflict"
    assert durable_store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_restart_replay_recovers_session_identity_without_agent(
    tmp_path,
):
    state_path = tmp_path / "state.db"
    durable_path = tmp_path / "durable-runs.db"
    session_db = SessionDB(state_path)
    session_db.create_session("managed-restart", "api_server")
    first_store = DurableRunStore(durable_path)
    first_adapter = _make_adapter(
        session_db=session_db,
        durable_store=first_store,
    )
    headers = {"Idempotency-Key": "restart-turn"}
    request = {"input": "hello", "session_id": "managed-restart"}

    with patch.object(first_adapter, "_create_agent") as first_create:
        first_create.return_value = _completed_agent({})
        async with TestClient(TestServer(_runs_app(first_adapter))) as client:
            first = await client.post("/v1/runs", json=request, headers=headers)
            first_body = await first.json()
            await _wait_for_run(first_adapter, first_body["run_id"])
    first_store.close()

    reopened_store = DurableRunStore(durable_path)
    restarted = _make_adapter(
        session_db=session_db,
        durable_store=reopened_store,
    )
    try:
        with patch.object(restarted, "_create_agent") as restarted_create:
            async with TestClient(TestServer(_runs_app(restarted))) as client:
                replay = await client.post("/v1/runs", json=request, headers=headers)
                replay_body = await replay.json()

        assert replay.status == 202
        assert replay_body["run_id"] == first_body["run_id"]
        assert replay_body["session_id"] == "managed-restart"
        assert replay_body["created"] is False
        restarted_create.assert_not_called()
    finally:
        reopened_store.close()
        session_db.close()
