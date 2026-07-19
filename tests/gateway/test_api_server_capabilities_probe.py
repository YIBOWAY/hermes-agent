"""V2.9 — capability behavioral probe (contract matrix §2 row 8).

``GET /v1/capabilities`` today advertises the run features (``run_submission``,
``run_events_sse``, ``run_stop``, ``run_approval_response`` …) as hardcoded
``True`` literals with **no reference to runtime evidence** — the platform
cannot distinguish "upstream actually grounds durable idempotency / event
replay / approval CAS" from "upstream merely claims it". This slice makes the
advertisement *honest*:

* **Purely additive** — the legacy (no durable store) payload is
  byte-identical; existing contract tests stay green.
* When a durable store IS configured, the payload gains a ``contract_version``
  (schema negotiation, additive-only per the descriptor idiom) and a
  ``durable`` block.
* Each durable capability is reported as ``{supported, grounded, evidence}``
  where ``grounded`` is backed by transactional write/read/CAS/state-transition
  behavior against the store. The probe always rolls its transaction back, so
  successful and failed probes leave no canonical rows behind.
* A store whose schema/probe fails degrades a capability to ``grounded=False``
  (fail-closed) rather than claiming support.

Broker-aware; legacy path unchanged. Red line: hermetic only — durable store
injected with a throwaway ``tmp_path`` DB; no live state, no network. Fails
until the durable block is present and grounded by probes.
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

# The durable capabilities the probe must honestly report, each mapped to the
# store method(s) that ground it.
_DURABLE_CAPS = (
    "idempotency",
    "event_replay",
    "approval_cas",
    "idempotent_stop",
    "restart_reconcile",
    "run_evidence",
)


def _make_adapter(durable_store=None) -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={})
    return APIServerAdapter(config, durable_store=durable_store)


def _create_caps_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


@pytest.fixture()
def store(tmp_path):
    s = DurableRunStore(db_path=tmp_path / "durable_runs.db")
    yield s
    s.close()


async def _get_caps(adapter):
    app = _create_caps_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        return await resp.json()


class TestLegacyUnchanged:
    @pytest.mark.asyncio
    async def test_no_store_payload_has_no_durable_block(self):
        """Legacy (no durable store) payload must stay byte-identical.

        No contract_version, no durable block — the additive keys only appear
        when a durable store is actually configured. This keeps the existing
        static contract tests valid.
        """
        adapter = _make_adapter(durable_store=None)
        data = await _get_caps(adapter)

        assert data["object"] == "hermes.api_server.capabilities"
        assert data["platform"] == "hermes-agent"
        # Static run features still advertised exactly as before.
        assert data["features"]["run_submission"] is True
        assert data["features"]["run_events_sse"] is True
        assert data["features"]["run_stop"] is True
        assert data["features"]["run_approval_response"] is True
        # Additive probe keys are ABSENT without a store.
        assert "contract_version" not in data
        assert "durable" not in data


class TestDurableProbePresent:
    @pytest.mark.asyncio
    async def test_contract_version_present_with_store(self, store):
        """A configured store surfaces a contract_version for schema negotiation."""
        adapter = _make_adapter(durable_store=store)
        data = await _get_caps(adapter)

        assert "contract_version" in data
        assert isinstance(data["contract_version"], int)
        assert data["contract_version"] >= 1

    @pytest.mark.asyncio
    async def test_durable_block_reports_each_capability(self, store):
        """Every durable capability is reported with supported+grounded+evidence."""
        adapter = _make_adapter(durable_store=store)
        data = await _get_caps(adapter)

        durable = data.get("durable")
        assert isinstance(durable, dict), "durable block missing"
        for cap in _DURABLE_CAPS:
            assert cap in durable, f"capability {cap!r} not reported"
            entry = durable[cap]
            assert entry["supported"] is True, f"{cap} should be supported with a store"
            assert entry["grounded"] is True, f"{cap} should be grounded by a live probe"
            # evidence names the grounding signal (never empty / never a bare flag).
            assert isinstance(entry["evidence"], str) and entry["evidence"]

    @pytest.mark.asyncio
    async def test_durable_block_does_not_mutate_static_features(self, store):
        """Adding the durable block must not rewrite the existing static features."""
        adapter = _make_adapter(durable_store=store)
        data = await _get_caps(adapter)

        # Static surface untouched by the additive block.
        assert data["features"]["run_submission"] is True
        assert data["features"]["chat_completions"] is True
        assert data["features"]["admin_config_rw"] is False
        assert data["endpoints"]["runs"]["path"] == "/v1/runs"


class TestProbeGrounding:
    @pytest.mark.parametrize(
        "method_name",
        [
            "submit_or_get",
            "get_run",
            "append_event",
            "replay_events",
            "issue_approval_challenge",
            "get_approval_challenge",
            "consume_approval_with_event",
            "transition",
            "list_non_terminal_runs",
            "set_requested_policy",
            "record_run_outcome",
            "finalize_run",
        ],
    )
    @pytest.mark.asyncio
    async def test_any_public_probe_seam_failure_is_ungrounded_and_rolled_back(
        self, store, monkeypatch, method_name
    ):
        tables = ("runs", "run_events", "approval_grants")
        before = {
            table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }

        def _fail(*args, **kwargs):
            raise OSError(f"{method_name} unavailable")

        monkeypatch.setattr(store, method_name, _fail)
        data = await _get_caps(_make_adapter(durable_store=store))

        assert all(
            data["durable"][cap]["grounded"] is False for cap in _DURABLE_CAPS
        )
        after = {
            table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }
        assert after == before

    @pytest.mark.asyncio
    async def test_grounded_reflects_behavioral_probe_not_mere_config(self, store):
        """grounded is backed by real store behavior, not just store-presence.

        Seeding + reading back an event through the store proves the event
        plane is genuinely durable; the probe must reflect that the store
        answers (not merely that the constructor arg was non-None).
        """
        # Seed a run + one event so replay_events has something to return.
        store.register_run(run_id="run_probe", session_id="s-probe")
        store.append_event("run_probe", "run.started", {"k": "v"})
        backlog = store.replay_events("run_probe", since_seq=0)
        assert len(backlog) == 1  # store really persists + replays

        adapter = _make_adapter(durable_store=store)
        data = await _get_caps(adapter)
        assert data["durable"]["event_replay"]["grounded"] is True

    @pytest.mark.asyncio
    async def test_broken_store_degrades_to_ungrounded_fail_closed(self):
        """A store whose probe raises must NOT be reported as grounded.

        Honesty cuts both ways: if the underlying table/read fails, the
        capability degrades to grounded=False (fail-closed) instead of
        claiming support on the strength of a configured flag.
        """

        class _BrokenStore:
            """Looks like a store (present) but cannot run a behavioral probe."""

            def probe_capabilities(self):
                raise RuntimeError("db gone")

            def get_run(self, run_id):  # noqa: ANN001 - test double
                raise RuntimeError("db gone")

            def replay_events(self, run_id, *, since_seq=0):  # noqa: ANN001
                raise RuntimeError("db gone")

            def get_approval_challenge(self, challenge_id):  # noqa: ANN001
                raise RuntimeError("db gone")

            def list_non_terminal_runs(self):
                raise RuntimeError("db gone")

        adapter = _make_adapter(durable_store=_BrokenStore())
        data = await _get_caps(adapter)

        durable = data.get("durable")
        assert isinstance(durable, dict)
        # Present-but-broken => not grounded (fail closed). supported may still
        # be reported (the code path exists) but grounded must be False.
        for cap in _DURABLE_CAPS:
            assert durable[cap]["grounded"] is False, (
                f"{cap} must not be grounded when its probe fails"
            )

    @pytest.mark.asyncio
    async def test_readable_but_unwritable_store_is_not_grounded(self):
        class _ReadableButUnwritableStore:
            def get_run(self, run_id):  # noqa: ANN001
                return None

            def replay_events(self, run_id, *, since_seq=0):  # noqa: ANN001
                return []

            def get_approval_challenge(self, challenge_id):  # noqa: ANN001
                return None

            def list_non_terminal_runs(self):
                return []

            def probe_capabilities(self):
                raise OSError("database is readable but write transaction is denied")

        data = await _get_caps(
            _make_adapter(durable_store=_ReadableButUnwritableStore())
        )

        assert all(
            data["durable"][cap]["grounded"] is False for cap in _DURABLE_CAPS
        )
