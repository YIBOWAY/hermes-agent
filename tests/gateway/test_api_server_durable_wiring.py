"""V2.13 — opt-in durable-store wiring for the api_server platform.

The reviewed V2.1–V2.9 durable-run semantics are dormant unless something
passes a ``durable_store`` into ``APIServerAdapter`` (``_broker_enabled()`` is
just ``store is not None``). ``build_durable_store(config)`` is the single,
unit-testable construction point ``gateway/run.py`` uses so the feature is
**opt-in and default-OFF**: with no enabling config the adapter gets
``durable_store=None`` and ``/v1/runs`` keeps its in-memory execution path.

Red line: default construction must be side-effect-free (no DB file, no store)
so installing this code never changes live behavior unless explicitly enabled.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from unittest.mock import MagicMock

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore
from gateway.platforms.api_server import APIServerAdapter, build_durable_store

_ENV_FLAG = "API_SERVER_DURABLE_RUNS"
_ENV_DB = "API_SERVER_DURABLE_RUNS_DB"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with a hermetic env: no durable-run env vars set."""
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    monkeypatch.delenv(_ENV_DB, raising=False)
    yield


def _close(store):
    if store is not None:
        store.close()


class TestDefaultOff:
    def test_disabled_by_default_returns_none(self):
        """No extra flag + no env -> None (legacy in-memory path preserved)."""
        store = build_durable_store(PlatformConfig(enabled=True, extra={}))
        assert store is None

    def test_disabled_constructs_no_db_file(self, tmp_path):
        """Default-off must be side-effect-free: no DB file is created."""
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={}), hermes_home=tmp_path
        )
        assert store is None
        assert not (tmp_path / "durable_runs.db").exists()

    def test_explicit_extra_false_wins_over_env(self, monkeypatch, tmp_path):
        """An explicit ``durable_runs_enabled: false`` overrides the env flag."""
        monkeypatch.setenv(_ENV_FLAG, "1")
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={"durable_runs_enabled": False}),
            hermes_home=tmp_path,
        )
        assert store is None


class TestEnabled:
    def test_enabled_via_extra_constructs_store(self, tmp_path):
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={"durable_runs_enabled": True}),
            hermes_home=tmp_path,
        )
        try:
            assert isinstance(store, DurableRunStore)
        finally:
            _close(store)

    def test_enabled_via_env_constructs_store(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_ENV_FLAG, "true")
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={}), hermes_home=tmp_path
        )
        try:
            assert isinstance(store, DurableRunStore)
        finally:
            _close(store)

    def test_default_db_path_is_under_hermes_home(self, tmp_path):
        """Enabled store lands at <hermes_home>/durable_runs.db and is usable."""
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={"durable_runs_enabled": True}),
            hermes_home=tmp_path,
        )
        try:
            result = store.submit_or_get(idempotency_key="k1", request_body={"input": "hi"})
            assert result.created is True
            assert (tmp_path / "durable_runs.db").exists()
        finally:
            _close(store)

    def test_db_path_override_via_extra(self, tmp_path):
        custom = tmp_path / "nested" / "custom_runs.db"
        store = build_durable_store(
            PlatformConfig(
                enabled=True,
                extra={
                    "durable_runs_enabled": True,
                    "durable_runs_db": str(custom),
                },
            ),
            hermes_home=tmp_path,
        )
        try:
            store.submit_or_get(idempotency_key="k1", request_body={})
            assert custom.exists()
            # Default location must NOT be used when an override is given.
            assert not (tmp_path / "durable_runs.db").exists()
        finally:
            _close(store)

    def test_db_path_override_via_env(self, monkeypatch, tmp_path):
        custom = tmp_path / "env_runs.db"
        monkeypatch.setenv(_ENV_FLAG, "1")
        monkeypatch.setenv(_ENV_DB, str(custom))
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={}), hermes_home=tmp_path
        )
        try:
            store.submit_or_get(idempotency_key="k1", request_body={})
            assert custom.exists()
        finally:
            _close(store)

    def test_default_home_resolves_via_get_hermes_home(self, monkeypatch, tmp_path):
        """Without an injected home, the store resolves get_hermes_home()."""
        import hermes_cli.config as hermes_config

        monkeypatch.setattr(hermes_config, "get_hermes_home", lambda: tmp_path)
        store = build_durable_store(
            PlatformConfig(enabled=True, extra={"durable_runs_enabled": True})
        )
        try:
            assert isinstance(store, DurableRunStore)
            store.submit_or_get(idempotency_key="k1", request_body={})
            assert (tmp_path / "durable_runs.db").exists()
        finally:
            _close(store)

    def test_factory_fences_true_process_before_sqlite_open(self, tmp_path):
        """A losing process cannot enable WAL/DDL before it is fenced."""
        db_path = tmp_path / "cross-process.db"
        config = PlatformConfig(
            enabled=True,
            extra={
                "durable_runs_enabled": True,
                "durable_runs_db": str(db_path),
            },
        )
        store = build_durable_store(config, hermes_home=tmp_path)
        try:
            before = db_path.stat()
            code = f"""
import sys
from gateway.config import PlatformConfig
from gateway.platforms.api_server import build_durable_store
config = PlatformConfig(enabled=True, extra={{
    'durable_runs_enabled': True,
    'durable_runs_db': {str(db_path)!r},
}})
try:
    build_durable_store(config)
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(2)
"""
            contender = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(tmp_path),
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                },
            )
            assert contender.returncode == 0, contender.stderr
            after = db_path.stat()
            assert (after.st_size, after.st_mtime_ns) == (
                before.st_size,
                before.st_mtime_ns,
            )
        finally:
            _close(store)


class TestDurableStoreOwnership:
    @pytest.mark.asyncio
    async def test_disconnect_closes_store_owned_by_adapter(self):
        store = MagicMock()
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={}),
            durable_store=store,
            owns_durable_store=True,
        )

        await adapter.disconnect()
        await adapter.disconnect()

        store.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_disconnect_closes_store_before_releasing_authority(self):
        order = []
        store = MagicMock()
        lock = MagicMock()
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={}),
            durable_store=store,
            owns_durable_store=True,
        )

        await adapter.disconnect()

        assert order == ["close", "release"]
