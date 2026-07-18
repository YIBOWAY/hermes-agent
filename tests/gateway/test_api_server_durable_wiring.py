"""V2.13 — opt-in durable-store wiring for the api_server platform.

The reviewed V2.1–V2.9 durable-run semantics are dormant unless something
passes a ``durable_store`` into ``APIServerAdapter`` (``_broker_enabled()`` is
just ``store is not None``). ``build_durable_store(config)`` is the single,
unit-testable construction point ``gateway/run.py`` uses so the feature is
**opt-in and default-OFF**: with no enabling config the adapter gets
``durable_store=None`` and ``/v1/runs`` keeps its legacy in-memory behavior
byte-identical.

Red line: default construction must be side-effect-free (no DB file, no store)
so installing this code never changes live behavior unless explicitly enabled.
"""

import pytest

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore
from gateway.platforms.api_server import build_durable_store

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
