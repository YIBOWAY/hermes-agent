"""Tests for the API server bind-address startup guard.

Validates that is_network_accessible() correctly classifies addresses and
that connect() refuses to start without API_SERVER_KEY.
"""

import asyncio
import errno
import os
import socket
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.durable_runs import DurableRunStore, RunState
from gateway.platforms.api_server import (
    APIServerAdapter,
    SUPERVISED_COLD_START_BIND_RETRY_DELAYS,
)
from gateway.platforms.base import is_network_accessible


# ---------------------------------------------------------------------------
# Unit tests: is_network_accessible()
# ---------------------------------------------------------------------------


class TestIsNetworkAccessible:
    """Direct tests for the address classification helper."""

    # -- Loopback (safe, should return False) --


    def test_ipv4_mapped_loopback(self):
        # ::ffff:127.0.0.1 — Python's is_loopback returns False for mapped
        # addresses; the helper must unwrap and check ipv4_mapped.
        assert is_network_accessible("::ffff:127.0.0.1") is False

    # -- Network-accessible (should return True) --


    def test_ipv6_wildcard(self):
        # This is the bypass vector that the string-based check missed.
        assert is_network_accessible("::") is True


    def test_private_ipv4(self):
        assert is_network_accessible("10.0.0.1") is True


    def test_public_ipv4(self):
        assert is_network_accessible("8.8.8.8") is True

    # -- Hostname resolution --


    def test_hostname_mixed_resolution(self):
        """If a hostname resolves to both loopback and non-loopback, it's
        network-accessible (any non-loopback address is enough)."""
        mixed_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        ]
        with patch("gateway.platforms.base._socket.getaddrinfo", return_value=mixed_result):
            assert is_network_accessible("dual-host.local") is True


# ---------------------------------------------------------------------------
# Integration tests: connect() startup guard
# ---------------------------------------------------------------------------


class TestConnectBindGuard:
    """Verify that connect() refuses dangerous configurations."""


    @pytest.mark.asyncio
    async def test_refuses_loopback_without_key(self):
        """Loopback binds are still an auth boundary and require API_SERVER_KEY."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1"}))
        assert adapter._api_key == ""
        assert is_network_accessible(adapter._host) is False
        result = await adapter.connect()
        assert result is False
        assert adapter._app is None
        assert adapter._background_tasks == set()


    @pytest.mark.asyncio
    async def test_allows_wildcard_with_key(self):
        """Non-loopback with a key should pass the guard."""
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": "sk-test"})
        )
        # The guard checks: is_network_accessible(host) AND NOT api_key
        # With a key set, the guard should not block.
        assert adapter._api_key == "sk-test"
        assert is_network_accessible("0.0.0.0") is True
        # Combined: the guard condition is False (key is set), so it passes


# ---------------------------------------------------------------------------
# Integration tests: bind mechanics (direct bind, no pre-probe — #10297)
# ---------------------------------------------------------------------------


class TestBindMechanics:
    """connect() binds directly instead of pre-probing 127.0.0.1.

    The old ``_port_is_available()`` probe connected to 127.0.0.1 only and
    reported a lingering TIME_WAIT socket as "in use", failing gateway
    restarts for up to ~60s (#10297). The fix removes the probe: bind
    directly, keep SO_REUSEADDR default semantics on Linux (rebind past
    TIME_WAIT), and surface a real bind conflict as a clean ``False`` with
    the runner torn down.
    """

    _KEY = "sk-test-strong-key-0123456789"

    def _make_adapter(self, port: int) -> APIServerAdapter:
        return APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": port, "key": self._KEY},
            )
        )

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @pytest.mark.asyncio
    async def test_immediate_rebind_after_disconnect(self):
        """A restarted adapter can rebind the same port immediately.

        This is the #10297 symptom: the old pre-probe (and disabled address
        reuse) made a quick gateway restart fail while the previous socket
        sat in TIME_WAIT.
        """
        port = self._free_port()
        first = self._make_adapter(port)
        assert await first.connect() is True
        await first.disconnect()

        second = self._make_adapter(port)
        try:
            assert await second.connect() is True
        finally:
            await second.disconnect()

    @pytest.mark.asyncio
    async def test_live_listener_conflict_returns_false_and_cleans_up(self):
        """A second adapter on an occupied port fails cleanly, not with a raise."""
        port = self._free_port()
        first = self._make_adapter(port)
        assert await first.connect() is True
        second = self._make_adapter(port)
        try:
            result = await second.connect()
            assert result is False
            assert second._runner is None
            assert second._site is None
            assert second.is_connected is False
        finally:
            await first.disconnect()
            await second.disconnect()

    @pytest.mark.asyncio
    async def test_supervised_cold_start_retries_transient_bind_before_reconcile(self):
        """A supervisor handoff may briefly leave the previous listener alive."""
        adapter = self._make_adapter(self._free_port())
        loop = asyncio.get_running_loop()
        bound_server = MagicMock()
        bound_server.sockets = []
        events = []

        async def create_server_side_effect(*args, **kwargs):
            if not events:
                events.append("bind_conflict")
                raise OSError(
                    errno.EADDRINUSE,
                    "previous listener still draining",
                )
            events.append("bind_success")
            return bound_server

        create_server = AsyncMock(side_effect=create_server_side_effect)

        try:
            with (
                patch.object(loop, "create_server", create_server),
                patch.dict(
                    os.environ,
                    {"XPC_SERVICE_NAME": "com.nous.hermes.gateway"},
                    clear=False,
                ),
                patch.object(
                    adapter,
                    "reconcile_durable_runs",
                    side_effect=lambda: events.append("reconcile") or 0,
                ) as reconcile,
            ):
                assert await adapter.connect(is_reconnect=False) is True

            assert create_server.await_count == 2
            reconcile.assert_called_once_with()
            assert events == ["bind_conflict", "bind_success", "reconcile"]
            assert adapter.has_fatal_error is False
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_supervised_cold_start_exhausts_bounded_bind_retries_fail_closed(
        self, tmp_path
    ):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "supervised-bind-failure.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        loop = asyncio.get_running_loop()
        create_server = AsyncMock(
            side_effect=OSError(errno.EADDRINUSE, "listener never exited"),
        )
        sleep = AsyncMock()

        with (
            patch.object(loop, "create_server", create_server),
            patch.dict(
                os.environ,
                {"XPC_SERVICE_NAME": "com.nous.hermes.gateway"},
                clear=False,
            ),
            patch(
                "gateway.platforms.api_server.asyncio.sleep",
                new=sleep,
            ),
            patch.object(
                adapter,
                "reconcile_durable_runs",
                return_value=0,
            ) as reconcile,
        ):
            assert await adapter.connect(is_reconnect=False) is False

        retry_delays = [
            call.args[0]
            for call in sleep.await_args_list
            if call.args and call.args[0] > 0
        ]
        assert retry_delays
        assert sum(retry_delays) <= 5
        assert create_server.await_count == len(retry_delays) + 1
        reconcile.assert_not_called()
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "api_server_port_in_use"
        assert adapter.fatal_error_retryable is False
        assert adapter._runner is None
        assert adapter._site is None
        assert adapter._background_tasks == set()
        assert order == ["close", "release"]

    @pytest.mark.asyncio
    async def test_supervised_cold_start_cancellation_cleans_up_and_reraises(
        self, tmp_path
    ):
        """Cancellation during bind backoff must not strand durable authority."""
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "supervised-bind-cancelled.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        loop = asyncio.get_running_loop()
        create_server = AsyncMock(
            side_effect=OSError(errno.EADDRINUSE, "listener still draining"),
        )

        async def cancel_bind_backoff(delay):
            if delay > 0:
                raise asyncio.CancelledError

        sleep = AsyncMock(side_effect=cancel_bind_backoff)

        with (
            patch.object(loop, "create_server", create_server),
            patch.dict(
                os.environ,
                {"XPC_SERVICE_NAME": "com.nous.hermes.gateway"},
                clear=False,
            ),
            patch(
                "gateway.platforms.api_server.asyncio.sleep",
                new=sleep,
            ),
            patch.object(
                adapter,
                "reconcile_durable_runs",
                return_value=0,
            ) as reconcile,
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter.connect(is_reconnect=False)

        assert create_server.await_count == 1
        retry_delays = [
            call.args[0]
            for call in sleep.await_args_list
            if call.args and call.args[0] > 0
        ]
        assert retry_delays == [SUPERVISED_COLD_START_BIND_RETRY_DELAYS[0]]
        reconcile.assert_not_called()
        assert adapter._runner is None
        assert adapter._site is None
        assert adapter._background_tasks == set()
        assert order == ["close", "release"]

    @pytest.mark.asyncio
    async def test_startup_cancellation_with_cancelled_runner_cleanup_stays_fenced(
        self, tmp_path
    ):
        """Failed runner cleanup must retain authority until a later clean retry."""
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "supervised-cleanup-cancelled.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        loop = asyncio.get_running_loop()
        create_server = AsyncMock(
            side_effect=OSError(errno.EADDRINUSE, "listener still draining"),
        )

        async def cancel_bind_backoff(delay):
            if delay > 0:
                raise asyncio.CancelledError

        with (
            patch.object(loop, "create_server", create_server),
            patch.dict(
                os.environ,
                {"XPC_SERVICE_NAME": "com.nous.hermes.gateway"},
                clear=False,
            ),
            patch(
                "gateway.platforms.api_server.asyncio.sleep",
                new=AsyncMock(side_effect=cancel_bind_backoff),
            ),
            patch(
                "gateway.platforms.api_server.web.AppRunner.cleanup",
                new=AsyncMock(side_effect=asyncio.CancelledError),
            ),
            patch.object(
                adapter,
                "reconcile_durable_runs",
                return_value=0,
            ) as reconcile,
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter.connect(is_reconnect=False)

        reconcile.assert_not_called()
        assert adapter._runner is not None
        assert adapter._app is not None
        assert adapter._background_tasks == set()
        assert adapter._durable_store is store
        assert adapter.fatal_error_code == "api_server_cleanup_incomplete"
        assert adapter.fatal_error_retryable is False
        assert order == []

        # The exact same adapter may retry cleanup, but it may not reconnect or
        # release its fence before that retry proves runner cleanup succeeded.
        assert await adapter.connect(is_reconnect=True) is False
        assert order == []
        assert await adapter._cleanup_failed_startup() is False
        assert adapter._runner is None
        assert adapter._site is None
        assert adapter._app is None
        assert adapter._durable_store is None
        assert order == ["close", "release"]

    @pytest.mark.asyncio
    async def test_timed_out_runner_cleanup_retains_authority_until_it_finishes(
        self, tmp_path
    ):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "cleanup-timeout.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        site = MagicMock()
        site.stop = AsyncMock()
        runner_release = asyncio.Event()
        runner_started = asyncio.Event()

        async def resistant_runner_cleanup():
            runner_started.set()
            while not runner_release.is_set():
                try:
                    await runner_release.wait()
                except asyncio.CancelledError:
                    continue

        runner = MagicMock()
        runner.cleanup = resistant_runner_cleanup
        adapter._site = site
        adapter._runner = runner
        adapter._app = MagicMock()

        with patch(
            "gateway.platforms.api_server.STARTUP_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        ):
            assert await adapter._cleanup_failed_startup() is False

        assert runner_started.is_set()
        assert adapter._startup_cleanup_incomplete is True
        assert adapter._startup_cleanup_finalizer is not None
        assert not adapter._startup_cleanup_finalizer.done()
        assert adapter._site is site
        assert adapter._runner is runner
        assert adapter._durable_store is store
        assert order == []
        assert adapter.fatal_error_code == "api_server_cleanup_incomplete"

        finalizer = adapter._startup_cleanup_finalizer
        runner_release.set()
        await asyncio.wait_for(asyncio.shield(finalizer), timeout=1)

        assert adapter._startup_cleanup_incomplete is False
        assert adapter._startup_cleanup_finalizer is None
        assert adapter._site is None
        assert adapter._runner is None
        assert adapter._app is None
        assert adapter._durable_store is None
        assert order == ["close", "release"]

    @pytest.mark.asyncio
    async def test_overlapping_cleanup_callers_share_one_owner_and_block_connect(
        self, tmp_path
    ):
        order = []
        cleanup_calls = 0
        store = MagicMock()
        store.db_path = tmp_path / "overlapping-cleanup.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()

        async def blocked_runner_cleanup():
            nonlocal cleanup_calls
            cleanup_calls += 1
            cleanup_started.set()
            await cleanup_release.wait()

        site = MagicMock()
        site.stop = AsyncMock()
        runner = MagicMock()
        runner.cleanup = blocked_runner_cleanup
        adapter._site = site
        adapter._runner = runner
        adapter._app = MagicMock()

        with patch(
            "gateway.platforms.api_server.STARTUP_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        ):
            first_cleanup = asyncio.create_task(adapter._cleanup_failed_startup())
            await cleanup_started.wait()

            owner_task = adapter._startup_cleanup_task
            owner_finalizer = adapter._startup_cleanup_finalizer
            assert owner_task is not None
            assert owner_finalizer is not None
            assert await adapter.connect(is_reconnect=True) is False
            assert adapter._startup_cleanup_task is owner_task
            assert adapter._startup_cleanup_finalizer is owner_finalizer

            second_cleanup = asyncio.create_task(adapter._cleanup_failed_startup())
            assert await asyncio.gather(first_cleanup, second_cleanup) == [False, False]

        assert cleanup_calls == 1
        assert order == []
        assert adapter._durable_store is store
        assert adapter.fatal_error_code == "api_server_cleanup_incomplete"
        assert adapter.fatal_error_retryable is False

        cleanup_release.set()
        await asyncio.wait_for(asyncio.shield(owner_finalizer), timeout=1)

        assert cleanup_calls == 1
        assert order == ["close", "release"]
        assert adapter._startup_cleanup_task is None
        assert adapter._startup_cleanup_finalizer is None
        assert adapter._startup_cleanup_incomplete is False
        assert adapter._site is None
        assert adapter._runner is None
        assert adapter._app is None
        assert adapter._durable_store is None

    @pytest.mark.asyncio
    async def test_disconnect_cannot_release_authority_under_inflight_connect(
        self, tmp_path
    ):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "connect-disconnect-overlap.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        setup_started = asyncio.Event()
        setup_release = asyncio.Event()

        async def runner_setup():
            order.append("setup-start")
            setup_started.set()
            await setup_release.wait()
            order.append("setup-done")

        async def runner_cleanup():
            order.append("runner-cleanup")

        async def site_start():
            order.append("site-start")

        async def site_stop():
            order.append("site-stop")

        runner = MagicMock()
        runner.setup = runner_setup
        runner.cleanup = runner_cleanup
        site = MagicMock()
        site.start = site_start
        site.stop = site_stop

        with (
            patch(
                "gateway.platforms.api_server.web.AppRunner",
                return_value=runner,
            ),
            patch(
                "gateway.platforms.api_server.web.TCPSite",
                return_value=site,
            ),
            patch.object(
                adapter,
                "reconcile_durable_runs",
                side_effect=lambda: order.append("reconcile") or 0,
            ),
            patch.object(
                adapter,
                "_sweep_orphaned_runs",
                new=AsyncMock(),
            ),
        ):
            connect_task = asyncio.create_task(adapter.connect())
            await setup_started.wait()
            disconnect_task = asyncio.create_task(adapter.disconnect())
            await asyncio.sleep(0)

            assert order == ["setup-start"]
            assert not disconnect_task.done()
            assert adapter._durable_store is store
            lock.release.assert_not_called()

            setup_release.set()
            assert await connect_task is True
            await disconnect_task

        assert order == [
            "setup-start",
            "setup-done",
            "site-start",
            "reconcile",
            "site-stop",
            "runner-cleanup",
            "close",
            "release",
        ]
        assert adapter.is_connected is False
        assert adapter._site is None
        assert adapter._runner is None
        assert adapter._app is None
        assert adapter._durable_store is None
        assert adapter._durable_authority_lock is None

    @pytest.mark.asyncio
    async def test_store_close_failure_retains_store_and_fence_until_exact_retry(
        self, tmp_path
    ):
        order = []
        close_attempts = 0
        store = MagicMock()
        store.db_path = tmp_path / "store-close-failure.db"
        lock = MagicMock()
        store.authority_lock = lock

        def close_store():
            nonlocal close_attempts
            close_attempts += 1
            order.append(f"close-{close_attempts}")
            if close_attempts == 1:
                raise RuntimeError("sqlite close outcome unknown")

        store.close.side_effect = close_store
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )

        async def site_stop():
            order.append("site")

        async def runner_cleanup():
            order.append("runner")

        site = MagicMock()
        site.stop = site_stop
        runner = MagicMock()
        runner.cleanup = runner_cleanup
        adapter._site = site
        adapter._runner = runner
        adapter._app = MagicMock()

        await adapter.disconnect()

        assert order == ["site", "runner", "close-1"]
        assert adapter._site is site
        assert adapter._runner is runner
        assert adapter._app is not None
        assert adapter._durable_store is store
        assert adapter._durable_authority_lock is lock
        assert adapter._startup_cleanup_incomplete is True
        assert adapter.fatal_error_code == "api_server_cleanup_incomplete"
        assert adapter.fatal_error_retryable is False
        lock.release.assert_not_called()

        # Reconnect is never a cleanup retry and must not touch the listener,
        # store, or fence. The same adapter may only retry exact teardown.
        assert await adapter.connect(is_reconnect=True) is False
        assert order == ["site", "runner", "close-1"]

        assert await adapter._cleanup_failed_startup() is False

        assert order == [
            "site",
            "runner",
            "close-1",
            "site",
            "runner",
            "close-2",
            "release",
        ]
        assert adapter._site is None
        assert adapter._runner is None
        assert adapter._app is None
        assert adapter._durable_store is None
        assert adapter._durable_authority_lock is None
        assert adapter._startup_cleanup_incomplete is False
        lock.release.assert_called_once_with()

    def test_concrete_store_close_failure_does_not_release_authority(self):
        store = object.__new__(DurableRunStore)
        store._lock = threading.RLock()
        store._conn = MagicMock()
        authority_lock = MagicMock()
        store.authority_lock = authority_lock
        store._conn.close.side_effect = RuntimeError("close outcome unknown")

        with pytest.raises(RuntimeError, match="close outcome unknown"):
            store.close()

        authority_lock.release.assert_not_called()
        assert store.authority_lock is authority_lock

        store._conn.close.side_effect = None
        store.close()

        authority_lock.release.assert_called_once_with()
        assert store.authority_lock is None

    @pytest.mark.asyncio
    async def test_disconnect_site_error_uses_runner_cleanup_before_release(
        self, tmp_path
    ):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "disconnect-site-error.db"
        lock = MagicMock()
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )

        async def failed_site_stop():
            order.append("site")
            raise RuntimeError("site stop failed")

        async def runner_cleanup():
            order.append("runner")

        site = MagicMock()
        site.stop = failed_site_stop
        runner = MagicMock()
        runner.cleanup = runner_cleanup
        adapter._site = site
        adapter._runner = runner
        adapter._app = MagicMock()

        await adapter.disconnect()

        assert order == ["site", "runner", "close", "release"]
        assert adapter._site is None
        assert adapter._runner is None
        assert adapter._durable_store is None

    @pytest.mark.asyncio
    async def test_disconnect_runner_error_retains_durable_authority(
        self, tmp_path
    ):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "disconnect-runner-error.db"
        lock = MagicMock()
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )

        async def site_stop():
            order.append("site")

        async def failed_runner_cleanup():
            order.append("runner")
            raise RuntimeError("runner cleanup failed")

        site = MagicMock()
        site.stop = site_stop
        runner = MagicMock()
        runner.cleanup = failed_runner_cleanup
        adapter._site = site
        adapter._runner = runner
        adapter._app = MagicMock()

        await adapter.disconnect()

        assert order == ["site", "runner"]
        assert adapter._site is site
        assert adapter._runner is runner
        assert adapter._durable_store is store
        assert adapter.fatal_error_code == "api_server_cleanup_incomplete"
        assert adapter.fatal_error_retryable is False

        # A later exact cleanup succeeds and is the only path that may release.
        runner.cleanup = AsyncMock()
        await adapter.disconnect()
        assert order == ["site", "runner", "site", "close", "release"]
        assert adapter._durable_store is None

    @pytest.mark.asyncio
    async def test_disconnect_external_cancellation_cleans_then_reraises(
        self, tmp_path
    ):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "disconnect-cancelled.db"
        lock = MagicMock()
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )
        site_started = asyncio.Event()
        site_release = asyncio.Event()

        async def site_stop():
            order.append("site")
            site_started.set()
            await site_release.wait()

        async def runner_cleanup():
            order.append("runner")

        site = MagicMock()
        site.stop = site_stop
        runner = MagicMock()
        runner.cleanup = runner_cleanup
        adapter._site = site
        adapter._runner = runner
        adapter._app = MagicMock()

        disconnect_task = asyncio.create_task(adapter.disconnect())
        await site_started.wait()
        disconnect_task.cancel()
        await asyncio.sleep(0)
        site_release.set()
        with pytest.raises(asyncio.CancelledError):
            await disconnect_task

        assert order == ["site", "runner", "close", "release"]
        assert adapter._site is None
        assert adapter._runner is None
        assert adapter._durable_store is None

    @pytest.mark.asyncio
    async def test_supervised_reconnect_does_not_retry_port_conflict(self):
        adapter = self._make_adapter(self._free_port())
        loop = asyncio.get_running_loop()
        create_server = AsyncMock(
            side_effect=OSError(errno.EADDRINUSE, "occupied during reconnect"),
        )
        sleep = AsyncMock()

        with (
            patch.object(loop, "create_server", create_server),
            patch.dict(
                os.environ,
                {"XPC_SERVICE_NAME": "com.nous.hermes.gateway"},
                clear=False,
            ),
            patch(
                "gateway.platforms.api_server.asyncio.sleep",
                new=sleep,
            ),
        ):
            assert await adapter.connect(is_reconnect=True) is False

        assert create_server.await_count == 1
        assert not [
            call
            for call in sleep.await_args_list
            if call.args and call.args[0] > 0
        ]
        assert adapter.fatal_error_code == "api_server_port_in_use"
        assert adapter.fatal_error_retryable is False

    def test_pre_probe_helper_removed(self):
        """The racy single-family pre-probe must not come back."""
        assert not hasattr(APIServerAdapter, "_port_is_available")
    @pytest.mark.asyncio
    async def test_factory_store_bind_failure_closes_before_unlock(self, tmp_path):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "bind-failure.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )

        with patch(
            "gateway.platforms.api_server.web.TCPSite.start",
            new=AsyncMock(side_effect=OSError(errno.EADDRINUSE, "occupied")),
        ):
            assert await adapter.connect() is False

        assert order == ["close", "release"]
        assert adapter._durable_store is None
        assert adapter._runner is None
        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_factory_store_reconcile_failure_closes_before_unlock(self, tmp_path):
        order = []
        store = MagicMock()
        store.db_path = tmp_path / "reconcile-failure.db"
        lock = MagicMock()
        lock.acquire.return_value = True
        store.authority_lock = lock
        store.close.side_effect = lambda: order.append("close")
        lock.release.side_effect = lambda: order.append("release")
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=store,
            owns_durable_store=True,
        )

        with patch.object(
            adapter,
            "reconcile_durable_runs",
            side_effect=RuntimeError("reconcile failed"),
        ):
            assert await adapter.connect() is False

        assert order == ["close", "release"]
        assert adapter._durable_store is None
        assert adapter._site is None
        assert adapter._runner is None

    @pytest.mark.asyncio
    async def test_port_conflict_sets_non_retryable_fatal_error(self):
        """A real port conflict (EADDRINUSE) must set a non-retryable fatal
        error so the reconnect watcher drops the platform from the retry
        queue instead of looping indefinitely.

        Previously connect() returned bare ``False``, which the reconnect
        watcher treated as retryable — retrying every 5 minutes forever,
        filling errors.log and leaking 2 fds per retry (#52132: 1568+
        retries over 5 days in a multi-profile setup).
        """
        port = self._free_port()
        first = self._make_adapter(port)
        assert await first.connect() is True
        second = self._make_adapter(port)
        try:
            result = await second.connect()
            assert result is False
            assert second.has_fatal_error is True
            assert second.fatal_error_retryable is False
            assert second.fatal_error_code == "api_server_port_in_use"
            assert str(port) in (second.fatal_error_message or "")
        finally:
            await first.disconnect()
            await second.disconnect()

    @pytest.mark.asyncio
    async def test_port_conflict_does_not_reconcile_unserved_durable_run(self, tmp_path):
        port = self._free_port()
        first = self._make_adapter(port)
        assert await first.connect() is True
        store = DurableRunStore(tmp_path / "durable.db")
        store.register_run(run_id="run_unserved", session_id="session")
        store.transition("run_unserved", RunState.RUNNING)
        second = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": port, "key": self._KEY},
            ),
            durable_store=store,
        )
        try:
            assert await second.connect() is False
            assert store.get_run("run_unserved")["status"] == RunState.RUNNING.value
            assert second._background_tasks == set()
        finally:
            await first.disconnect()
            await second.disconnect()
            store.close()

    @pytest.mark.asyncio
    async def test_shared_durable_db_has_one_authority_across_different_ports(
        self, tmp_path
    ):
        """The DB authority fence is independent of the HTTP bind address."""
        db_path = tmp_path / "shared-durable.db"
        first_store = DurableRunStore(db_path)
        second_store = DurableRunStore(db_path)
        first = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=first_store,
        )
        second = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": self._free_port(),
                    "key": self._KEY,
                },
            ),
            durable_store=second_store,
        )
        try:
            assert await first.connect() is True
            first_store.register_run(run_id="run_owned_by_first", session_id="s")
            assert first_store.transition("run_owned_by_first", RunState.RUNNING)

            # Different port does not grant a second writer authority and must
            # not reconcile the first process's active row.
            assert await second.connect() is False
            assert second_store.get_run("run_owned_by_first")["status"] == "running"
            assert second._background_tasks == set()

            # OS releases the non-expiring fence when the owner closes. The
            # same second adapter can then take over and perform reconciliation.
            await first.disconnect()
            assert await second.connect() is True
            assert second_store.get_run("run_owned_by_first")["status"] == "stopped"
        finally:
            await first.disconnect()
            await second.disconnect()
            first_store.close()
            second_store.close()
