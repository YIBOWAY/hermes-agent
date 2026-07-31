# API Server Lifecycle Runbook

This runbook covers single-instance ownership, listener startup, supervised
restart handoff, and cleanup for `APIServerAdapter`.

## Evidence boundary

The behavior below describes the current checked-out source in
`gateway/platforms/api_server.py`, `gateway/durable_runs.py`, and
`gateway/restart.py`. It is not evidence that these bytes are installed or
running. Before making a claim about a live gateway, independently bind the
checkout, installed package, process argv, configuration, and loopback behavior
to the same reviewed source.

This runbook does not authorize installation, restart, provider use, dispatch,
trading, or public cutover.

## Startup invariants

When the API broker is enabled, startup spans two explicit phases:

1. During gateway factory construction, resolve the durable database path,
   acquire the process-scoped durable authority lock, then open and
   schema-initialize the SQLite store while fenced.
2. Construct `APIServerAdapter` with that already-fenced store.
3. When `connect()` begins, refuse any adapter whose earlier cleanup task or
   cleanup finalizer is still present, or whose cleanup is already known
   incomplete.
4. Validate the API-server key and required durable store.
5. Configure the aiohttp application and runner.
6. Bind the listener directly.
7. Reconcile durable Runs only after listener ownership is established.
8. Start the orphan sweep task and mark the adapter connected.

One adapter-level lifecycle lock serializes the complete `connect()` and
`disconnect()` bodies. A disconnect therefore cannot close the store or release
authority beneath a connect suspended in runner setup/listener bind; duplicate
serialized connect calls are idempotent and never replace a live
application/runner/site ownership tuple.

The factory must route every rejected/failed `connect()` through adapter
disconnect so that a key failure also closes the already-open store before
releasing its fence. A source-level call to `build_durable_store()` by itself
owns resources and therefore requires an explicit close; it is not a
side-effect-free capability probe.

A failed bind must not reconcile durable Runs or start the orphan sweep task.
The implementation intentionally does not pre-probe the port: a separate probe
can race the real bind and can misclassify a `TIME_WAIT` socket.

On macOS the listener explicitly disables `SO_REUSEADDR`; on Linux it retains
aiohttp's default behavior. The code never enables `SO_REUSEPORT`.

## Durable single-writer authority

`DurableAuthorityLock` canonicalizes the durable database path and uses
`<canonical-db-path>.authority.lock`. The operating-system lock is:

- non-blocking and process-scoped;
- held by an open file descriptor;
- non-expiring, with no lease timeout;
- released explicitly during cleanup or automatically when the process exits.

The fence is keyed by the database, not by the HTTP port. Two API servers using
different ports still cannot reconcile or mutate the same durable database.
Failure to acquire the fence sets non-retryable fatal error
`durable_authority_held` before application setup or listener bind.

Do **not** delete `.authority.lock` as a recovery action. The path may remain
after the owner exits even though the operating-system lock is free. Conversely,
unlinking a locked file can let a new process lock a different inode and defeat
the single-writer guarantee.

## Supervised cold-start bind handoff

Only an initial connection (`is_reconnect=False`) recognized as
supervisor-owned receives bounded `EADDRINUSE` retries. The delays are:

```text
0.25s, 0.5s, 1.0s, 2.0s
```

This is one initial bind plus at most four retries, with at most 3.75 seconds of
configured backoff. Each retry creates a fresh `TCPSite`. Other bind errors are
raised immediately.

Supervisor ownership is recognized when any of these conditions is true:

- `INVOCATION_ID` is non-empty;
- `HERMES_S6_SUPERVISED_CHILD` is non-empty;
- `XPC_SERVICE_NAME` is non-empty and not `0`;
- `HERMES_GATEWAY_EXTERNAL_SUPERVISOR` is one of
  `1`, `true`, `yes`, or `on`, case-insensitively.

| Situation | Retry `EADDRINUSE`? | Result after no successful bind |
|---|---:|---|
| Supervisor-owned initial connect | Yes, bounded | `api_server_port_in_use` |
| Reconnect, including under a supervisor | No | `api_server_port_in_use` |
| Non-supervised initial connect | No | `api_server_port_in_use` |
| Any non-`EADDRINUSE` bind error | No | Generic startup failure |

`api_server_port_in_use` is non-retryable so the gateway reconnect watcher does
not loop indefinitely. After resolving the conflict, change
`platforms.api_server.port` when necessary and run
`/platform resume api_server`.

## Cleanup and cancellation

Cleanup preserves database ownership until a factory-owned SQLite store has
finished its final close/checkpoint:

```text
stop partial site
  -> clean runner
  -> close factory-owned durable store
  -> release durable authority
  -> clear exact site / runner / application references
```

A borrowed durable store is not closed, but the adapter-owned authority lock is
still released.

If cancellation occurs during bind backoff or another startup await,
`connect()` performs the same cleanup and then re-raises
`asyncio.CancelledError`. Cancellation is not converted into `False` or a
retryable platform failure.

The first cleanup caller installs one shared cleanup task and one self-held
finalizer before its first await. Overlapping disconnect/cleanup callers join
that exact owner; they cannot create a second cleanup. `connect()` refuses while
either owner exists, including the interval before the five-second bound has
expired.

The caller waits at most five seconds for the combined site/runner cleanup, but
the cleanup task is not cancelled merely to satisfy that bound. If cleanup is
still pending, cancels itself, or raises before the runner proves cleanup, the
adapter sets non-retryable fatal `api_server_cleanup_incomplete` and retains the
exact site/runner references, durable store, and authority fence. A self-held
deferred finalizer performs `store close -> authority release -> state clear`
only after the original cleanup really succeeds. If store close or authority
release raises, references and the fence are likewise retained and only an
exact cleanup retry may try again. If cleanup never succeeds, the fence remains
held until process exit.

An external caller cancellation is remembered while cleanup continues. Once
cleanup has either completed or entered the fenced-incomplete state,
`asyncio.CancelledError` is re-raised; cancellation is never converted into a
successful disconnect or a retryable reconnect.

If bounded `EADDRINUSE` retries are exhausted, runner cleanup and
close-before-release occur before `connect()` returns `False`. No reconcile or
background sweep is allowed on this path.

Normal `disconnect()` uses the same cleanup fence. If direct `site.stop()`
raises but `AppRunner.cleanup()` succeeds, runner cleanup is the authoritative
recovery and close-before-release proceeds. If runner cleanup fails or remains
pending, disconnect retains authority and reports
`api_server_cleanup_incomplete`; a later exact cleanup retry or process exit is
required.

## Operator diagnosis

| Fatal code | Meaning | Recovery boundary |
|---|---|---|
| `durable_authority_held` | Another process owns writer/reconciler authority for the same durable database. | Identify the intended process, profile, database path, and service-manager owner. Do not unlink the lock file. Resume only after the competing owner has exited or configuration has been corrected. |
| `api_server_port_in_use` | The listener could not own the configured port after the applicable retry policy. | Identify the listener, stop only an unintended owner or choose another `platforms.api_server.port`, then run `/platform resume api_server`. |
| `api_server_cleanup_incomplete` | Site/runner cleanup has not proved completion, so this process intentionally retains durable authority. | Do not resume, unlink the lock, or start a replacement. Let the deferred cleanup finish; if it cannot, stop the owning process through its supervisor and verify process exit before replacement. |

On systems that provide `lsof`, these read-only checks can help:

```bash
api_server_port=8642
authority_lock_path=/path/to/durable-runs.db.authority.lock
lsof -nP -iTCP:"$api_server_port" -sTCP:LISTEN
lsof "$authority_lock_path"
```

Treat an empty `lsof` result as an observation, not permission to delete the
lock file or restart a service. Container or service-manager namespaces may
hide the actual owner.

## Source-level verification

The focused source test is:

```bash
python -m pytest tests/gateway/test_api_server_bind_guard.py -q
```

It covers:

- same-database single-writer exclusion;
- bind-before-reconcile;
- transient supervised cold-start handoff;
- bounded retry exhaustion;
- no bind retry during reconnect;
- close-before-release on bind and reconcile failure;
- cancellation cleanup and cancellation propagation;
- pending/cancel-resistant cleanup retains authority until real completion;
- overlapping cleanup callers share one owner and block reconnect immediately;
- reverse overlap keeps disconnect behind an in-flight connect lifecycle;
- store-close uncertainty retains both the store reference and authority fence;
- normal disconnect recovery, failure fencing, and external cancellation.

Passing source tests does not establish installed or live-runtime identity.

## Related documents

- [DurableRunAuthority Contract Matrix](durable-run-authority-contract.md)
- [Session Lifecycle](session-lifecycle.md) — conversation/session recovery,
  which is separate from listener and durable-database ownership
