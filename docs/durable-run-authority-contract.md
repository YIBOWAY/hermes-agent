# DurableRunAuthority — Contract Matrix (V2.1)

> Status: **authoritative contract** for Agent v0.2 Slice V2. Frozen 2026-07-17 on
> base `0bf44d557f4564c9d7d84cbf7632b02015f00271` (candidate branch
> `codex/agent-v0-2-durable-runs`, isolated worktree — **no live checkout change**).
>
> This matrix is the single alignment baseline for the nine durable-run semantics.
> Every V2.2–V2.8 implementation slice must cite the rows it satisfies; the V2.11
> unfakeable acceptance tests are derived directly from §6. **Red line: zero live
> effect until V2.11 is green and V2.12 (install/restart) is separately authorized.**

> **Current delivery evidence (2026-07-19): ISOLATED CODE COMMITTED / LIVE NOT
> INSTALLED.** The implementation is committed on
> `codex/v2-live-integration@a22d21b207661849a78bb236e81192cf5295cbd6`
> (release parent `b3343a658f62`); live remains
> `codex/v2-live-installed@916f5fbf5` with durable OFF. Fresh release validation
> has 389 focused durable/API/approval/provider/stop tests and 436 real
> conversation-loop tests passing, with Ruff and `git diff --check` green. The
> earlier 408-test frozen suite and two independent adversarial ACCEPT reviews
> remain historical pre-commit evidence. This evidence does
> not authorize install, restart, provider use, dispatch, browser mutation, or a
> public composer.
>
> The implementation refines §1/§2 approval and provider evidence as follows:
> `approval.decision_recorded` is the immutable human choice;
> `approval.release_committed` is the durable at-most-once linearization point
> with `waiter_signal_status=unknown`; only a later `approval.signalled` proves
> the exact in-process waiter observed `Event.set()`. HTTP reports
> `decision_status=committed` plus `waiter_signal_status=confirmed|unknown` and
> never treats either as tool execution outcome. Actual provider/model/usage is
> emitted only from an execution-middleware response receipt; preflight,
> middleware short-circuit, provider exception, requested route, and fallback
> config never counterfeit actual evidence. Public evidence contains only the
> observed model/provider, not fallback secrets or routing configuration.

The platform must never *project* fake upstream run state. All run truth lives in
Hermes' canonical authority; the platform reads it through `OfficialHermesHttpAdapter`
(V2.10). If any guarantee in this matrix cannot be met, the production adapter reports
`unavailable` and dispatch stays **OFF**.

## 0. Terminology

| Term | Meaning |
|---|---|
| **Run** | One durable unit of agent work submitted via `POST /v1/runs`. |
| **request identity** | The caller-supplied `Idempotency-Key` **plus** the canonical request digest. Two submissions share identity iff both match. |
| **canonical request digest** | `sha256` over the request serialized as **canonical JSON** (UTF-8, `sort_keys`, tight separators `(",", ":")`, no `repr()`, normalized types) of the full request-identity field set. |
| **event** | A canonical, persisted fact about a Run (state transition, message delta, tool start/complete, reasoning, approval request/response, terminal). |

## 1. Run state machine (normative)

States and allowed transitions. **Terminal states are immutable** — a terminal Run is
never rewritten (mirrors the append-only discipline already proven for the platform
ledger in V1.2A).

```
queued ──▶ running ──▶ succeeded        (terminal)
  │          │
  │          ├──────▶ failed            (terminal)
  │          │
  │          └──────▶ stopped           (terminal; via stop intent)
  │
  └──────────▶ stopped                  (stop before start)
```

| Contract state | Upstream today (api_server.py) | Mapping rule |
|---|---|---|
| `queued` | `queued` | 1:1 |
| `running` | `running` | 1:1 |
| `succeeded` | `completed` | rename to contract term |
| `failed` | `failed` | 1:1 |
| `stopped` | `cancelled` | rename to contract term |
| — | `waiting_for_approval` | **sub-state of `running`**, surfaced as `running` + `substate=waiting_for_approval` |
| — | `stopping` | **sub-state**, surfaced as `running` + `substate=stopping` until terminal `stopped` |

Invariant: exactly the five contract states; upstream-only states are projected onto
sub-states, never exposed as top-level status.

## 2. The nine semantics (normative matrix)

| # | Semantic | Contract requirement | Upstream status (base `0bf44d557`) | V2 slice |
|---|---|---|---|---|
| 1 | **Idempotency key + canonical digest** | `POST /v1/runs` honors caller `Idempotency-Key`; server computes & persists the canonical digest as part of run identity. | partial — key honored only on `/v1/chat/completions` + `/v1/responses` non-stream (`api_server.py:2791,3889`); `_IdempotencyCache` in-mem TTL300/LRU1000; digest via `repr()` (non-canonical). `POST /v1/runs` has **none** (`:4677,:4750`). | V2.2 |
| 2 | **submit-or-get + recovery by request identity** | Same request identity ⇒ returns the *same* Run (no duplicate); a by-identity lookup recovers a Run after restart. | partial — durable submit-or-get only at kanban *task* granularity (`hermes_cli/kanban_db.py:2545-2555`); no standalone by-key lookup; none on `/v1/runs`. | V2.3 |
| 3 | **Persistent Run identity/status/session policy** | Run id, status, and run→session binding survive process restart; resume/reattach by persisted id. | **missing** — `_run_statuses`/`_run_streams` in-memory, empty on restart, terminal records TTL-swept 3600s (`api_server.py:987,4612`). | V2.4 |
| 4 | **Stable event ID + monotonic cursor + replay + SSE** | Each event has a stable `event_id` and a per-Run monotonic `seq`; replay from any cursor with **no gap, no duplicate**; SSE disconnect must **not** delete canonical events. | **missing** — no event_id/cursor, no Last-Event-ID, no persistent log; SSE `finally` deletes the queue; 300s sweep; two subscribers race one queue (`api_server.py:5039-5088,975,4760`). | V2.5 |
| 5 | **requested vs actual provider/model/fallback/usage evidence** | Record requested policy *and* actual provider/model, any fallback (+reason), and usage; never overwrite the requested route. | partial — usage recorded, but `first_accounted_route` **overwrites** requested route (`hermes_state.py:2875-2895`); fallback only to agent.log; no `requested_model` column. | V2.6 |
| 6 | **exact approval challenge + TTL + single-use + CAS** | Gated action issues an exact challenge (digest/nonce); grant is time-boxed (TTL), consume-once (single-use), and compare-and-swap so stale/expired/digest-mismatched grants cannot mutate. | partial — approval gate exists (`tools/approval.py`); no challenge token/nonce, session/always grants have no TTL, no atomic consume-once, no CAS. Reusable precedent: 60s dangerous-confirmation expiry (`agent/replay_cleanup.py:210`). | V2.7 |
| 7 | **idempotent stop intent + restart reconcile** | Repeat stop on a Run is a no-op returning the same result (even after terminal); on restart, in-flight Runs reconcile to a deterministic terminal/known state (no phantom-running). | partial — `agent.interrupt()` idempotent; but `/v1/runs/{id}/stop` 404s after terminal (`api_server.py:5189`); session reconcile drives toward *resume* not mark-stopped. | V2.8 |
| 8 | **reviewed API contract + behavioral capability probe** | `/v1/capabilities` reports **behavioral** evidence (can it actually submit/persist/interrupt/stream now), not static flags; supports feature negotiation. | partial — five `/v1/runs` endpoints exist; but `features` block is hardcoded literals (`api_server.py:1980-2006`), no handshake, no behavioral probe. Reusable: `relay/descriptor.py` `CONTRACT_VERSION`, `collect_runtime_readiness`. | V2.9 |
| 9 | **(cross-cutting) durability across restart** | All run-truth (identity, status, events, approval, evidence) persists across process restart. | **missing** — all run state in-memory. | V2.4 (store) underlies 2,3,4,6,7 |

## 3. HTTP contract (target surface)

| Method & path | Purpose | Notes |
|---|---|---|
| `POST /v1/runs` | submit-or-get by request identity | 202 + `run_id`; honors `Idempotency-Key`; persists canonical digest |
| `GET /v1/runs/{run_id}` | current status + evidence | 404 `run_not_found`; includes requested/actual policy, usage, fallback |
| `GET /v1/runs/{run_id}/events?since={cursor}` | page replay (no gap/dup) + SSE live tail | honors `Last-Event-ID` / `since`; stable `event_id` + monotonic `seq` |
| `POST /v1/runs/{run_id}/approval` | respond to an exact challenge | single-use + TTL + CAS; reports committed decision and confirmed/unknown waiter signal separately; 409 on stale/expired/digest-mismatch |
| `POST /v1/runs/{run_id}/stop` | idempotent stop intent | repeat ⇒ same result, even after terminal |
| `GET /v1/capabilities` | **behavioral** capability probe + negotiation | reports what the authority can do *now*, with evidence |

## 4. Durable store

A single content-addressed, append-only SQLite authority (WAL) backing rows 2,3,4,6,7,9:
`runs` (identity + digest + status + policy evidence), `run_events` (append-only,
`run_id`, monotonic `seq`, `event_id`, payload), `approval_grants` (challenge digest,
TTL, single-use, CAS version). Mirrors the platform's proven append-only + readiness
discipline (V1.2A) and reuses upstream assets (`_IdempotencyCache.get_or_set`
single-flight, `_make_request_fingerprint` upgraded to canonical JSON).

## 5. Capability negotiation

Reuse `relay/descriptor.py`'s `CONTRACT_VERSION` additive-only pattern: the adapter and
authority negotiate a `DurableRunAuthority` contract version; unknown keys are ignored
for forward/backward compatibility. The probe must be **behavioral** (e.g. can it
persist, can it interrupt, is the approval bus live) — never a static literal.

## 6. Unfakeable acceptance (V2.11 derives from these)

1. Hermes accepted a request but is **killed before responding**; after restart the same
   request identity recovers the **same** Run.
2. Duplicate submit yields **one** Run; same id with a different digest **conflicts** (409).
3. Run status / events / provider policy / evidence **survive restart**.
4. Replay from any cursor has **no gap, no duplicate**; SSE disconnect does **not** delete
   canonical events.
5. Fallback only follows a **pre-authorized chain** and records the reason; unauthorized
   fallback fails explicitly.
6. Approval and stop repeats are **idempotent**; stale/expired/digest-mismatch never
   changes the original fact.

**Any failure ⇒ production adapter reports `unavailable`; dispatch stays OFF.**

## 7. Red lines (unchanged for all of V2)

Zero live effect until V2.11 green + V2.12 separately authorized. No real trading, no
provider/paper/live/broker/Gate/redirect flips, no service start/stop. `chat_write_ready`
/ browser mutation / worker claim-dispatch / public composer stay **OFF**. All V2 work
lands in the isolated worktree; the live checkout is untouched.

## 8. Manual-update compatibility policy

The owner may update the live Hermes checkout/install manually. Updates are never
performed by HQA, a cron job, or an agent. A local deterministic `--no-agent`
compatibility watcher may compare checkout/install/runtime identity with the last
accepted observation and, only after a change, run bounded local-file and loopback
GET-only probes against HQA and `ai-quant-platform`. It must not read provider
credentials, submit a Run, mutate a Session, auto-patch, roll back, restart Hermes,
or enable any write gate.

An identity change is not automatically compatible. Contract drift, an unreachable
authority, or a source/install/runtime mismatch produces a content-addressed report
and keeps write readiness false. The normal post-update gate is the focused contract
suite plus the deterministic probe; Hermes' roughly 40k-test upstream whole-product
suite is reserved for an explicitly chosen high-risk/upstream-wide audit and is not a
routine post-update requirement.
