"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header; opt-in long-term memory scoping via X-Hermes-Session-Key header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id; X-Hermes-Session-Key supported)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent and any configured model_routes aliases
- GET  /v1/capabilities            — machine-readable API capabilities for external UIs
- GET  /api/sessions               — list client-visible Hermes sessions
- POST /api/sessions               — create an empty Hermes session
- GET/PATCH/DELETE /api/sessions/{session_id} — read/update/delete a session
- GET  /api/sessions/{session_id}/messages — read session message history
- POST /api/sessions/{session_id}/fork — branch a session using SessionDB lineage
- POST /api/sessions/{session_id}/chat[/stream] — chat with a persisted session
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}           — retrieve current run status
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- POST /v1/runs/{run_id}/approval — resolve a pending run approval
- POST /v1/runs/{run_id}/stop       — interrupt a running agent
- GET  /health                     — health check
- GET  /health/detailed            — rich status for cross-container dashboard probing

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1 and
authenticating with API_SERVER_KEY.

When ``gateway.multiplex_profiles`` is on, the default profile owns this
listener and secondary profiles are reached via a URL prefix — same contract
as the webhook adapter:

    GET  /p/<profile>/v1/models
    POST /p/<profile>/v1/chat/completions
    ...

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import errno
import hashlib
import hmac
import json
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Sentinel returned by _resolve_request_profile when a /p/<profile>/ prefix
# names a profile this gateway does not serve (→ 404). Distinct from None
# (no prefix / multiplexing off → handle as the default profile).
_PROFILE_REJECTED = object()

# Profile selected by the /p/<profile>/ URL prefix for the current request.
# Set by the profile-prefix middleware; read by handlers / _run_agent.
_api_request_profile: ContextVar[Optional[str]] = ContextVar(
    "api_server_request_profile", default=None
)

def _approval_event_choices(*, smart_denied: bool, allow_permanent: bool) -> list[str]:
    if smart_denied:
        return ["once", "deny"]
    return ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]


try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MEDIA_TAG_CLEANUP_RE,
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
    validate_media_delivery_path,
)
from agent.redact import redact_sensitive_text
from gateway.readiness import collect_runtime_readiness
from utils import is_truthy_value
from gateway.durable_runs import DurableRunStore
from gateway.relay.descriptor import CONTRACT_VERSION as _RELAY_CONTRACT_VERSION

logger = logging.getLogger(__name__)

# Schema version advertised on ``GET /v1/capabilities`` when the durable
# behavioral probe (V2.9) is active. Tracks the relay descriptor's additive-only
# contract version so platform/upstream negotiate the same schema idiom.
RELAY_CONTRACT_VERSION = _RELAY_CONTRACT_VERSION


class DurableEvidenceError(RuntimeError):
    """Canonical run evidence exists but cannot be decoded faithfully."""


def _hermes_version() -> str:
    """Return the hermes-agent version string, or "dev" if it can't be resolved.

    Tries the installed package metadata first (authoritative for a pip/uv
    install), then the in-tree ``hermes_cli.__version__`` (covers editable /
    source checkouts where metadata may be stale or absent). Never raises —
    a version probe must not be able to break the health endpoint.
    """
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:
        pass
    try:
        from hermes_cli import __version__

        return __version__
    except Exception:
        return "dev"


# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 10_000_000  # 10 MB — accommodates long agent conversations with tool calls
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0
MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array


def _coerce_port(value: Any, default: int = DEFAULT_PORT) -> int:
    """Parse a listen port without letting malformed env/config values crash startup."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like API payload values.

    External clients should send real JSON booleans, but some OpenAI-compatible
    frontends and middleware serialize flags like ``stream`` as strings.  Using
    Python truthiness on those values misroutes requests because ``"false"`` is
    still truthy.  Treat only explicit bool-ish scalars as booleans; everything
    else falls back to the caller's default.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        if normalized in _FALSE_REQUEST_BOOL_STRINGS:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string.

    Some clients (Open WebUI, LobeChat, etc.) send content as an array of
    typed parts instead of a plain string::

        [{"type": "text", "text": "hello"}, {"type": "input_text", "text": "..."}]

    This function flattens those into a single string so the agent pipeline
    (which expects strings) doesn't choke.

    Defensive limits prevent abuse: recursion depth, list size, and output
    length are all bounded.
    """
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        total_len = 0
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        for item in items:
            if isinstance(item, str):
                if item:
                    part = item[:MAX_NORMALIZED_TEXT_LENGTH]
                    parts.append(part)
                    total_len += len(part)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            part = str(text)[:MAX_NORMALIZED_TEXT_LENGTH]
                            parts.append(part)
                            total_len += len(part)
                        except Exception:
                            pass
                # Silently skip image_url / other non-text parts
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    parts.append(nested)
                    total_len += len(nested)
            # Check accumulated size
            if total_len >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    # Fallback for unexpected types (int, float, bool, etc.)
    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


# Content part type aliases used by the OpenAI Chat Completions and Responses
# APIs.  We accept both spellings on input and emit a single canonical internal
# shape (``{"type": "text", ...}`` / ``{"type": "image_url", ...}``) that the
# rest of the agent pipeline already understands.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def _normalize_multimodal_content(content: Any) -> Any:
    """Validate and normalize multimodal content for the API server.

    Returns a plain string when the content is text-only, or a list of
    ``{"type": "text"|"image_url", ...}`` parts when images are present.
    The output shape is the native OpenAI Chat Completions vision format,
    which the agent pipeline accepts verbatim (OpenAI-wire providers) or
    converts (``_preprocess_anthropic_content`` for Anthropic).

    Raises ``ValueError`` with an OpenAI-style code on invalid input:
      * ``unsupported_content_type`` — file/input_file/file_id parts, or
        non-image ``data:`` URLs.
      * ``invalid_image_url`` — missing URL or unsupported scheme.
      * ``invalid_content_part`` — malformed text/image objects.

    Callers translate the ValueError into a 400 response.
    """
    # Scalar passthrough mirrors ``_normalize_chat_content``.
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content
    if not isinstance(content, list):
        # Mirror the legacy text-normalizer's fallback so callers that
        # pre-existed image support still get a string back.
        return _normalize_chat_content(content)

    items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
    normalized_parts: List[Dict[str, Any]] = []
    text_accum_len = 0

    for part in items:
        if isinstance(part, str):
            if part:
                trimmed = part[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if not isinstance(part, dict):
            # Ignore unknown scalars for forward compatibility with future
            # Responses API additions (e.g. ``refusal``).  The same policy
            # the text normalizer applies.
            continue

        raw_type = part.get("type")
        part_type = str(raw_type or "").strip().lower()

        if part_type in _TEXT_PART_TYPES:
            text = part.get("text")
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if text:
                trimmed = text[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if part_type in _IMAGE_PART_TYPES:
            detail = part.get("detail")
            image_ref = part.get("image_url")
            # OpenAI Responses sends ``input_image`` with a top-level
            # ``image_url`` string; Chat Completions sends ``image_url`` as
            # ``{"url": "...", "detail": "..."}``.  Support both.
            if isinstance(image_ref, dict):
                url_value = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url_value = image_ref
            if not isinstance(url_value, str) or not url_value.strip():
                raise ValueError("invalid_image_url:Image parts must include a non-empty image URL.")
            url_value = url_value.strip()
            lowered = url_value.lower()
            if lowered.startswith("data:"):
                if not lowered.startswith("data:image/") or "," not in url_value:
                    raise ValueError(
                        "unsupported_content_type:Only image data URLs are supported. "
                        "Non-image data payloads are not supported."
                    )
            elif not (lowered.startswith("http://") or lowered.startswith("https://")):
                raise ValueError(
                    "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs."
                )
            image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
            if detail is not None:
                if not isinstance(detail, str) or not detail.strip():
                    raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
                image_part["image_url"]["detail"] = detail.strip()
            normalized_parts.append(image_part)
            continue

        if part_type in _FILE_PART_TYPES:
            raise ValueError(
                "unsupported_content_type:Inline image inputs are supported, "
                "but uploaded files and document inputs are not supported on this endpoint."
            )

        # Unknown part type — reject explicitly so clients get a clear error
        # instead of a silently dropped turn.
        raise ValueError(
            f"unsupported_content_type:Unsupported content part type {raw_type!r}. "
            "Only text and image_url/input_image parts are supported."
        )

    if not normalized_parts:
        return ""

    # Text-only: collapse to a plain string so downstream logging/trajectory
    # code sees the native shape and prompt caching on text-only turns is
    # unaffected.
    if all(p.get("type") == "text" for p in normalized_parts):
        return "\n".join(p["text"] for p in normalized_parts if p.get("text"))

    return normalized_parts


def _content_has_visible_payload(content: Any) -> bool:
    """True when content has any text or image attachment.  Used to reject empty turns."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip():
                    return True
                if ptype in _IMAGE_PART_TYPES:
                    return True
    return False


def _multimodal_validation_error(exc: ValueError, *, param: str) -> "web.Response":
    """Translate a ``_normalize_multimodal_content`` ValueError into a 400 response."""
    raw = str(exc)
    code, _, message = raw.partition(":")
    if not message:
        code, message = "invalid_content_part", raw
    return web.json_response(
        _openai_error(message, code=code, param=param),
        status=400,
    )


def _session_chat_user_message(body: Dict[str, Any], *, param: str = "message") -> tuple[Any, Optional["web.Response"]]:
    """Parse and normalize session chat ``message`` / ``input`` like chat completions."""
    user_message = body.get("message") or body.get("input")
    if not _content_has_visible_payload(user_message):
        return None, web.json_response(
            _openai_error("Missing 'message' field", code="missing_message"),
            status=400,
        )
    try:
        return _normalize_multimodal_content(user_message), None
    except ValueError as exc:
        return None, _multimodal_validation_error(exc, param=param)


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """
    SQLite-backed LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.

    Persists across gateway restarts.  Falls back to in-memory SQLite
    if the on-disk path is unavailable.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
            except Exception:
                db_path = ":memory:"
        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        # Use shared WAL-fallback helper so response_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same filesystem
        # issue addressed for state.db/kanban.db — see
        # hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="response_store.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )"""
        )
        self._conn.commit()
        # response_store.db contains conversation history (tool payloads,
        # prompts, results). Tighten to owner-only after creation so other
        # local users on a shared box can't read it. Run once at __init__
        # rather than after every commit — chmod-on-every-write is wasted
        # syscalls on a hot path.
        self._tighten_file_permissions()

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars."""
        if not self._db_path:
            return
        for candidate in (
            Path(self._db_path),
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug(
                    "Failed to restrict response store permissions for %s",
                    candidate,
                    exc_info=True,
                )

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id),
        )
        self._conn.commit()
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Corrupted JSON in response store for id=%s, evicting entry",
                response_id,
            )
            self._conn.execute(
                "DELETE FROM responses WHERE response_id = ?",
                (response_id,),
            )
            self._conn.commit()
            return None

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            # Collect IDs that will be evicted
            evict_ids = [
                row[0]
                for row in self._conn.execute(
                    "SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?",
                    (count - self._max_size,),
                ).fetchall()
            ]
            if evict_ids:
                placeholders = ",".join("?" for _ in evict_ids)
                # Clear conversation mappings pointing to evicted responses
                self._conn.execute(
                    f"DELETE FROM conversations WHERE response_id IN ({placeholders})",
                    evict_ids,
                )
                # Delete evicted responses
                self._conn.execute(
                    f"DELETE FROM responses WHERE response_id IN ({placeholders})",
                    evict_ids,
                )
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        # Clear conversation mappings pointing to this response
        self._conn.execute(
            "DELETE FROM conversations WHERE response_id = ?", (response_id,)
        )
        cursor = self._conn.execute(
            "DELETE FROM responses WHERE response_id = ?", (response_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute(
            "SELECT response_id FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            (name, response_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin):
                return web.Response(status=403)
            cors_headers = adapter._cors_headers_for_origin(origin)

        if request.method == "OPTIONS":
            if cors_headers is None:
                return web.Response(status=403)
            return web.Response(status=200, headers=cors_headers)

        response = await handler(request)
        if cors_headers is not None:
            response.headers.update(cors_headers)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


_MEDIA_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MEDIA_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MEDIA_DATA_URL_MAX_BYTES = 5 * 1024 * 1024  # skip images larger than 5MB


def _resolve_media_to_data_urls(text: str) -> str:
    """Replace ``MEDIA:<path>`` image tags with inline base64 data URLs.

    Remote OpenAI-compatible frontends can't read local file paths, so
    ``MEDIA:`` tags referencing images on the server are useless to them.
    Inline small local images as markdown data URLs; non-image or unreadable
    paths are left untouched.

    Uses the same anchored ``MEDIA_TAG_CLEANUP_RE`` matcher and
    ``validate_media_delivery_path`` safety check every other platform
    adapter's media delivery already goes through (gateway/platforms/base.py)
    — an absolute-path anchor plus a known-extension requirement, and a
    resolved-path check against the credential/system-path denylist. The
    prior pattern here matched any bare token after ``MEDIA:`` (including a
    relative/traversal path like ``../../etc/passwd.png``) and read the file
    directly with no denylist, so any image-suffixed, readable file the
    process could see was base64-exfiltrated to the API caller if its path
    merely appeared in the model's own final reply text.
    """
    if not text or "MEDIA:" not in text:
        return text
    import base64

    def _to_data_url(path_str: str) -> Optional[str]:
        # validate_media_delivery_path() strips wrapping quotes/backticks
        # and trailing punctuation internally, same as MEDIA_TAG_CLEANUP_RE's
        # other callers (extract_media / _strip_media_tag_directives) rely on.
        safe_path = validate_media_delivery_path(path_str)
        if not safe_path:
            return None
        p = Path(safe_path)
        suffix = p.suffix.lower()
        if suffix not in _MEDIA_IMG_EXT:
            return None
        try:
            if p.stat().st_size > _MEDIA_DATA_URL_MAX_BYTES:
                return None
            b64 = base64.b64encode(p.read_bytes()).decode()
        except OSError:
            return None
        return f"![image](data:{_MEDIA_MIME[suffix]};base64,{b64})"

    def _repl(m: "re.Match[str]") -> str:
        return _to_data_url(m.group("path")) or m.group(0)

    try:
        return MEDIA_TAG_CLEANUP_RE.sub(_repl, text)
    except Exception:
        return text


def _redact_api_error_text(value: Any, *, limit: int | None = None) -> str:
    """Redact API-bound error text before it crosses the HTTP boundary."""
    redacted = redact_sensitive_text(str(value), force=True)
    if limit is not None:
        return redacted[:limit]
    return redacted


def _openai_error(message: str, err_type: str = "invalid_request_error", param: str = None, code: str = None) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": _redact_api_error_text(message),
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


_api_agent_request_reservation: ContextVar[Optional[dict[str, bool]]] = ContextVar(
    "api_agent_request_reservation", default=None
)


def _admit_api_agent_request(handler):
    """Reserve an authenticated API turn before its handler first awaits.

    Gateway shutdown and aiohttp requests share an event loop. Keeping the
    drain check and reservation in one non-awaiting block prevents a request
    admitted immediately before shutdown from becoming invisible while it is
    still parsing its body or resolving session state. The mutable reservation
    is intentionally shared with child tasks so agent/task bookkeeping releases
    this one slot exactly once.
    """
    @wraps(handler)
    async def _wrapped(self, request, *args, **kwargs):
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        draining = self._draining_response()
        if draining is not None:
            return draining
        reservation = {"active": True}
        token = _api_agent_request_reservation.set(reservation)
        self._pending_agent_requests += 1
        try:
            return await handler(self, request, *args, **kwargs)
        finally:
            if reservation["active"]:
                reservation["active"] = False
                self._pending_agent_requests = max(0, self._pending_agent_requests - 1)
            _api_agent_request_reservation.reset(token)

    return _wrapped


def _release_pending_api_work(adapter, reservation: dict[str, bool]) -> None:
    """Release a pending-work reservation exactly once."""
    if reservation["active"]:
        reservation["active"] = False
        adapter._pending_agent_requests = max(0, adapter._pending_agent_requests - 1)


@contextmanager
def _reserve_pending_api_work(adapter):
    """Keep externally-triggered background work visible across awaits.

    A handler can detach the reservation to an asyncio task; its done callback
    then owns release so shutdown cannot miss the handoff to background work.
    """
    reservation = {"active": True, "detached": False}
    adapter._pending_agent_requests += 1
    try:
        yield reservation
    finally:
        if not reservation["detached"]:
            _release_pending_api_work(adapter, reservation)


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > MAX_REQUEST_BYTES:
                        return web.json_response(_openai_error("Request body too large.", code="body_too_large"), status=413)
                except ValueError:
                    return web.json_response(_openai_error("Invalid Content-Length header.", code="invalid_content_length"), status=400)
        try:
            return await handler(request)
        except web.HTTPRequestEntityTooLarge:
            # aiohttp's client_max_size tripped mid-read (chunked bodies carry
            # no Content-Length) — return a proper 413 instead of letting the
            # handler's broad JSON except turn it into 400 "Invalid JSON".
            return web.json_response(
                _openai_error("Request body too large.", code="body_too_large"),
                status=413,
            )
else:
    body_limit_middleware = None  # type: ignore[assignment]

_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response
else:
    security_headers_middleware = None  # type: ignore[assignment]


class _IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""
    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._inflight: Dict[tuple[str, str], "asyncio.Task[Any]"] = {}
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def get_or_set(self, key: str, fingerprint: str, compute_coro):
        self._purge()
        item = self._store.get(key)
        if item and item["fp"] == fingerprint:
            return item["resp"]

        inflight_key = (key, fingerprint)
        task = self._inflight.get(inflight_key)
        if task is None:
            async def _compute_and_store():
                resp = await compute_coro()
                import time as _t
                self._store[key] = {"resp": resp, "fp": fingerprint, "ts": _t.time()}
                self._purge()
                return resp

            task = asyncio.create_task(_compute_and_store())
            self._inflight[inflight_key] = task

            def _clear_inflight(done_task: "asyncio.Task[Any]") -> None:
                if self._inflight.get(inflight_key) is done_task:
                    self._inflight.pop(inflight_key, None)

            task.add_done_callback(_clear_inflight)

        return await asyncio.shield(task)


_idem_cache = _IdempotencyCache()


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    from hashlib import sha256
    subset = {k: body.get(k) for k in keys}
    return sha256(repr(subset).encode("utf-8")).hexdigest()


def _derive_chat_session_id(
    system_prompt: Optional[str],
    first_user_message: str,
) -> str:
    """Derive a stable session ID from the conversation's first user message.

    OpenAI-compatible frontends (Open WebUI, LibreChat, etc.) send the full
    conversation history with every request.  The system prompt and first user
    message are constant across all turns of the same conversation, so hashing
    them produces a deterministic session ID that lets the API server reuse
    the same Hermes session (and therefore the same Docker container sandbox
    directory) across turns.
    """
    seed = f"{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


_CRON_AVAILABLE = False
try:
    from cron.jobs import (
        list_jobs as _cron_list,
        get_job as _cron_get,
        create_job as _cron_create,
        update_job as _cron_update,
        remove_job as _cron_remove,
        pause_job as _cron_pause,
        resume_job as _cron_resume,
        trigger_job as _cron_trigger,
    )
    _CRON_AVAILABLE = True
except ImportError:
    _cron_list = None
    _cron_get = None
    _cron_create = None
    _cron_update = None
    _cron_remove = None
    _cron_pause = None
    _cron_resume = None
    _cron_trigger = None


def _notify_cron_provider_jobs_changed() -> None:
    """Tell the active cron scheduler provider the job set changed after a REST
    mutation (no-op for the built-in). Best-effort — never breaks the handler."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()
    except Exception:
        pass

# Defense-in-depth: mirror the agent-facing cronjob tool, which scans the
# user-supplied prompt for exfiltration/injection payloads at create/update
# time (tools/cronjob_tools.py).  The REST cron endpoints are authenticated
# (every handler runs _check_auth, and connect() refuses to start without
# API_SERVER_KEY), so this is not the trust boundary — it's parity with the
# tool path so a malicious prompt is rejected the same way regardless of
# which surface created the job.  Imported defensively: a missing scanner
# must not disable the cron REST API.
try:
    from tools.cronjob_tools import _scan_cron_prompt as _scan_cron_prompt
except Exception:  # pragma: no cover - scanner is optional hardening
    _scan_cron_prompt = None


def build_durable_store(
    config: PlatformConfig, *, hermes_home: Optional[Any] = None
) -> Optional[DurableRunStore]:
    """Construct the DurableRunAuthority store when enabled, else ``None``.

    V2.13: the reviewed durable-run semantics (V2.2–V2.9) are dormant unless a
    store is passed to ``APIServerAdapter`` (``_broker_enabled()`` is simply
    ``store is not None``). This is the single, unit-testable construction
    point ``gateway/run.py`` uses, and it is **opt-in / default-OFF** so durable
    persistence does not alter the in-memory execution path unless enabled.

    Enable via ``platforms.api_server.extra.durable_runs_enabled`` (an explicit
    value wins over the env var) or the ``API_SERVER_DURABLE_RUNS`` env var.
    The SQLite DB lives at ``<hermes_home>/durable_runs.db`` so each
    profile/instance (``HERMES_HOME``) gets its own store; override with
    ``extra.durable_runs_db`` or the ``API_SERVER_DURABLE_RUNS_DB`` env var.
    When disabled this is side-effect-free — no store is constructed and no DB
    file is created.
    """
    extra = config.extra or {}
    enabled_raw = extra.get("durable_runs_enabled")
    if enabled_raw is None:
        enabled_raw = os.getenv("API_SERVER_DURABLE_RUNS")
    if not is_truthy_value(enabled_raw, default=False):
        return None
    db_path = extra.get("durable_runs_db") or os.getenv("API_SERVER_DURABLE_RUNS_DB")
    if not db_path:
        home = hermes_home
        if home is None:
            from hermes_cli.config import get_hermes_home

            home = get_hermes_home()
        db_path = str(Path(home) / "durable_runs.db")
    # Fence before opening SQLite: construction enables WAL and may execute
    # CREATE/ALTER, so a losing process must be excluded before the first DB
    # side effect, not merely later during HTTP connect().
    from gateway.durable_runs import DurableAuthorityLock

    authority_lock = DurableAuthorityLock(db_path)
    if not authority_lock.acquire():
        raise RuntimeError(
            "another API server process owns the durable run database"
        )
    try:
        return DurableRunStore(
            db_path=str(db_path), authority_lock=authority_lock
        )
    except Exception:
        authority_lock.release()
        raise


class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    # Stateless request/response: every route (the OpenAI-spec
    # /v1/chat/completions and /v1/responses, and the proprietary /v1/runs SSE
    # stream) tears down its channel when the turn ends. There is no persistent
    # outbound channel to push a background completion to a client that already
    # received its response, and ``send()`` is a no-op stub. So async-delivery
    # tools (terminal notify_on_complete / watch_patterns, delegate_task
    # background=True) must NOT promise delivery on this path — see
    # ``async_delivery_supported()``.
    supports_async_delivery: bool = False

    # Same statelessness applies to the startup auto-resume prompt: no client
    # is waiting to answer "session restored — what next?", so a resumed turn
    # should complete the interrupted work rather than acknowledge (#57056).
    interactive_resume: bool = False

    def __init__(
        self,
        config: PlatformConfig,
        durable_store: Optional[Any] = None,
        *,
        owns_durable_store: bool = False,
    ):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        # Optional DurableRunAuthority store (gateway.durable_runs.DurableRunStore).
        # When None, /v1/runs keeps its legacy in-memory-only behavior; when set,
        # POST /v1/runs honors Idempotency-Key via durable submit-or-get (V2.2).
        self._durable_store = durable_store
        self._owns_durable_store = bool(
            owns_durable_store and durable_store is not None
        )
        self._durable_store_required = durable_store is not None
        # Separate ownership dimensions: the adapter may borrow a store object
        # while still exclusively owning the cross-process mutation fence.
        self._durable_authority_lock: Optional[Any] = getattr(
            durable_store, "authority_lock", None
        )
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        raw_port = extra.get("port")
        if raw_port is None:
            raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
        self._port: int = _coerce_port(raw_port, DEFAULT_PORT)
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        # model_routes: maps incoming ``model`` field values to specific
        # provider/model configs so one API server instance can serve
        # multiple clients on different backends.
        #
        # Config format (platforms.api_server.extra in the gateway config):
        #   model_routes:
        #     minimax-m2:          # alias the client sends as the "model" field
        #       model: "minimax/minimax-m1"
        #       provider: "openrouter"   # optional — resolved via the provider
        #                                # credential chain when set
        #       api_key: "sk-…"          # optional — per-route UPSTREAM provider
        #                                # key override (NOT caller auth; never logged)
        #       base_url: "https://…"    # optional — per-route base URL override
        self._model_routes: Dict[str, Dict[str, Any]] = self._parse_model_routes(
            extra.get("model_routes"),
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        # Runs with a connected SSE consumer; their queue is actively draining.
        self._run_stream_subscribers: set[str] = set()
        # Active run agent/task references for stop support
        self._active_run_agents: Dict[str, Any] = {}
        self._active_run_tasks: Dict[str, "asyncio.Task"] = {}
        # Stop is cooperative: the executor thread may outlive the HTTP request.
        self._stopping_run_ids: set[str] = set()
        # Exact durable stop intent is append-once while a live task remains.
        self._stop_intent_run_ids: set[str] = set()
        # Local execution ended but canonical terminalization failed.  Never
        # expose the surviving queued/running row as trustworthy live status.
        self._run_authority_unavailable: set[str] = set()
        # Pollable run status for dashboards and external control-plane UIs.
        self._run_statuses: Dict[str, Dict[str, Any]] = {}
        # Active approval session key for each run_id.  The approval core
        # resolves requests by session key, while API clients address the
        # in-flight run by run_id.
        self._run_approval_sessions: Dict[str, str] = {}
        # V2.5 durable event-plane broker (only active when a durable_store is
        # configured).  Per run: monotonically increasing seq, per-subscriber
        # fan-out queues, and a terminal flag.  Canonical events live in the
        # durable store; these structures are transport only.
        self._run_event_seq: Dict[str, int] = {}
        self._run_event_subscribers: Dict[str, list] = {}
        self._run_event_terminal: set[str] = set()
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity
        # One provider turn at a time may mutate a managed Hermes Session.
        # SessionDB is thread-safe at the SQL level, but two AIAgent instances
        # replaying the same prefix concurrently would both append divergent
        # continuations.  The event loop owns this map, so claim/check/release
        # need no cross-thread lock.
        self._managed_session_runs: Dict[str, Dict[str, str]] = {}
        # Concurrency cap shared across all agent-serving endpoints
        # (/v1/chat/completions, /v1/responses, /v1/runs). Read from
        # config.yaml gateway.api_server.max_concurrent_runs; 0 disables
        # the cap. Bounds CPU / memory / upstream-LLM-quota exhaustion
        # from a request flood (#7483).
        self._max_concurrent_runs: int = self._resolve_max_concurrent_runs()
        # Number of in-flight runs on the non-streaming chat/responses paths
        # (the /v1/runs path tracks its own in-flight set via
        # _active_run_tasks).
        self._inflight_agent_runs: int = 0
        # Back-reference to the owning GatewayRunner (set by gateway/run.py)
        # so /api/platforms/{platform}/events can resolve sibling adapters.
        # BasePlatformAdapter declares the class-level default of None.
        self.gateway_runner: Optional[Any] = None
        # Requests admitted before their handler reaches agent bookkeeping.
        # Shutdown counts this reservation so the request cannot slip through
        # the drain between its first await and _run_agent()/task registration.
        self._pending_agent_requests: int = 0

    def active_agent_work_count(self) -> int:
        """Return all live agent work owned by this API adapter.

        ``/v1/runs`` registers an asyncio task before it constructs and stores
        its agent, so ``_active_run_agents`` has a real queued-before-agent gap.
        Reuse the task-based accounting used by the concurrent-run limit: it
        covers that gap and excludes completed tasks retained until cleanup.
        """
        try:
            return (
                int(getattr(self, "_pending_agent_requests", 0))
                + int(self._inflight_agent_runs)
                + sum(not task.done() for task in self._active_run_tasks.values())
            )
        except Exception:
            return 0

    @staticmethod
    def _gateway_is_draining() -> bool:
        """Whether the owning gateway currently refuses new agent turns."""
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
            return bool(
                runner
                and (
                    getattr(runner, "_draining", False)
                    or getattr(runner, "_external_drain_active", False)
                )
            )
        except Exception:
            return False

    def _draining_response(self) -> Optional["web.Response"]:
        """Return a retryable response while the gateway drains existing work."""
        if not self._gateway_is_draining():
            return None
        return web.json_response(
            _openai_error(
                "Gateway is draining existing work; retry shortly.",
                code="gateway_draining",
            ),
            status=503,
            headers={"Retry-After": "1"},
        )

    def _activate_admitted_request(self) -> None:
        """Transfer this request's drain reservation to agent bookkeeping."""
        reservation = _api_agent_request_reservation.get()
        if reservation and reservation["active"]:
            reservation["active"] = False
            self._pending_agent_requests = max(0, self._pending_agent_requests - 1)

    def _readiness_work_counts(self) -> tuple[int, int, int]:
        """Return bounded work counts from each subsystem's public state."""
        active_api_runs = sum(
            1
            for status in self._run_statuses.values()
            if status.get("status") in {"queued", "running", "waiting_for_approval"}
        )
        process_depth = 0
        active_delegations = 0
        try:
            from tools.process_registry import process_registry

            process_depth = process_registry.completion_queue.qsize()
        except Exception:
            pass
        try:
            from tools.async_delegation import active_count

            active_delegations = active_count()
        except Exception:
            pass
        return active_api_runs, process_depth, active_delegations

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_max_concurrent_runs() -> int:
        """Read the concurrent-run cap from config.yaml (0 disables).

        gateway.api_server.max_concurrent_runs. Falls back to the historical
        default of 10 when unset or malformed. Negative values are clamped
        to 0 (disabled).
        """
        default = 10
        try:
            from hermes_cli.config import cfg_get, load_config

            raw = cfg_get(
                load_config(),
                "gateway",
                "api_server",
                "max_concurrent_runs",
                default=default,
            )
            value = int(raw)
        except Exception:
            return default
        return max(0, value)

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"
        """
        if explicit and explicit.strip():
            return explicit.strip()
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in {"default", "custom"}:
                return profile
        except Exception:
            pass
        return "hermes-agent"

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    @staticmethod
    def _clean_log_value(value: Any, *, max_len: int = 200) -> str:
        """Sanitize request metadata before it reaches security logs."""
        if value is None:
            return ""
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        return text[:max_len]

    def _request_audit_context(self, request: "web.Request") -> Dict[str, str]:
        """Return non-secret source metadata for security/audit warnings."""
        peer_ip = ""
        try:
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if isinstance(peer, (tuple, list)) and peer:
                peer_ip = str(peer[0])
        except Exception:
            peer_ip = ""

        return {
            "remote": self._clean_log_value(getattr(request, "remote", "") or peer_ip),
            "peer_ip": self._clean_log_value(peer_ip),
            "forwarded_for": self._clean_log_value(request.headers.get("X-Forwarded-For", "")),
            "real_ip": self._clean_log_value(request.headers.get("X-Real-IP", "")),
            "method": self._clean_log_value(request.method, max_len=16),
            "path": self._clean_log_value(request.path_qs, max_len=500),
            "user_agent": self._clean_log_value(request.headers.get("User-Agent", ""), max_len=300),
        }

    def _request_audit_log_suffix(self, request: "web.Request") -> str:
        ctx = self._request_audit_context(request)
        fields = [f"{key}={value!r}" for key, value in ctx.items() if value]
        return " ".join(fields) if fields else "source='unknown'"

    def _cron_origin_from_request(self, request: "web.Request") -> Dict[str, str]:
        """Persist safe API source metadata on cron jobs created over HTTP."""
        ctx = self._request_audit_context(request)
        origin = {
            "platform": "api_server",
            "chat_id": "api",
        }
        if ctx.get("remote"):
            origin["source_ip"] = ctx["remote"]
        if ctx.get("peer_ip"):
            origin["peer_ip"] = ctx["peer_ip"]
        if ctx.get("forwarded_for"):
            origin["forwarded_for"] = ctx["forwarded_for"]
        if ctx.get("real_ip"):
            origin["real_ip"] = ctx["real_ip"]
        if ctx.get("user_agent"):
            origin["user_agent"] = ctx["user_agent"]
        return origin

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        connect() refuses to start the API server without API_SERVER_KEY, so
        the no-key branch only exists for tests or unsupported manual wiring.
        """
        if not self._api_key:
            return None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            # Compare as bytes: ``hmac.compare_digest`` raises TypeError on a
            # str containing non-ASCII characters, and ``token`` is the raw
            # client-supplied header. A stray non-ASCII byte in the key would
            # otherwise crash this handler (500) instead of returning a clean
            # 401. Encoding both sides keeps the timing-safe comparison and
            # matches web_server.py's dashboard-token check.
            if hmac.compare_digest(token.encode(), self._api_key.encode()):
                return None  # Auth OK

        logger.warning(
            "API server rejected invalid API key: %s",
            self._request_audit_log_suffix(request),
        )
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    @staticmethod
    def _normalize_callback_platform(value: str) -> str:
        normalized = (value or "").strip().lower().replace("-", "_")
        if not re.fullmatch(r"[a-z0-9_]+", normalized):
            return ""
        return normalized

    def _get_platform_callback_adapter(
        self,
        request: "web.Request",
        platform_name: str,
    ) -> Optional[Any]:
        injected = request.app.get("platform_event_adapters")
        if isinstance(injected, dict):
            adapter = injected.get(platform_name)
            if adapter is not None:
                return adapter

        adapter = request.app.get(f"{platform_name}_adapter")
        if adapter is not None:
            return adapter

        runner = self.gateway_runner or request.app.get("gateway_runner")
        adapters = getattr(runner, "adapters", None)
        if not adapters:
            return None

        try:
            from gateway.config import Platform as _Platform
            return adapters.get(_Platform(platform_name))
        except Exception:
            for platform, candidate in adapters.items():
                if getattr(platform, "value", platform) == platform_name:
                    return candidate
        return None

    async def _handle_platform_event_callback(self, request: "web.Request") -> "web.Response":
        platform_name = self._normalize_callback_platform(
            request.match_info.get("platform", "")
        )
        if not platform_name:
            return web.json_response(
                _openai_error(
                    "Invalid platform name",
                    code="invalid_platform",
                ),
                status=400,
            )

        adapter = self._get_platform_callback_adapter(request, platform_name)
        if adapter is None:
            return web.json_response(
                _openai_error(
                    "Platform adapter is not connected",
                    code="platform_unavailable",
                ),
                status=503,
            )

        verifier = getattr(adapter, "verify_http_event_request", None)
        dispatcher = getattr(adapter, "dispatch_http_event", None)
        if verifier is None or dispatcher is None:
            return web.json_response(
                _openai_error(
                    "Platform adapter does not support HTTP events",
                    code="platform_http_events_unsupported",
                ),
                status=503,
            )

        auth_header = request.headers.get("Authorization", "")
        try:
            if asyncio.iscoroutinefunction(verifier):
                ok, code = await verifier(auth_header)
            else:
                # Platform verifiers may do blocking network I/O (e.g. Google
                # signing-cert fetches) — keep that off the event loop.
                ok, code = await asyncio.to_thread(verifier, auth_header)
        except Exception:
            # Fail closed: a crashing verifier must never admit the event.
            logger.exception(
                "Platform HTTP event verifier failed for %s", platform_name
            )
            ok, code = False, "platform_event_verifier_error"
        if not ok:
            return web.json_response(
                _openai_error(
                    "Invalid platform event authorization",
                    code=code or "invalid_platform_event_authorization",
                ),
                status=401,
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                _openai_error("Invalid JSON in platform event", code="invalid_json"),
                status=400,
            )

        if not isinstance(payload, dict):
            return web.json_response(
                _openai_error(
                    "Platform event must be a JSON object",
                    code="invalid_request",
                ),
                status=400,
            )

        try:
            result = await dispatcher(payload)
        except Exception:
            logger.exception("Platform HTTP event dispatch failed for %s", platform_name)
            return web.json_response(
                _openai_error(
                    "Platform event dispatch failed",
                    err_type="server_error",
                    code="platform_event_dispatch_failed",
                ),
                status=500,
            )

        return web.json_response(result if isinstance(result, dict) else {})

    # ------------------------------------------------------------------
    # Multi-profile multiplexing (/p/<profile>/…)
    # ------------------------------------------------------------------

    def _resolve_request_profile(self, request: "web.Request"):
        """Resolve + validate the /p/<profile>/ URL prefix on an API request.

        Returns:
          - ``None`` when no profile prefix is present, or multiplexing is off
            (the prefix is ignored; request handled as the default profile).
          - the profile name (str) when present, multiplexing is on, and the
            profile is one this gateway serves.
          - ``_PROFILE_REJECTED`` when a prefix is present but the profile is
            unknown/unconfigured (handler/middleware returns 404).
        """
        profile = (request.match_info.get("profile") or "").strip()
        if not profile:
            return None
        runner = getattr(self, "gateway_runner", None)
        cfg = getattr(runner, "config", None)
        if not getattr(cfg, "multiplex_profiles", False):
            # Prefix supplied but multiplexing is off — ignore it, behave as
            # the single-profile gateway (don't 404 a would-be valid route).
            return None
        try:
            from hermes_cli.profiles import profiles_to_serve

            served = {name for name, _ in profiles_to_serve(multiplex=True)}
        except Exception:
            return _PROFILE_REJECTED
        if profile not in served:
            return _PROFILE_REJECTED
        return profile

    @staticmethod
    def _profile_scope(profile: Optional[str]):
        """Enter the multiplex profile runtime scope, or a no-op when unset.

        When no ``/p/<profile>/`` prefix was given AND multiplexing is active,
        enter the DEFAULT profile's scope instead of a no-op: api_server is a
        port-binding platform that lives on the default profile, and with
        multiplex fail-closed ``get_secret`` active, an unscoped agent run
        raises ``UnscopedSecretError`` on its first credential read (#61276).
        Single-profile gateways keep the no-op — ``get_secret`` falls through
        to ``os.environ`` there, unchanged.
        """
        if not profile:
            try:
                from agent.secret_scope import is_multiplex_active

                if is_multiplex_active():
                    from gateway.run import _profile_runtime_scope
                    from hermes_constants import get_hermes_home

                    return _profile_runtime_scope(get_hermes_home())
            except Exception:
                pass
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        return _profile_runtime_scope(get_profile_dir(profile))

    def _make_profile_prefix_middleware(self):
        """Reject unknown /p/<profile>/ prefixes and scope the request home."""

        @web.middleware
        async def profile_prefix_middleware(request: "web.Request", handler):
            profile = self._resolve_request_profile(request)
            if profile is _PROFILE_REJECTED:
                return web.json_response(
                    {"error": "Unknown or unconfigured profile"},
                    status=404,
                )
            token = _api_request_profile.set(profile)
            try:
                with self._profile_scope(profile):
                    return await handler(request)
            finally:
                _api_request_profile.reset(token)

        return profile_prefix_middleware

    def _http_route_table(self) -> List[tuple]:
        """Return (method, path, handler) rows registered by ``connect()``.

        Kept as a method so multiplex tests can assert the /p/<profile>/
        mirrors without starting a real aiohttp listener.
        """
        routes: List[tuple] = [
            ("GET", "/health", self._handle_health),
            ("GET", "/health/detailed", self._handle_health_detailed),
            ("GET", "/v1/health", self._handle_health),
            ("GET", "/v1/models", self._handle_models),
            ("GET", "/v1/capabilities", self._handle_capabilities),
            ("GET", "/v1/skills", self._handle_skills),
            ("GET", "/v1/toolsets", self._handle_toolsets),
            ("GET", "/api/sessions", self._handle_list_sessions),
            ("POST", "/api/sessions", self._handle_create_session),
            ("GET", "/api/sessions/{session_id}", self._handle_get_session),
            ("PATCH", "/api/sessions/{session_id}", self._handle_patch_session),
            ("DELETE", "/api/sessions/{session_id}", self._handle_delete_session),
            ("GET", "/api/sessions/{session_id}/messages", self._handle_session_messages),
            ("POST", "/api/sessions/{session_id}/fork", self._handle_fork_session),
            ("POST", "/api/sessions/{session_id}/chat", self._handle_session_chat),
            ("POST", "/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream),
            ("POST", "/v1/chat/completions", self._handle_chat_completions),
            ("POST", "/v1/responses", self._handle_responses),
            ("GET", "/v1/responses/{response_id}", self._handle_get_response),
            ("DELETE", "/v1/responses/{response_id}", self._handle_delete_response),
            # Generic platform HTTP event callback ingress. Authenticated by
            # the target adapter's own verifier (platform-signed bearer), NOT
            # API_SERVER_KEY — external platforms hold no API server key.
            ("POST", "/api/platforms/{platform}/events", self._handle_platform_event_callback),
            ("GET", "/api/jobs", self._handle_list_jobs),
            ("POST", "/api/jobs", self._handle_create_job),
            ("GET", "/api/jobs/{job_id}", self._handle_get_job),
            ("PATCH", "/api/jobs/{job_id}", self._handle_update_job),
            ("DELETE", "/api/jobs/{job_id}", self._handle_delete_job),
            ("POST", "/api/jobs/{job_id}/pause", self._handle_pause_job),
            ("POST", "/api/jobs/{job_id}/resume", self._handle_resume_job),
            ("POST", "/api/jobs/{job_id}/run", self._handle_run_job),
            ("POST", "/v1/runs", self._handle_runs),
            ("GET", "/v1/runs/{run_id}", self._handle_get_run),
            ("GET", "/v1/runs/{run_id}/events", self._handle_run_events),
            ("POST", "/v1/runs/{run_id}/approval", self._handle_run_approval),
            ("POST", "/v1/runs/{run_id}/stop", self._handle_stop_run),
        ]
        if _CRON_AVAILABLE:
            # Chronos managed-cron fire webhook (NAS → agent). Authenticated
            # by a NAS-minted JWT (NOT API_SERVER_KEY).
            routes.append(("POST", "/api/cron/fire", self._handle_cron_fire))
        return routes

    # ------------------------------------------------------------------
    # Session header helpers
    # ------------------------------------------------------------------

    # Soft length cap for session identifiers.  Headers are bounded in
    # aggregate by aiohttp (``client_max_size`` / default 8 KiB per
    # header), but we impose a tighter limit on the session headers so a
    # caller can't burn memory by passing a multi-kilobyte "session key".
    # 256 chars is well above any realistic stable channel identifier
    # (e.g. ``agent:main:webui:dm:user-42``) while staying small enough
    # that the sanitized form is safe to pass into Honcho / state.db.
    _MAX_SESSION_HEADER_LEN = 256

    # Soft length cap for the Idempotency-Key header.  Bounded in aggregate by
    # aiohttp, but capped tighter here so a caller can't burn memory / bloat the
    # durable store's primary key with a multi-kilobyte "key".  256 chars is far
    # above any realistic idempotency token (a UUID is 36).
    _MAX_IDEMPOTENCY_KEY_LEN = 256

    def _parse_session_key_header(
        self, request: "web.Request"
    ) -> tuple[Optional[str], Optional["web.Response"]]:
        """Extract and validate the ``X-Hermes-Session-Key`` header.

        The session key is a stable per-channel identifier that scopes
        long-term memory (e.g. Honcho sessions) across transcripts.  It
        is independent of ``X-Hermes-Session-Id``: callers may send
        either, both, or neither.

        Returns ``(session_key, None)`` on success (with an empty/absent
        header yielding ``None`` for the key), or ``(None, error_response)``
        on validation failure.

        Security: like session continuation, accepting a caller-supplied
        memory scope requires API-key authentication so that an
        unauthenticated client on a local-only server can't inject itself
        into another user's long-term memory scope by guessing a key.
        """
        raw = request.headers.get("X-Hermes-Session-Key", "").strip()
        if not raw:
            return None, None

        if not self._api_key:
            logger.warning(
                "X-Hermes-Session-Key rejected: no API key configured. "
                "Set API_SERVER_KEY to enable long-term memory scoping."
            )
            return None, web.json_response(
                _openai_error(
                    "X-Hermes-Session-Key requires API key authentication. "
                    "Configure API_SERVER_KEY to enable this feature."
                ),
                status=403,
            )

        # Reject control characters that could enable header injection on
        # the echo path.
        if re.search(r'[\r\n\x00]', raw):
            return None, web.json_response(
                {"error": {"message": "Invalid session key", "type": "invalid_request_error"}},
                status=400,
            )

        if len(raw) > self._MAX_SESSION_HEADER_LEN:
            return None, web.json_response(
                {"error": {"message": "Session key too long", "type": "invalid_request_error"}},
                status=400,
            )

        return raw, None

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _ensure_session_db(self):
        """Lazily initialise and return the SessionDB for the active profile home.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.

        Under multiplex ``/p/<profile>/`` requests the profile runtime scope
        redirects ``get_hermes_home()``, so each profile gets its own DB —
        never the default profile's file.
        """
        # Explicit override (tests / manual wiring) wins. Production never sets
        # this externally, so the per-home cache below is the live path — and
        # we deliberately do NOT write back into ``self._session_db`` there, or
        # the first profile served would pin every later request to its DB.
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_constants import get_hermes_home
            from hermes_state import SessionDB

            home = get_hermes_home()
            cache = getattr(self, "_session_dbs", None)
            if cache is None:
                cache = {}
                self._session_dbs = cache
            key = str(home)
            db = cache.get(key)
            if db is None:
                db = SessionDB(db_path=home / "state.db")
                cache[key] = db
            return db
        except Exception as e:
            logger.debug("SessionDB unavailable for API server: %s", e)
            return None

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_routes(raw: Any) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize the ``model_routes`` config block.

        Accepts a mapping of ``alias -> {model, provider?, api_key?, base_url?}``.
        Invalid shapes are dropped (never raised) so a config typo can't take
        the whole API server down.  Route values are coerced to strings.

        Security: per-route ``api_key`` values are UPSTREAM provider
        credentials (used to call the routed model's backend), not caller
        authentication — callers still authenticate with the global
        API_SERVER_KEY bearer token via ``_check_auth``.  Route api_keys must
        never be logged; only alias names and non-secret fields may appear in
        logs.
        """
        if not isinstance(raw, dict):
            if raw:
                logger.warning(
                    "api_server model_routes ignored: expected a mapping, got %s",
                    type(raw).__name__,
                )
            return {}

        allowed_keys = ("model", "provider", "api_key", "base_url")
        routes: Dict[str, Dict[str, Any]] = {}
        for alias, cfg in raw.items():
            alias_str = str(alias).strip()
            if not alias_str or not isinstance(cfg, dict):
                logger.warning(
                    "api_server model_routes: dropping invalid route entry %r", alias_str or alias
                )
                continue
            route = {
                key: str(cfg[key]).strip()
                for key in allowed_keys
                if cfg.get(key) is not None and str(cfg[key]).strip()
            }
            if not route.get("model"):
                logger.warning(
                    "api_server model_routes: route %r has no 'model'; dropping", alias_str
                )
                continue
            routes[alias_str] = route
        return routes

    def _resolve_route(self, model_alias: Any) -> Optional[Dict[str, Any]]:
        """Return the model_routes entry for *model_alias*, or None."""
        if not self._model_routes or not isinstance(model_alias, str):
            return None
        return self._model_routes.get(model_alias)

    def _session_model_override_for(self, session_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the gateway's session ``/model`` override for *session_key*, if any.

        The gateway tracks per-session ``/model`` switches in
        ``GatewayRunner._session_model_overrides``.  API-server requests that
        share such a session key must keep honouring the explicit session
        override even when the request's ``model`` field matches a configured
        route — a user-issued ``/model`` always wins over static config.
        """
        if not session_key:
            return None
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
            if runner is None:
                return None
            override = runner._session_model_overrides.get(session_key)
            return dict(override) if isinstance(override, dict) else None
        except Exception:
            return None

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        gateway_session_key: Optional[str] = None,
        route: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.

        ``gateway_session_key`` is a stable per-channel identifier supplied
        by the client (via ``X-Hermes-Session-Key``).  Unlike ``session_id``
        which scopes the short-term transcript and rotates on /new, this
        key is meant to persist across transcripts so long-term memory
        providers (e.g. Honcho) can scope their per-chat state correctly
        — matching the semantics of the native gateway's ``session_key``.

        ``route`` is an optional ``model_routes`` entry (per-client model
        routing).  When set — and no session ``/model`` override exists for
        this session — its model/provider/api_key/base_url override the
        global defaults for this agent instance only.
        """
        from run_agent import AIAgent
        from gateway.run import (
            _current_max_iterations,
            _resolve_runtime_agent_kwargs,
            _resolve_gateway_model,
            _load_gateway_config,
            GatewayRunner,
        )
        from hermes_cli.tools_config import _get_platform_tools

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        reasoning_config = GatewayRunner._load_reasoning_config()
        model = _resolve_gateway_model()

        # When the primary provider's auth fails (expired token / 429 quota
        # cap), _resolve_runtime_agent_kwargs() falls through to the fallback
        # provider chain, whose runtime dict carries its own ``model`` key.
        # Pop it and let it override the config model, mirroring the native
        # gateway path (_resolve_session_agent_runtime in run.py). Otherwise
        # the explicit ``model=model`` below collides with the ``**runtime_kwargs``
        # spread → "got multiple values for keyword argument 'model'", 500ing
        # every /v1/chat/completions request while a fallback is active.
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            model = runtime_model

        # Per-client model routing (model_routes config).  The route was
        # resolved from the request's ``model`` field by the HTTP handler.
        # Precedence (highest first): session ``/model`` override → model_routes
        # route → global config — an explicit user-issued ``/model`` on the
        # session always beats static per-client route config.
        session_override = self._session_model_override_for(
            gateway_session_key or session_id
        )
        if route and not session_override:
            if route.get("provider"):
                # Resolve real credentials for the routed provider (mirrors
                # the channel_overrides path in gateway/run.py) so a route
                # without an explicit api_key/base_url still gets the right
                # provider auth instead of the default provider's key.
                try:
                    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
                    provider_kwargs = _resolve_runtime_agent_kwargs_for_provider(
                        route["provider"]
                    )
                    provider_kwargs.pop("model", None)
                    runtime_kwargs.update(provider_kwargs)
                except Exception:
                    # Fall back to just switching the provider name; explicit
                    # per-route api_key/base_url below can still complete auth.
                    runtime_kwargs["provider"] = route["provider"]
            if route.get("model"):
                model = route["model"]
            # Per-route secrets are upstream provider credentials. Never log
            # them (compare _check_auth: caller auth stays the global bearer
            # key checked with hmac.compare_digest).
            if route.get("api_key"):
                runtime_kwargs["api_key"] = route["api_key"]
            if route.get("base_url"):
                runtime_kwargs["base_url"] = route["base_url"]
            logger.debug(
                "api_server model route applied: model=%s provider=%s",
                model,
                runtime_kwargs.get("provider"),
            )
        elif route and session_override:
            logger.debug(
                "api_server model route skipped: session /model override wins for %s",
                gateway_session_key or session_id,
            )

        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))

        max_iterations = _current_max_iterations()

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        fallback_model = GatewayRunner._load_fallback_model()

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            reasoning_config=reasoning_config,
            gateway_session_key=gateway_session_key,
        )
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response(
            {"status": "ok", "platform": "hermes-agent", "version": _hermes_version()}
        )

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  Requires the same Bearer auth as other API routes.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        from gateway.status import (
            derive_gateway_busy,
            derive_gateway_drainable,
            parse_active_agents,
            read_runtime_status,
        )

        runtime = read_runtime_status() or {}
        gw_state = runtime.get("gateway_state")
        gw_active = parse_active_agents(runtime.get("active_agents", 0))
        # This endpoint is served BY the gateway process, so it is by definition
        # alive — gateway_running is True. Derive busy/drainable from the same
        # shared contract /api/status uses so the two surfaces never disagree.
        active_api_runs, process_depth, active_delegations = self._readiness_work_counts()
        from gateway.run import _resolve_gateway_model

        readiness = collect_runtime_readiness(
            configured_model=_resolve_gateway_model(),
            runtime_status=runtime,
            active_api_runs=active_api_runs,
            process_completion_queue_depth=process_depth,
            active_delegations=active_delegations,
        )
        return web.json_response({
            "status": readiness["status"],
            "readiness": readiness,
            "platform": "hermes-agent",
            "version": _hermes_version(),
            "gateway_state": gw_state,
            "platforms": runtime.get("platforms", {}),
            "active_agents": gw_active,
            "gateway_busy": derive_gateway_busy(
                gateway_running=True,
                gateway_state=gw_state,
                active_agents=gw_active,
            ),
            "gateway_drainable": derive_gateway_drainable(
                gateway_running=True,
                gateway_state=gw_state,
            ),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — list hermes-agent and any configured model_routes aliases.

        Under ``/p/<profile>/v1/models`` (multiplex on) the advertised primary
        model id follows that profile's name/config, not the default adapter's
        cached ``_model_name``.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        now = int(time.time())
        # Middleware already entered the profile runtime scope when a /p/
        # prefix was present, so get_active_profile_name() resolves correctly.
        model_name = (
            self._resolve_model_name("")
            if _api_request_profile.get()
            else self._model_name
        )
        models = [
            {
                "id": model_name,
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": model_name,
                "parent": None,
            }
        ]
        # Expose configured model route aliases so clients can discover them.
        # Only the alias and resolved model name are exposed — never provider
        # credentials.
        for alias, route_cfg in self._model_routes.items():
            if alias == model_name:
                continue  # already listed above
            models.append({
                "id": alias,
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": route_cfg.get("model", alias),
                "parent": model_name,
            })

        return web.json_response({"object": "list", "data": models})

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities — advertise the stable API surface.

        External UIs and orchestrators use this endpoint to discover the API
        server's plugin-safe contract without scraping docs or assuming that
        every Hermes version exposes the same endpoints.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        payload = {
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "model": self._model_name,
            "auth": {
                "type": "bearer",
                "required": bool(self._api_key),
            },
            "runtime": {
                "mode": "server_agent",
                "tool_execution": "server",
                "split_runtime": False,
                "description": (
                    "The API server creates a server-side Hermes AIAgent; "
                    "tools execute on the API-server host unless a future "
                    "explicit split-runtime mode is enabled."
                ),
            },
            "features": {
                "chat_completions": True,
                "chat_completions_streaming": True,
                "responses_api": True,
                "responses_streaming": True,
                "run_submission": True,
                "run_status": True,
                "run_events_sse": True,
                "run_stop": True,
                "run_approval_response": True,
                "tool_progress_events": True,
                "approval_events": True,
                "session_resources": True,
                "session_chat": True,
                "session_chat_streaming": True,
                "session_fork": True,
                "managed_run_sessions": True,
                "managed_run_history_authority": "hermes_session_db",
                "managed_session_fork_mode": "preserve_source_exact_message_cursor",
                "admin_config_rw": False,
                "jobs_admin": False,
                "memory_write_api": False,
                "skills_api": True,
                "audio_api": False,
                "realtime_voice": False,
                "session_continuity_header": "X-Hermes-Session-Id",
                "session_key_header": "X-Hermes-Session-Key",
                "cors": bool(self._cors_origins),
            },
            "endpoints": {
                "health": {"method": "GET", "path": "/health"},
                "health_detailed": {"method": "GET", "path": "/health/detailed"},
                "models": {"method": "GET", "path": "/v1/models"},
                "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                "responses": {"method": "POST", "path": "/v1/responses"},
                "runs": {"method": "POST", "path": "/v1/runs"},
                "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
                "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
                "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
                "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
                "skills": {"method": "GET", "path": "/v1/skills"},
                "toolsets": {"method": "GET", "path": "/v1/toolsets"},
                "sessions": {"method": "GET", "path": "/api/sessions"},
                "session_create": {"method": "POST", "path": "/api/sessions"},
                "session": {"method": "GET", "path": "/api/sessions/{session_id}"},
                "session_update": {"method": "PATCH", "path": "/api/sessions/{session_id}"},
                "session_delete": {"method": "DELETE", "path": "/api/sessions/{session_id}"},
                "session_messages": {"method": "GET", "path": "/api/sessions/{session_id}/messages"},
                "session_fork": {"method": "POST", "path": "/api/sessions/{session_id}/fork"},
                "session_chat": {"method": "POST", "path": "/api/sessions/{session_id}/chat"},
                "session_chat_stream": {"method": "POST", "path": "/api/sessions/{session_id}/chat/stream"},
            },
        }

        # V2.9 (contract row 8): additive behavioral probe. Only when a durable
        # store is configured do we additively surface a contract_version and a
        # `durable` block whose per-capability `grounded` is backed by a real
        # transactional write/read/CAS probe that always rolls back. Legacy
        # (no-store) payload above is byte-identical to before.
        if self._broker_enabled():
            payload["contract_version"] = RELAY_CONTRACT_VERSION
            payload["durable"] = self._capability_probes()

        return web.json_response(payload)

    def _capability_probes(self) -> Dict[str, Dict[str, Any]]:
        """Behavioral capability probe for ``GET /v1/capabilities`` (V2.9).

        Each durable capability is reported as ``{supported, grounded,
        evidence}``:

        * ``supported`` — the code path exists (a durable store is configured).
        * ``grounded`` — the store's transaction probe exercised the capability's
          write/read/CAS behavior and rolled the entire probe transaction back.
        * ``evidence`` — names the transactional grounding signal.

        Fail-closed: any probe that raises degrades that capability to
        ``grounded=False`` rather than claiming support on a stale flag.
        """
        capabilities = (
            "idempotency",
            "event_replay",
            "approval_cas",
            "idempotent_stop",
            "restart_reconcile",
            "run_evidence",
        )
        try:
            grounded = self._durable_store.probe_capabilities()
        except Exception:
            logger.debug("transactional durable capability probe failed", exc_info=True)
            grounded = {}
        return {
            capability: {
                "supported": True,
                "grounded": grounded.get(capability) is True,
                "evidence": f"store.transactional_probe:{capability}",
            }
            for capability in capabilities
        }

    async def _handle_skills(self, request: "web.Request") -> "web.Response":
        """GET /v1/skills — list installed skills visible to the API-server agent.

        Read-only listing intended for external clients that need to know
        which skills are available without sending a chat message and asking
        the model. Mirrors what the gateway/CLI surfaces through
        ``/skills list``, but as a deterministic JSON payload.

        Returns the same skill metadata (name, description, category) the
        skills hub uses internally. Disabled skills are excluded so the
        listing matches what the agent actually loads.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from tools.skills_tool import _find_all_skills, _sort_skills
            skills = _sort_skills(_find_all_skills(skip_disabled=False))
        except Exception:
            logger.exception("GET /v1/skills failed")
            return web.json_response(
                _openai_error("Failed to enumerate skills", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "data": skills,
        })

    async def _handle_toolsets(self, request: "web.Request") -> "web.Response":
        """GET /v1/toolsets — list toolsets and their resolved tools.

        Returns the toolset surface the api_server platform actually exposes
        to its agent: each toolset's enabled/configured state plus the
        concrete tool names it expands to. This is the deterministic
        equivalent of what a client would otherwise have to recover by
        asking the model what tools it can call.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from hermes_cli.config import load_config
            from hermes_cli.tools_config import (
                _get_effective_configurable_toolsets,
                _get_platform_tools,
                _toolset_has_keys,
            )
            from toolsets import resolve_toolset

            config = load_config()
            enabled_toolsets = _get_platform_tools(
                config,
                "api_server",
                include_default_mcp_servers=False,
            )
            data: List[Dict[str, Any]] = []
            for name, label, desc in _get_effective_configurable_toolsets():
                try:
                    tools = sorted(set(resolve_toolset(name)))
                except Exception:
                    tools = []
                is_enabled = name in enabled_toolsets
                data.append({
                    "name": name,
                    "label": label,
                    "description": desc,
                    "enabled": is_enabled,
                    "configured": _toolset_has_keys(name, config),
                    "tools": tools,
                })
        except Exception:
            logger.exception("GET /v1/toolsets failed")
            return web.json_response(
                _openai_error("Failed to enumerate toolsets", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "platform": "api_server",
            "data": data,
        })

    # ------------------------------------------------------------------
    # /api/sessions — thin client/session resource API
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_nonnegative_int(value: Any, default: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed < 0:
            return default
        return min(parsed, maximum)

    @staticmethod
    def _session_response(session: Dict[str, Any]) -> Dict[str, Any]:
        """Return a stable, client-safe session representation."""
        safe_keys = (
            "id", "source", "user_id", "model", "title", "started_at", "ended_at",
            "end_reason", "message_count", "tool_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "api_call_count", "parent_session_id", "last_active", "preview",
            "_lineage_root_id",
        )
        payload = {key: session.get(key) for key in safe_keys if key in session}
        # Avoid exposing full system prompts/model_config through the client API;
        # callers only need to know whether those snapshots exist.
        payload["has_system_prompt"] = bool(session.get("system_prompt"))
        payload["has_model_config"] = bool(session.get("model_config"))
        return payload

    @staticmethod
    def _message_response(message: Dict[str, Any]) -> Dict[str, Any]:
        safe_keys = (
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
            "tool_name", "timestamp", "token_count", "finish_reason", "reasoning",
            "reasoning_content",
        )
        return {key: message.get(key) for key in safe_keys if key in message}

    async def _read_json_body(self, request: "web.Request") -> tuple[Dict[str, Any], Optional["web.Response"]]:
        try:
            body = await request.json()
        except Exception:
            return {}, web.json_response(_openai_error("Invalid JSON in request body"), status=400)
        if not isinstance(body, dict):
            return {}, web.json_response(_openai_error("Request body must be a JSON object"), status=400)
        return body, None

    def _get_existing_session_or_404(self, session_id: str) -> tuple[Optional[Dict[str, Any]], Optional["web.Response"]]:
        db = self._ensure_session_db()
        if db is None:
            return None, web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)
        session = db.get_session(session_id)
        if not session:
            return None, web.json_response(_openai_error(f"Session not found: {session_id}", code="session_not_found"), status=404)
        return session, None

    def _conversation_history_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        db = self._ensure_session_db()
        if db is None:
            return []
        try:
            return db.get_messages_as_conversation(session_id)
        except Exception as exc:
            logger.warning("Failed to load session history for %s: %s", session_id, exc)
            return []

    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions — list persisted Hermes sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        db = self._ensure_session_db()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)

        limit = self._parse_nonnegative_int(request.query.get("limit"), default=50, maximum=200)
        offset = self._parse_nonnegative_int(request.query.get("offset"), default=0, maximum=1_000_000)
        source = request.query.get("source") or None
        include_children = _coerce_request_bool(request.query.get("include_children"), default=False)
        sessions = db.list_sessions_rich(
            source=source,
            limit=limit,
            offset=offset,
            include_children=include_children,
            order_by_last_active=True,
        )
        return web.json_response({
            "object": "list",
            "data": [self._session_response(s) for s in sessions],
            "limit": limit,
            "offset": offset,
            "has_more": len(sessions) == limit,
        })

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions — create an empty Hermes session row."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        body, err = await self._read_json_body(request)
        if err:
            return err

        db = self._ensure_session_db()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)

        raw_id = body.get("id") or body.get("session_id")
        session_id = str(raw_id).strip() if raw_id else f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        from gateway.session import _is_path_unsafe
        if not session_id or re.search(r'[\r\n\x00]', session_id) or _is_path_unsafe(session_id):
            return web.json_response(_openai_error("Invalid session ID", code="invalid_session_id"), status=400)
        if len(session_id) > self._MAX_SESSION_HEADER_LEN:
            return web.json_response(_openai_error("Session ID too long", code="invalid_session_id"), status=400)
        if db.get_session(session_id):
            return web.json_response(_openai_error(f"Session already exists: {session_id}", code="session_exists"), status=409)

        model = body.get("model") or self._model_name
        system_prompt = body.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_prompt must be a string", code="invalid_system_prompt"), status=400)
        db.create_session(session_id, "api_server", model=str(model) if model else None, system_prompt=system_prompt)
        title = body.get("title")
        if title is not None:
            try:
                db.set_session_title(session_id, str(title))
            except ValueError as exc:
                db.delete_session(session_id)
                return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        session = db.get_session(session_id) or {"id": session_id, "source": "api_server", "model": model, "title": title}
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)}, status=201)

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session, err = self._get_existing_session_or_404(request.match_info["session_id"])
        if err:
            return err
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    async def _handle_patch_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} — update client-safe session metadata."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        allowed = {"title", "end_reason"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            return web.json_response(_openai_error(f"Unsupported session fields: {', '.join(unknown)}", code="unsupported_session_field"), status=400)

        db = self._ensure_session_db()
        if "title" in body:
            try:
                db.set_session_title(session_id, "" if body["title"] is None else str(body["title"]))
            except ValueError as exc:
                return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        if body.get("end_reason"):
            db.end_session(session_id, str(body["end_reason"]))
        session = db.get_session(session_id) or session
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = self._ensure_session_db()
        deleted = db.delete_session(session_id)
        return web.json_response({"object": "hermes.session.deleted", "id": session_id, "deleted": bool(deleted)})

    async def _handle_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        _, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = self._ensure_session_db()
        resolved_id = db.resolve_resume_session_id(session_id)
        messages = db.get_messages(resolved_id)
        return web.json_response({
            "object": "list",
            "session_id": resolved_id,
            "data": [self._message_response(m) for m in messages],
        })

    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork — immutable, exact-cursor fork."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        source_id = request.match_info["session_id"]
        source, err = self._get_existing_session_or_404(source_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        db = self._ensure_session_db()

        # A managed fork is a snapshot, not a hand-off: the external/Hermes
        # source remains independently live and byte-for-byte unchanged.
        fork_point = body.get("fork_point")
        if fork_point is None:
            return web.json_response(
                _openai_error(
                    "fork_point is required",
                    code="fork_point_required",
                ),
                status=400,
            )
        if not isinstance(fork_point, str):
            return web.json_response(
                _openai_error(
                    "fork_point must be an exact message:<id> cursor",
                    code="invalid_fork_point",
                ),
                status=400,
            )
        match = re.fullmatch(r"message:([1-9][0-9]*)", fork_point)
        if match is None:
            return web.json_response(
                _openai_error(
                    "fork_point must be an exact message:<id> cursor",
                    code="invalid_fork_point",
                ),
                status=400,
            )
        if body.get("preserve_source") is not True:
            return web.json_response(
                _openai_error(
                    "managed forks require preserve_source=true",
                    code="preserve_source_required",
                ),
                status=400,
            )
        fork_message_id = int(match.group(1))

        from gateway.session import _is_path_unsafe

        fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}").strip()
        if (
            not fork_id
            or re.search(r'[\r\n\x00]', fork_id)
            or _is_path_unsafe(fork_id)
            or len(fork_id) > self._MAX_SESSION_HEADER_LEN
        ):
            return web.json_response(_openai_error("Invalid session ID", code="invalid_session_id"), status=400)
        if db.get_session(fork_id):
            return web.json_response(_openai_error(f"Session already exists: {fork_id}", code="session_exists"), status=409)

        try:
            resolved_source_id = db.resolve_resume_session_id(source_id)
            resolved_source = db.get_session(resolved_source_id)
            if resolved_source is None:
                raise LookupError("resolved source session is missing")
            source_messages = db.get_messages(resolved_source_id)
        except Exception:
            logger.exception(
                "[api_server] managed fork could not read source session %s",
                source_id,
            )
            return web.json_response(
                _openai_error(
                    "Source session history is unavailable",
                    code="session_history_unavailable",
                ),
                status=503,
            )

        prefix: List[Dict[str, Any]] = []
        found_fork_point = False
        for message in source_messages:
            prefix.append(message)
            if message.get("id") == fork_message_id:
                found_fork_point = True
                break
        if not found_fork_point:
            return web.json_response(
                _openai_error(
                    "fork_point does not identify an active source message",
                    code="fork_point_not_found",
                ),
                status=409,
            )

        title = body.get("title")
        if title is None:
            base = source.get("title") or "fork"
            try:
                title = db.get_next_title_in_lineage(base)
            except Exception:
                title = f"{base} fork"
        try:
            # Validate before creating the child so invalid metadata cannot
            # leave a half-created session behind.
            title = db.sanitize_title(str(title))
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)

        try:
            db.create_session(
                fork_id,
                "api_server",
                model=resolved_source.get("model"),
                system_prompt=resolved_source.get("system_prompt"),
                parent_session_id=resolved_source_id,
            )
            db.replace_messages(fork_id, prefix)
            db.set_session_title(fork_id, title)
        except Exception:
            logger.exception(
                "[api_server] managed fork write failed for child %s",
                fork_id,
            )
            try:
                db.delete_session(fork_id)
            except Exception:
                logger.exception(
                    "[api_server] failed to remove partial managed fork %s",
                    fork_id,
                )
            return web.json_response(
                _openai_error(
                    "Managed session fork is unavailable; retry later",
                    code="session_fork_unavailable",
                ),
                status=503,
                headers={"Retry-After": "1"},
            )
        fork = db.get_session(fork_id) or {
            "id": fork_id,
            "parent_session_id": resolved_source_id,
        }
        return web.json_response(
            {
                "object": "hermes.session",
                "session": self._session_response(fork),
                "source_session_id": source_id,
                "resolved_source_session_id": resolved_source_id,
                "fork_point": fork_point,
                "preserve_source": True,
            },
            status=201,
        )

    @_admit_api_agent_request
    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat — one synchronous agent turn."""
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        session_id = request.match_info["session_id"]
        _, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_message must be a string", code="invalid_system_message"), status=400)
        history = self._conversation_history_for_session(session_id)
        result, usage = await self._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            gateway_session_key=gateway_session_key,
        )
        effective_session_id = result.get("session_id") if isinstance(result, dict) else session_id
        final_response = _resolve_media_to_data_urls(result.get("final_response", "") if isinstance(result, dict) else "")
        headers = {"X-Hermes-Session-Id": effective_session_id or session_id}
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        return web.json_response(
            {
                "object": "hermes.session.chat.completion",
                "session_id": effective_session_id or session_id,
                "message": {"role": "assistant", "content": final_response},
                "usage": usage,
            },
            headers=headers,
        )

    @_admit_api_agent_request
    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream — SSE wrapper over _run_agent."""
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        session_id = request.match_info["session_id"]
        _, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_message must be a string", code="invalid_system_message"), status=400)

        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]]" = asyncio.Queue()
        message_id = f"msg_{uuid.uuid4().hex}"
        run_id = f"run_{uuid.uuid4().hex}"
        seq = 0

        def _event_payload(name: str, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            nonlocal seq
            seq += 1
            payload.setdefault("session_id", session_id)
            payload.setdefault("run_id", run_id)
            payload.setdefault("seq", seq)
            payload.setdefault("ts", time.time())
            return name, payload

        def _enqueue(name: str, payload: Dict[str, Any]) -> None:
            event = _event_payload(name, payload)
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            try:
                if running_loop is loop:
                    queue.put_nowait(event)
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass

        def _delta(delta: str) -> None:
            if delta:
                _enqueue("assistant.delta", {"message_id": message_id, "delta": delta})

        def _tool_progress(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs) -> None:
            if event_type == "reasoning.available":
                _enqueue("tool.progress", {"message_id": message_id, "tool_name": tool_name or "_thinking", "delta": preview or ""})
            elif event_type in {"tool.started", "tool.completed", "tool.failed"}:
                event_name = event_type.replace("tool.", "tool.")
                _enqueue(event_name, {"message_id": message_id, "tool_name": tool_name, "preview": preview, "args": args})

        async def _run_and_signal() -> None:
            try:
                await queue.put(_event_payload("run.started", {"user_message": {"role": "user", "content": user_message}}))
                await queue.put(_event_payload("message.started", {"message": {"id": message_id, "role": "assistant"}}))
                history = self._conversation_history_for_session(session_id)
                result, usage = await self._run_agent(
                    user_message=user_message,
                    conversation_history=history,
                    ephemeral_system_prompt=system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_delta,
                    tool_progress_callback=_tool_progress,
                    gateway_session_key=gateway_session_key,
                )
                final_response = _resolve_media_to_data_urls(result.get("final_response", "") if isinstance(result, dict) else "")
                effective_session_id = result.get("session_id", session_id) if isinstance(result, dict) else session_id
                turn_messages = self._turn_transcript_messages(history, user_message, result) if isinstance(result, dict) else []
                await queue.put(_event_payload("assistant.completed", {
                    "session_id": effective_session_id,
                    "message_id": message_id,
                    "content": final_response,
                    "completed": True,
                    "partial": False,
                    "interrupted": False,
                }))
                await queue.put(_event_payload("run.completed", {
                    "session_id": effective_session_id,
                    "message_id": message_id,
                    "completed": True,
                    "messages": turn_messages,
                    "usage": usage,
                }))
            except Exception as exc:
                logger.exception("[api_server] session chat stream failed")
                await queue.put(_event_payload("error", {"message": _redact_api_error_text(exc)}))
            finally:
                await queue.put(_event_payload("done", {}))
                await queue.put(None)

        task = asyncio.create_task(_run_and_signal())
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Hermes-Session-Id": session_id,
        }
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)
        last_write = time.monotonic()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    last_write = time.monotonic()
                    continue
                if item is None:
                    break
                name, payload = item
                data = json.dumps(payload, ensure_ascii=False)
                await response.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
                last_write = time.monotonic()
        except (asyncio.CancelledError, ConnectionResetError):
            task.cancel()
            raise
        except Exception as exc:
            logger.debug("[api_server] session SSE stream error: %s", exc)
        return response

    @_admit_api_agent_request
    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        # Bound total in-flight agent runs (configurable; #7483).
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = _coerce_request_bool(body.get("stream"), default=False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                # System messages don't support images (Anthropic rejects, OpenAI
                # text-model systems don't render them).  Flatten to text.
                content = _normalize_chat_content(raw_content)
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in {"user", "assistant"}:
                try:
                    content = _normalize_multimodal_content(raw_content)
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"messages[{idx}].content")
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message: Any = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not _content_has_visible_payload(user_message):
            return web.json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        # Allow caller to scope long-term memory (e.g. Honcho) with a
        # stable per-channel identifier via X-Hermes-Session-Key.  This
        # is independent of X-Hermes-Session-Id: the key persists across
        # transcripts while the id rotates when the caller starts a new
        # transcript (i.e. /new semantics).  See _parse_session_key_header.
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        #
        # Security: session continuation exposes conversation history, so it is
        # only allowed when the API key is configured and the request is
        # authenticated.  Without this gate, any unauthenticated client could
        # read arbitrary session history by guessing/enumerating session IDs.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity."
                )
                return web.json_response(
                    _openai_error(
                        "Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature."
                    ),
                    status=403,
                )
            # Sanitize: reject control characters that could enable header
            # injection, and path-traversal-shaped IDs that would escape the
            # sessions directory when interpolated into on-disk artifact
            # filenames (session snapshots, request dumps). Mirrors the native
            # gateway's entry-boundary guard (gateway.session._is_path_unsafe).
            from gateway.session import _is_path_unsafe
            if re.search(r'[\r\n\x00]', provided_session_id) or _is_path_unsafe(provided_session_id):
                return web.json_response(
                    {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                    status=400,
                )
            if len(provided_session_id) > self._MAX_SESSION_HEADER_LEN:
                return web.json_response(
                    {"error": {"message": "Session ID too long", "type": "invalid_request_error"}},
                    status=400,
                )
            session_id = provided_session_id
            try:
                db = self._ensure_session_db()
                if db is not None:
                    history = db.get_messages_as_conversation(session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Derive a stable session ID from the conversation fingerprint so
            # that consecutive messages from the same Open WebUI (or similar)
            # conversation map to the same Hermes session.  The first user
            # message + system prompt are constant across all turns.
            first_user = ""
            for cm in conversation_messages:
                if cm.get("role") == "user":
                    first_user = cm.get("content", "")
                    break
            session_id = _derive_chat_session_id(system_prompt, first_user)
            # history already set from request body above

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", self._model_name)
        created = int(time.time())

        # Per-client model routing: if the requested model matches a
        # configured model_routes alias, this request's agent is created
        # with that route's model/provider instead of the global default.
        route = self._resolve_route(model_name)

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # Filter out None — the agent fires stream_delta_callback(None)
                # to signal the CLI display to close its response box before
                # tool execution, but the SSE writer uses None as end-of-stream
                # sentinel.  Forwarding it would prematurely close the HTTP
                # response, causing Open WebUI (and similar frontends) to miss
                # the final answer after tool calls.  The SSE loop detects
                # completion via agent_task.done() instead.
                if delta is not None:
                    _stream_q.put(delta)

            # Track which tool_call_ids we've emitted a "running" lifecycle
            # event for, so a "completed" event without a matching "running"
            # (e.g. internal/filtered tools) is silently dropped instead of
            # producing an orphaned event clients can't correlate.
            _started_tool_call_ids: set[str] = set()

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Emit ``hermes.tool.progress`` with ``status: running``.

                Replaces the old ``tool_progress_callback("tool.started",
                ...)`` emit so SSE consumers receive a single event per
                tool start, carrying both the legacy ``tool``/``emoji``/
                ``label`` payload (for #6972 frontends) and the new
                ``toolCallId``/``status`` correlation fields (#16588).

                Skips tools whose names start with ``_`` so internal
                events (``_thinking``, …) stay off the wire — matching
                the prior ``_on_tool_progress`` filter exactly.
                """
                if not tool_call_id or function_name.startswith("_"):
                    return
                _started_tool_call_ids.add(tool_call_id)
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(function_name, function_args) or function_name
                _stream_q.put(("__tool_progress__", {
                    "tool": function_name,
                    "emoji": get_tool_emoji(function_name),
                    "label": label,
                    "toolCallId": tool_call_id,
                    "status": "running",
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Emit the matching ``status: completed`` event.

                Dropped if the start was filtered (internal tool, missing
                id, or never seen) so clients never get an orphaned
                ``completed`` they can't correlate to a prior ``running``.
                """
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                _stream_q.put(("__tool_progress__", {
                    "tool": function_name,
                    "toolCallId": tool_call_id,
                    "status": "completed",
                }))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            #
            # ``tool_progress_callback`` is intentionally not wired here:
            # it would duplicate every emit because ``run_agent`` fires it
            # side-by-side with ``tool_start_callback``/``tool_complete_callback``.
            # The structured callbacks are strictly richer (they carry
            # the tool_call id), so they own the chat-completions SSE channel.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
                route=route,
            ))
            # Ensure SSE drain loops can terminate without relying on polling
            # agent_task.done(), which can race with queue timeout checks.
            agent_task.add_done_callback(lambda _fut: _stream_q.put(None))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                route=route,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_completion()
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = _resolve_media_to_data_urls(result.get("final_response") or "")
        is_partial = bool(result.get("partial"))
        is_failed = bool(result.get("failed"))
        completed = bool(result.get("completed", True))
        raw_err_msg = result.get("error")
        err_msg = _redact_api_error_text(raw_err_msg) if raw_err_msg else raw_err_msg

        # Decide finish_reason. OpenAI uses "length" for truncation, "stop"
        # for normal completion, and downstream SDKs accept "error" / custom
        # codes. See issue #22496.
        if is_partial and err_msg and "truncat" in err_msg.lower():
            finish_reason = "length"
        elif is_failed or (not completed and err_msg):
            finish_reason = "error"
        else:
            finish_reason = "stop"

        response_headers = {
            "X-Hermes-Session-Id": result.get("session_id", session_id),
        }
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key

        # Hard-fail path: no usable assistant text AND a real failure → 5xx
        # with OpenAI-style error envelope so SDK clients raise instead of
        # silently rendering the internal failure string as message.content.
        if not final_response and (is_failed or is_partial):
            err_body = _openai_error(
                err_msg or "Agent run did not produce a response.",
                err_type="server_error",
                code="agent_incomplete",
            )
            err_body["error"]["hermes"] = {
                "completed": completed,
                "partial": is_partial,
                "failed": is_failed,
            }
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            return web.json_response(err_body, status=502, headers=response_headers)

        # Soft-partial path: we have *some* text but the run did not complete
        # (e.g. truncation with partial buffered output). Still 200 but signal
        # truncation via finish_reason="length" + Hermes-specific extras.
        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        if is_partial or is_failed or not completed:
            response_data["hermes"] = {
                "completed": completed,
                "partial": is_partial,
                "failed": is_failed,
                "error": err_msg,
                "error_code": "output_truncated" if finish_reason == "length" else "agent_error",
            }
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            if err_msg:
                response_headers["X-Hermes-Error"] = _redact_api_error_text(err_msg, limit=200)

        return web.json_response(response_data, headers=response_headers)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
        gateway_session_key: str = None,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue.

        If the client disconnects mid-stream (network drop, browser tab close),
        the agent is interrupted via ``agent.interrupt()`` so it stops making
        LLM API calls, and the asyncio task wrapper is cancelled.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        # CORS middleware can't inject headers into StreamResponse after
        # prepare() flushes them, so resolve CORS headers up front.
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            last_activity = time.monotonic()

            # Role chunk
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
            last_activity = time.monotonic()

            # Helper — route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples ``("__tool_progress__", payload)`` are sent
                as a custom ``event: hermes.tool.progress`` SSE event so
                frontends can display them without storing the markers in
                conversation history.  See #6972 for the original event,
                #16588 for the ``toolCallId``/``status`` lifecycle fields.
                """
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: hermes.tool.progress\ndata: {event_data}\n\n".encode()
                    )
                else:
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                return time.monotonic()

            # Stream content chunks as they arrive from the agent
            loop = asyncio.get_running_loop()
            while True:
                try:
                    delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain any remaining items
                        while True:
                            try:
                                delta = stream_q.get_nowait()
                                if delta is None:
                                    break
                                last_activity = await _emit(delta)
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if delta is None:  # End of stream sentinel
                    break

                last_activity = await _emit(delta)

            # Get usage from completed agent. The agent can fail two ways
            # after the content queue terminates cleanly: (1) ``agent_task``
            # raises, or (2) it returns a ``result`` dict flagged
            # failed/partial/incomplete. Both previously fell through to a
            # ``finish_reason: "stop"`` chunk, so OpenAI-compatible clients
            # saw a fake success. Surface either as a non-"stop" finish so
            # the failure is detectable — mirroring the non-streaming path's
            # decision logic (see the finish_reason block above).
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            result = None
            agent_error = None
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
            except Exception as exc:
                agent_error = exc
                logger.error(
                    "Agent task %s failed during SSE streaming: %s", completion_id, exc
                )

            # Inspect the result dict for a flagged (non-exception) failure.
            is_partial = bool(result.get("partial")) if isinstance(result, dict) else False
            is_failed = bool(result.get("failed")) if isinstance(result, dict) else False
            completed = bool(result.get("completed", True)) if isinstance(result, dict) else True
            err_msg = result.get("error") if isinstance(result, dict) else None
            if agent_error is not None:
                is_failed = True
                err_msg = err_msg or str(agent_error)

            # Decide finish_reason, matching the non-streaming logic: "length"
            # for truncation, "error" for failure, "stop" for normal completion.
            if is_partial and err_msg and "truncat" in err_msg.lower():
                finish_reason = "length"
            elif agent_error is not None or is_failed or (not completed and err_msg):
                finish_reason = "error"
            else:
                finish_reason = "stop"

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            if finish_reason != "stop":
                finish_chunk["choices"][0]["delta"] = {}
                if err_msg:
                    finish_chunk["error"] = {
                        "message": err_msg,
                        "type": type(agent_error).__name__ if agent_error else "agent_error",
                    }
                finish_chunk["hermes"] = {
                    "completed": completed,
                    "partial": is_partial,
                    "failed": is_failed,
                    "error": err_msg,
                    "error_code": "output_truncated" if finish_reason == "length" else "agent_error",
                }
            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected mid-stream.  Interrupt the agent so it
            # stops making LLM API calls at the next loop iteration, then
            # cancel the asyncio task wrapper.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)
        except Exception as _exc:
            # Agent crashed mid-stream.  Try to emit an error chunk
            # so the client gets a proper response instead of a
            # TransferEncodingError from incomplete chunked encoding.
            import traceback as _tb
            logger.error("Agent crashed mid-stream for %s: %s", completion_id, _tb.format_exc()[:300])
            try:
                error_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
                await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
            except Exception:
                pass

        return response

    async def _write_sse_responses(
        self,
        request: "web.Request",
        response_id: str,
        model: str,
        created_at: int,
        stream_q,
        agent_task,
        agent_ref,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        instructions: Optional[str],
        conversation: Optional[str],
        store: bool,
        session_id: str,
        gateway_session_key: Optional[str] = None,
    ) -> "web.StreamResponse":
        """Write an SSE stream for POST /v1/responses (OpenAI Responses API).

        Emits spec-compliant event types as the agent runs:

        - ``response.created`` — initial envelope (status=in_progress)
        - ``response.output_text.delta`` / ``response.output_text.done`` —
          streamed assistant text
        - ``response.output_item.added`` / ``response.output_item.done``
          with ``item.type == "function_call"`` — when the agent invokes a
          tool (both events fire; the ``done`` event carries the finalized
          ``arguments`` string)
        - ``response.output_item.added`` with
          ``item.type == "function_call_output"`` — tool result with
          ``{call_id, output, status}``
        - ``response.completed`` — terminal event carrying the full
          response object with all output items + usage (same payload
          shape as the non-streaming path for parity)
        - ``response.failed`` — terminal event on agent error

        If the client disconnects mid-stream, ``agent.interrupt()`` is
        called so the agent stops issuing upstream LLM calls, then the
        asyncio task is cancelled.  When ``store=True`` an initial
        ``in_progress`` snapshot is persisted immediately after
        ``response.created`` and disconnects update it to an
        ``incomplete`` snapshot so GET /v1/responses/{id} and
        ``previous_response_id`` chaining still have something to
        recover from.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        # State accumulated during the stream
        final_text_parts: List[str] = []
        # Track open function_call items by name so we can emit a matching
        # ``done`` event when the tool completes.  Order preserved.
        pending_tool_calls: List[Dict[str, Any]] = []
        # Output items we've emitted so far (used to build the terminal
        # response.completed payload).  Kept in the order they appeared.
        emitted_items: List[Dict[str, Any]] = []
        # Monotonic counter for output_index (spec requires it).
        output_index = 0
        # Monotonic counter for call_id generation if the agent doesn't
        # provide one (it doesn't, from tool_progress_callback).
        call_counter = 0
        # Canonical Responses SSE events include a monotonically increasing
        # sequence_number. Add it server-side for every emitted event so
        # clients that validate the OpenAI event schema can parse our stream.
        sequence_number = 0
        # Track the assistant message item id + content index for text
        # delta events — the spec ties deltas to a specific item.
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_output_index: Optional[int] = None
        message_opened = False

        async def _write_event(event_type: str, data: Dict[str, Any]) -> None:
            nonlocal sequence_number
            if "sequence_number" not in data:
                data["sequence_number"] = sequence_number
            sequence_number += 1
            payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            await response.write(payload.encode())

        def _envelope(status: str) -> Dict[str, Any]:
            env: Dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "status": status,
                "created_at": created_at,
                "model": model,
            }
            return env

        final_response_text = ""
        agent_error: Optional[str] = None
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        terminal_snapshot_persisted = False

        def _persist_response_snapshot(
            response_env: Dict[str, Any],
            *,
            conversation_history_snapshot: Optional[List[Dict[str, Any]]] = None,
        ) -> None:
            if not store:
                return
            if conversation_history_snapshot is None:
                conversation_history_snapshot = list(conversation_history)
                conversation_history_snapshot.append({"role": "user", "content": user_message})
            self._response_store.put(response_id, {
                "response": response_env,
                "conversation_history": conversation_history_snapshot,
                "instructions": instructions,
                "session_id": session_id,
            })
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        def _persist_incomplete_if_needed() -> None:
            """Persist an ``incomplete`` snapshot if no terminal one was written.

            Called from both the client-disconnect (``ConnectionResetError``)
            and server-cancellation (``asyncio.CancelledError``) paths so
            GET /v1/responses/{id} and ``previous_response_id`` chaining keep
            working after abrupt stream termination.
            """
            if not store or terminal_snapshot_persisted:
                return
            incomplete_text = "".join(final_text_parts) or final_response_text
            incomplete_items: List[Dict[str, Any]] = list(emitted_items)
            if incomplete_text:
                incomplete_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": incomplete_text}],
                })
            incomplete_env = _envelope("incomplete")
            incomplete_env["output"] = incomplete_items
            incomplete_env["usage"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            incomplete_history = list(conversation_history)
            incomplete_history.append({"role": "user", "content": user_message})
            if incomplete_text:
                incomplete_history.append({"role": "assistant", "content": incomplete_text})
            _persist_response_snapshot(
                incomplete_env,
                conversation_history_snapshot=incomplete_history,
            )

        try:
            # response.created — initial envelope, status=in_progress
            created_env = _envelope("in_progress")
            created_env["output"] = []
            await _write_event("response.created", {
                "type": "response.created",
                "response": created_env,
            })
            _persist_response_snapshot(created_env)
            last_activity = time.monotonic()

            async def _open_message_item() -> None:
                """Emit response.output_item.added for the assistant message
                the first time any text delta arrives."""
                nonlocal message_opened, message_output_index, output_index
                if message_opened:
                    return
                message_opened = True
                message_output_index = output_index
                output_index += 1
                item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": item,
                })

            async def _emit_text_delta(delta_text: str) -> None:
                await _open_message_item()
                final_text_parts.append(delta_text)
                await _write_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta_text,
                    "logprobs": [],
                })

            async def _emit_tool_started(payload: Dict[str, Any]) -> str:
                """Emit response.output_item.added for a function_call.

                Returns the call_id so the matching completion event can
                reference it.  Prefer the real ``tool_call_id`` from the
                agent when available; fall back to a generated call id for
                safety in tests or older code paths.
                """
                nonlocal output_index, call_counter
                call_counter += 1
                call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{call_counter}"
                args = payload.get("arguments", {})
                if isinstance(args, dict):
                    arguments_str = json.dumps(args)
                else:
                    arguments_str = str(args)
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": payload.get("name", ""),
                    "call_id": call_id,
                    "arguments": arguments_str,
                }
                idx = output_index
                output_index += 1
                pending_tool_calls.append({
                    "call_id": call_id,
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "item_id": item["id"],
                    "output_index": idx,
                })
                emitted_items.append({
                    "type": "function_call",
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "call_id": call_id,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": item,
                })
                return call_id

            async def _emit_tool_completed(payload: Dict[str, Any]) -> None:
                """Emit response.output_item.done (function_call) followed
                by response.output_item.added (function_call_output)."""
                nonlocal output_index
                call_id = payload.get("tool_call_id")
                result = payload.get("result", "")
                pending = None
                if call_id:
                    for i, p in enumerate(pending_tool_calls):
                        if p["call_id"] == call_id:
                            pending = pending_tool_calls.pop(i)
                            break
                if pending is None:
                    # Completion without a matching start — skip to avoid
                    # emitting orphaned done events.
                    return

                # function_call done
                done_item = {
                    "id": pending["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "name": pending["name"],
                    "call_id": pending["call_id"],
                    "arguments": pending["arguments"],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": pending["output_index"],
                    "item": done_item,
                })

                # function_call_output added (result)
                result_str = result if isinstance(result, str) else json.dumps(result)
                output_parts = [{"type": "input_text", "text": result_str}]
                output_item = {
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                    "status": "completed",
                }
                idx = output_index
                output_index += 1
                emitted_items.append({
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": output_item,
                })
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": output_item,
                })

            # Main drain loop — thread-safe queue fed by agent callbacks.
            async def _dispatch(it) -> None:
                """Route a queue item to the correct SSE emitter.

                Plain strings are text deltas — they are batched (50ms)
                to reduce Open WebUI re-render storms.  Tagged tuples
                with ``__tool_started__`` / ``__tool_completed__``
                prefixes are tool lifecycle events and flush the buffer
                before emitting.
                """
                nonlocal _batch_timer
                if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                    tag, payload = it
                    # Flush batched text before tool events
                    if _batch_buf:
                        await _flush_batch()
                    if tag == "__tool_started__":
                        await _emit_tool_started(payload)
                    elif tag == "__tool_completed__":
                        await _emit_tool_completed(payload)
                elif isinstance(it, str):
                    # Batch text deltas — append to buffer, flush on timer
                    _batch_buf.append(it)
                    if _batch_timer is None:
                        _batch_timer = asyncio.create_task(_batch_flush_after(0.05))
                # Other types are silently dropped.

            # ── Batching state ──
            _batch_buf: List[str] = []
            _batch_timer: Optional[asyncio.Task] = None
            _batch_lock = asyncio.Lock()

            async def _batch_flush_after(delay: float) -> None:
                """Wait delay seconds, then flush accumulated text deltas."""
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
                # Clear timer reference BEFORE flush so new deltas
                # can start a fresh timer while we emit
                nonlocal _batch_buf, _batch_timer
                _batch_timer = None
                await _flush_batch()

            async def _flush_batch() -> None:
                """Emit a single SSE delta for all accumulated text."""
                nonlocal _batch_buf
                async with _batch_lock:
                    if _batch_buf:
                        combined = "".join(_batch_buf)
                        _batch_buf = []
                        await _emit_text_delta(combined)

            loop = asyncio.get_running_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain remaining
                        while True:
                            try:
                                item = stream_q.get_nowait()
                                if item is None:
                                    break
                                await _dispatch(item)
                                last_activity = time.monotonic()
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if item is None:  # EOS sentinel
                    # Cancel pending timer and flush remaining batched text
                    if _batch_timer and not _batch_timer.done():
                        _batch_timer.cancel()
                        _batch_timer = None
                    if _batch_buf:
                        await _flush_batch()
                    break

                await _dispatch(item)
                last_activity = time.monotonic()

            # Flush any final batched text before processing result
            if _batch_buf:
                await _flush_batch()

            # Pick up agent result + usage from the completed task
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                # If the agent produced a final_response but no text
                # deltas were streamed (e.g. some providers only emit
                # the full response at the end), emit a single fallback
                # delta so Responses clients still receive a live text part.
                agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
                if agent_final and not final_text_parts:
                    await _emit_text_delta(agent_final)
                if agent_final and not final_response_text:
                    final_response_text = agent_final
                if isinstance(result, dict) and result.get("error") and not final_response_text:
                    agent_error = _redact_api_error_text(result["error"])
            except Exception as e:  # noqa: BLE001
                logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
                agent_error = _redact_api_error_text(e)

            # Close the message item if it was opened
            final_response_text = "".join(final_text_parts) or final_response_text
            if message_opened:
                await _write_event("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": final_response_text,
                    "logprobs": [],
                })
                msg_done_item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_response_text}
                    ],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": msg_done_item,
                })

            # Always append a final message item in the completed
            # response envelope so clients that only parse the terminal
            # payload still see the assistant text.  This mirrors the
            # shape produced by _extract_output_items in the batch path.
            final_items: List[Dict[str, Any]] = list(emitted_items)

            # Trim large content from tool call arguments to keep the
            # response.completed event under ~100KB.  Clients already
            # received full details via incremental events.
            for _item in final_items:
                if _item.get("type") == "function_call":
                    try:
                        _args = json.loads(_item.get("arguments", "{}")) if isinstance(_item.get("arguments"), str) else _item.get("arguments", {})
                        if isinstance(_args, dict):
                            for _k in ("content", "query", "pattern", "old_string", "new_string"):
                                if isinstance(_args.get(_k), str) and len(_args[_k]) > 500:
                                    _args[_k] = "[" + str(len(_args[_k])) + " chars — truncated for response.completed]"
                            _item["arguments"] = json.dumps(_args)
                    except Exception:
                        pass
                elif _item.get("type") == "function_call_output":
                    _output = _item.get("output", [])
                    if isinstance(_output, list) and _output:
                        _first = _output[0]
                        if isinstance(_first, dict) and _first.get("type") == "input_text":
                            _text = _first.get("text", "")
                            if len(_text) > 1000:
                                _first["text"] = _text[:500] + "...[" + str(len(_text) - 500) + " more chars]"
                                _item["output"] = [_first]

            final_items.append({
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_response_text or (_redact_api_error_text(agent_error) if agent_error else "")}
                ],
            })

            if agent_error:
                failed_env = _envelope("failed")
                failed_env["output"] = final_items
                failed_env["error"] = {"message": _redact_api_error_text(agent_error), "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                _failed_history = list(conversation_history)
                _failed_history.append({"role": "user", "content": user_message})
                if final_response_text or agent_error:
                    _failed_history.append({
                        "role": "assistant",
                        "content": final_response_text or _redact_api_error_text(agent_error),
                    })
                _persist_response_snapshot(
                    failed_env,
                    conversation_history_snapshot=_failed_history,
                )
                terminal_snapshot_persisted = True
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            else:
                completed_env = _envelope("completed")
                completed_env["output"] = final_items
                completed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                full_history = self._build_response_conversation_history(
                    conversation_history,
                    user_message,
                    result,
                    final_response_text,
                )
                _persist_response_snapshot(
                    completed_env,
                    conversation_history_snapshot=full_history,
                )
                terminal_snapshot_persisted = True
                await _write_event("response.completed", {
                    "type": "response.completed",
                    "response": completed_env,
                })

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            _persist_incomplete_if_needed()
            # Client disconnected — interrupt the agent so it stops
            # making upstream LLM calls, then cancel the task.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", response_id)
        except asyncio.CancelledError:
            # Server-side cancellation (e.g. shutdown, request timeout) —
            # persist an incomplete snapshot so GET /v1/responses/{id} and
            # previous_response_id chaining still work, then re-raise so the
            # runtime's cancellation semantics are respected.
            _persist_incomplete_if_needed()
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE task cancelled")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
            logger.info("SSE task cancelled; persisted incomplete snapshot for %s", response_id)
            raise
        except Exception as _exc:
            # Agent crashed with an unhandled error (e.g. model API error like
            # BadRequestError, AuthenticationError).  Emit a response.failed
            # event and properly terminate the SSE stream so the client doesn't
            # get a TransferEncodingError from incomplete chunked encoding.
            import traceback as _tb
            _persist_incomplete_if_needed()
            agent_error = _redact_api_error_text(_tb.format_exc())
            try:
                failed_env = _envelope("failed")
                failed_env["output"] = list(emitted_items)
                failed_env["error"] = {"message": _redact_api_error_text(_exc, limit=500), "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            except Exception:
                pass
            logger.error("Agent crashed mid-stream for %s: %s", response_id, str(agent_error)[:300])

        return response

    @_admit_api_agent_request
    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        # Bound total in-flight agent runs (configurable; #7483).
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited

        # Long-term memory scope header (see chat_completions for details).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = _coerce_request_bool(body.get("store"), default=True)

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._response_store.get_conversation(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, Any]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for idx, item in enumerate(raw_input):
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    try:
                        content = _normalize_multimodal_content(item.get("content", ""))
                    except ValueError as exc:
                        return _multimodal_validation_error(exc, param=f"input[{idx}].content")
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(_openai_error("'input' must be a string or array"), status=400)

        # Accept explicit conversation_history from the request body.
        # This lets stateless clients supply their own history instead of
        # relying on server-side response chaining via previous_response_id.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                try:
                    entry_content = _normalize_multimodal_content(entry["content"])
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
                conversation_history.append({"role": str(entry["role"]), "content": entry_content})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return web.json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
        if not _content_has_visible_payload(user_message):
            return web.json_response(_openai_error("No user message found in input"), status=400)

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # Reuse session from previous_response_id chain so the dashboard
        # groups the entire conversation under one session entry.
        session_id = stored_session_id or str(uuid.uuid4())

        # Per-client model routing for /v1/responses (see model_routes).
        route = self._resolve_route(body.get("model"))

        stream = _coerce_request_bool(body.get("stream"), default=False)
        if stream:
            # Streaming branch — emit OpenAI Responses SSE events as the
            # agent runs so frontends can render text deltas and tool
            # calls in real time.  See _write_sse_responses for details.
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # None from the agent is a CLI box-close signal, not EOS.
                # Forwarding would kill the SSE stream prematurely; the
                # SSE writer detects completion via agent_task.done().
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Queue non-start tool progress events if needed in future.

                The structured Responses stream uses ``tool_start_callback``
                and ``tool_complete_callback`` for exact call-id correlation,
                so progress events are currently ignored here.
                """
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Queue a started tool for live function_call streaming."""
                _stream_q.put(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Queue a completed tool result for live function_call_output streaming."""
                _stream_q.put(("__tool_completed__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                    "result": function_result,
                }))

            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
                route=route,
            ))
            # Ensure SSE drain loops can terminate without relying on polling
            # agent_task.done(), which can race with queue timeout checks.
            agent_task.add_done_callback(lambda _fut: _stream_q.put(None))

            response_id = f"resp_{uuid.uuid4().hex[:28]}"
            model_name = body.get("model", self._model_name)
            created_at = int(time.time())

            return await self._write_sse_responses(
                request=request,
                response_id=response_id,
                model=model_name,
                created_at=created_at,
                stream_q=_stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=conversation_history,
                user_message=user_message,
                instructions=instructions,
                conversation=conversation,
                store=store,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        async def _compute_response():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                route=route,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(
                body,
                keys=["input", "instructions", "previous_response_id", "conversation", "model", "tools"],
            )
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_response)
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_response()
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = _resolve_media_to_data_urls(result.get("final_response", ""))
        if not final_response:
            final_response = _redact_api_error_text(result.get("error", "(No response generated)"))

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())

        # Build the full conversation history for storage
        # (includes tool calls from the agent run)
        full_history = self._build_response_conversation_history(
            conversation_history,
            user_message,
            result,
            final_response,
        )

        # Build output items from the current turn only.  AIAgent returns a
        # full transcript in result["messages"], while older/mocked paths may
        # return only the current turn suffix.
        output_start_index = self._response_messages_turn_start_index(
            conversation_history,
            user_message,
            result,
        )
        output_items = self._extract_output_items(result, start_index=output_start_index)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", self._model_name),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        # Store the complete response object for future chaining / GET retrieval
        if store:
            self._response_store.put(response_id, {
                "response": response_data,
                "conversation_history": full_history,
                "instructions": instructions,
                "session_id": session_id,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        response_headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        return web.json_response(response_data, headers=response_headers)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update — prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    @staticmethod
    def _check_jobs_available() -> Optional["web.Response"]:
        """Return error response if cron module isn't available."""
        if not _CRON_AVAILABLE:
            return web.json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: "web.Request") -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            logger.warning(
                "Cron jobs API rejected invalid job_id %r: %s",
                job_id,
                self._request_audit_log_suffix(request),
            )
            return job_id, web.json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs — list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in {"true", "1"}
            jobs = _cron_list(include_disabled=include_disabled)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs — create a new cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            deliver = body.get("deliver", "local")
            skills = body.get("skills")
            repeat = body.get("repeat")

            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if prompt and _scan_cron_prompt is not None:
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return web.json_response({"error": scan_error}, status=400)
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
                "origin": self._cron_origin_from_request(request),
            }
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = _cron_create(**kwargs)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} — get a single cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_get(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} — update a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return web.json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if sanitized.get("prompt") and _scan_cron_prompt is not None:
                scan_error = _scan_cron_prompt(sanitized["prompt"])
                if scan_error:
                    return web.json_response({"error": scan_error}, status=400)
            job = _cron_update(job_id, sanitized)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} — delete a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            success = _cron_remove(job_id)
            if not success:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause — pause a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_pause(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume — resume a paused cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_resume(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run — trigger immediate execution."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        draining = self._draining_response()
        if draining is not None:
            return draining
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_trigger(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_cron_fire(self, request: "web.Request") -> "web.Response":
        """POST /api/cron/fire — Chronos managed-cron fire webhook (NAS → agent).

        Authenticated by a NAS-minted JWT (verified via the pluggable
        fire-verifier), NOT API_SERVER_KEY — NAS holds no API server key, and
        this is the only inbound that can trigger remote job execution, so it
        gets its own purpose-scoped token check.

        Returns 202 + runs the job in the background so a long agent turn never
        trips NAS's HTTP timeout. The store CAS claim inside fire_due guards
        against double-fire on a NAS/scheduler retry.
        """
        from hermes_cli.config import cfg_get, load_config
        from plugins.cron_providers.chronos.verify import get_fire_verifier

        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""

        cfg = load_config()
        claims = get_fire_verifier()(
            token=token,
            expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
            jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
            issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None,
        )
        if claims is None:
            logger.warning(
                "cron fire: rejected invalid token: %s",
                self._request_audit_log_suffix(request),
            )
            return web.json_response({"error": "invalid fire token"}, status=401)
        draining = self._draining_response()
        if draining is not None:
            return draining

        with _reserve_pending_api_work(self) as reservation:
            try:
                body = await request.json()
            except Exception:
                body = {}
            job_id = (body or {}).get("job_id")
            if not job_id:
                return web.json_response({"error": "missing job_id"}, status=400)

            from cron.scheduler_provider import resolve_cron_scheduler
            provider = resolve_cron_scheduler()

            loop = asyncio.get_running_loop()
            # Fire in the background (202 immediately). fire_due claims via the
            # store CAS, so a retry while this is in flight is de-duped.
            task = asyncio.create_task(
                asyncio.to_thread(provider.fire_due, job_id, adapters=None, loop=loop)
            )
            reservation["detached"] = True
            task.add_done_callback(
                lambda _task: _release_pending_api_work(self, reservation)
            )
            try:
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except (TypeError, AttributeError):
                pass

            return web.json_response({"status": "accepted", "job_id": job_id}, status=202)


    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response_conversation_history(
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
        final_response: Any,
    ) -> List[Dict[str, Any]]:
        """Build the stored Responses transcript without duplicating history."""
        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        agent_messages = result.get("messages") if isinstance(result, dict) else None

        if isinstance(agent_messages, list) and agent_messages:
            turn_start = APIServerAdapter._response_messages_turn_start_index(
                conversation_history,
                user_message,
                result,
            )
            if turn_start:
                return list(agent_messages)

            full_history = prior
            full_history.append(current_user)
            full_history.extend(agent_messages)
            return full_history

        full_history = prior
        full_history.append(current_user)
        full_history.append({"role": "assistant", "content": final_response})
        return full_history

    @staticmethod
    def _response_messages_turn_start_index(
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
    ) -> int:
        """Detect transcript-shaped result["messages"] and return turn start."""
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return 0

        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        expected_prefix = prior + [current_user]
        if agent_messages[:len(expected_prefix)] == expected_prefix:
            return len(expected_prefix)
        if prior and agent_messages[:len(prior)] == prior:
            return len(prior)
        return 0

    @classmethod
    def _turn_transcript_messages(
        cls,
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return this turn's assistant/tool messages in client-safe shape.

        The streaming SSE contract delivers all assistant text as
        ``assistant.delta`` events under one ``message_id`` interleaved with
        ``tool.*`` events, and a single ``assistant.completed`` carrying only
        the final reply.  A client that accumulates deltas into one buffer
        cannot reconstruct *intermediate* assistant text segments that preceded
        tool calls — so when the page is re-opened mid/post-stream those
        segments appear lost, even though state.db persisted them correctly.

        Emitting the authoritative per-turn transcript on ``run.completed`` lets
        any SSE consumer reconcile its live view against ground truth without a
        separate ``GET /messages`` round-trip.  Purely additive: clients that
        ignore the field are unaffected.  Refs #34703.
        """
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return []
        start = cls._response_messages_turn_start_index(
            conversation_history, user_message, result
        )
        turn = agent_messages[start:]
        out: List[Dict[str, Any]] = []
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in {"assistant", "tool"}:
                continue
            out.append(cls._message_response(msg))
        return out

    @staticmethod
    def _extract_output_items(result: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Build the output item array from the agent's messages.

        Walks *result["messages"]* starting at *start_index* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])
        if start_index > 0:
            messages = messages[start_index:]

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = _redact_api_error_text(result.get("error", "(No response generated)"))

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _concurrency_limited_response(self) -> Optional["web.Response"]:
        """Return a 429 response if the concurrent-run cap is reached, else None.

        The cap bounds total in-flight agent activity across every
        agent-serving endpoint. Reuse the same adapter-owned work count that
        shutdown draining uses, including an admitted request before it reaches
        agent/task bookkeeping. Stream queues are transport state and may
        disappear while their underlying run remains active, so they must not
        define run concurrency. A configured value of 0 disables the cap.
        """
        limit = self._max_concurrent_runs
        if limit <= 0:
            return None
        inflight = self.active_agent_work_count()
        # The current request owns one reservation until it hands off to
        # _run_agent() or /v1/runs task registration. It must not consume its
        # own last available slot; other admitted requests remain counted.
        reservation = _api_agent_request_reservation.get()
        if reservation and reservation["active"]:
            inflight -= 1
        if inflight >= limit:
            return web.json_response(
                _openai_error(
                    f"Too many concurrent runs (max {limit})",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                ),
                status=429,
                headers={"Retry-After": "1"},
            )
        return None

    @staticmethod
    def _bind_api_server_session(
        *,
        chat_id: str = "",
        session_key: str = "",
        session_id: str = "",
    ) -> list:
        """Bind session contextvars for an API-server agent run.

        This is the SINGLE structural chokepoint every API-server agent-entry
        path must use to seed session context — it hardwires
        ``platform="api_server"`` and ``async_delivery=False`` so a new route
        physically cannot reintroduce the silent-no-op bug (#10760) by
        forgetting to mark the channel as non-delivering. There is no
        ``async_delivery`` parameter to get wrong; the stateless HTTP path can
        never wake the agent after the turn ends, on ANY route.

        Returns reset tokens; pass them to ``clear_session_vars`` in a
        ``finally`` block (the binding is request-scoped and must not outlive
        the turn — a session resumed later on a delivering interface, e.g. the
        CLI or a gateway platform, re-binds fresh and is NOT blocked).
        """
        from gateway.session_context import set_session_vars

        return set_session_vars(
            platform="api_server",
            chat_id=chat_id,
            session_key=session_key,
            session_id=session_id,
            async_delivery=False,
        )

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        gateway_session_key: Optional[str] = None,
        route: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        *route* is an optional ``model_routes`` entry (resolved from the
        request's ``model`` field) that overrides the global model/provider
        for this specific request.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.
        """
        loop = asyncio.get_running_loop()
        # Capture before hopping to the executor — ContextVars do not follow
        # run_in_executor threads, so the profile scope must be re-entered
        # inside _run() from this explicit value.
        request_profile = _api_request_profile.get()

        def _run():
            from gateway.session_context import clear_session_vars

            with self._profile_scope(request_profile):
                tokens = self._bind_api_server_session(
                    chat_id=session_id or "",
                    session_key=gateway_session_key or session_id or "",
                    session_id=session_id or "",
                )
                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=ephemeral_system_prompt,
                        session_id=session_id,
                        stream_delta_callback=stream_delta_callback,
                        tool_progress_callback=tool_progress_callback,
                        tool_start_callback=tool_start_callback,
                        tool_complete_callback=tool_complete_callback,
                        gateway_session_key=gateway_session_key,
                        route=route,
                    )
                    if agent_ref is not None:
                        agent_ref[0] = agent
                    effective_task_id = session_id or str(uuid.uuid4())
                    result = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id=effective_task_id,
                    )
                    usage = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    # Include the effective session ID in the result so callers
                    # (e.g. X-Hermes-Session-Id header) can track compression-
                    # triggered session rotations. (#16938)
                    _eff_sid = getattr(agent, "session_id", session_id)
                    if isinstance(_eff_sid, str) and _eff_sid:
                        result["session_id"] = _eff_sid
                    return result, usage
                finally:
                    clear_session_vars(tokens)

        self._activate_admitted_request()
        self._inflight_agent_runs += 1
        try:
            return await loop.run_in_executor(None, _run)
        finally:
            self._inflight_agent_runs -= 1

    # ------------------------------------------------------------------
    # /v1/runs — structured event streaming
    # ------------------------------------------------------------------

    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept
    _RUN_STATUS_TTL = 3600  # seconds to retain terminal run status for polling
    _APPROVAL_CHALLENGE_TTL = 300  # seconds an approval challenge stays consumable

    def _set_run_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        """Update pollable run status without exposing private agent objects."""
        # Durable authority is canonical.  Persist/verify it before mutating the
        # in-memory projection so a failed transition can never be reported as
        # running/succeeded to clients while provider work continues.
        self._sync_durable_status(run_id, status)
        return self._update_run_status_memory(run_id, status, **fields)

    def _update_run_status_memory(
        self, run_id: str, status: str, **fields: Any
    ) -> Dict[str, Any]:
        """Project a status after its canonical durable write already succeeded."""
        now = time.time()
        current = dict(self._run_statuses.get(run_id, {}))
        current.update({
            "object": "hermes.run",
            "run_id": run_id,
            "status": status,
            "updated_at": now,
        })
        current.setdefault("created_at", fields.pop("created_at", now))
        current.update(fields)
        self._run_statuses[run_id] = current
        return current

    # ------------------------------------------------------------------
    # /v1/runs — V2.5 durable event-plane broker (contract matrix §2 row 4)
    # ------------------------------------------------------------------
    #
    # These helpers are active only when a durable_store is configured. They
    # give each run event a stable event_id + per-run monotonic seq, persist it
    # to the store (so it survives restart and SSE disconnect), and fan it out
    # to every subscriber's own queue (so concurrent subscribers don't race a
    # single shared queue). The legacy single-queue path is used when no store
    # is configured.

    def _broker_enabled(self) -> bool:
        return self._durable_store is not None

    def _broker_seed(self, run_id: str) -> None:
        """Initialise broker state for a new run (idempotent)."""
        self._run_event_seq.setdefault(run_id, 0)
        self._run_event_subscribers.setdefault(run_id, [])

    def _broker_fanout_record(self, record: Any) -> None:
        """Fan out an already-committed canonical event without re-appending."""
        run_id = str(record.run_id)
        if run_id in self._run_event_terminal:
            return
        self._broker_seed(run_id)
        self._run_event_seq[run_id] = int(record.seq)
        enriched = dict(record.payload)
        enriched["event"] = record.event_type
        enriched["seq"] = record.seq
        enriched["event_id"] = record.event_id
        for queue in list(self._run_event_subscribers.get(run_id, [])):
            try:
                queue.put_nowait(enriched)
            except Exception:
                pass

    def _broker_publish(
        self,
        run_id: str,
        event: Optional[Dict[str, Any]],
        *,
        interrupt_on_failure: bool = True,
    ) -> None:
        """Persist + fan out one event; ``None`` closes every subscriber queue.

        Thread-safe: producers call this from the executor thread via
        ``loop.call_soon_threadsafe``, so it always runs on the event loop.
        """
        if event is None:
            self._run_event_terminal.add(run_id)
            for q in list(self._run_event_subscribers.get(run_id, [])):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            return
        if run_id in self._run_event_terminal:
            logger.warning(
                "[api_server] rejected post-terminal event for run %s", run_id
            )
            return
        self._broker_seed(run_id)
        event_type = str(event.get("event", "event"))
        # Persist first: no subscriber may observe an event that lacks a
        # canonical durable identity.
        try:
            rec = self._durable_store.append_event(run_id, event_type, dict(event))
        except Exception as exc:
            self._run_authority_unavailable.add(run_id)
            logger.exception("[api_server] durable append_event failed for run %s", run_id)
            agent = self._active_run_agents.get(run_id)
            if interrupt_on_failure and agent is not None:
                try:
                    agent.interrupt(
                        "Durable run authority failed while recording an event; "
                        "the run is stopping fail-closed."
                    )
                except Exception:
                    logger.debug(
                        "[api_server] could not interrupt run after durable failure",
                        exc_info=True,
                    )
            raise RuntimeError("durable event persistence failed") from exc
        self._broker_fanout_record(rec)

    def _broker_subscribe(self, run_id: str) -> "asyncio.Queue":
        """Attach a new subscriber queue for a live run."""
        self._broker_seed(run_id)
        q: "asyncio.Queue" = asyncio.Queue()
        self._run_event_subscribers[run_id].append(q)
        return q

    def _broker_unsubscribe(self, run_id: str, q: "asyncio.Queue") -> None:
        subs = self._run_event_subscribers.get(run_id)
        if subs is not None:
            try:
                subs.remove(q)
            except ValueError:
                pass

    def _broker_is_terminal(self, run_id: str) -> bool:
        """True when the run will never emit more live events.

        Live map uses upstream names (completed/failed/cancelled). After a
        process restart those entries are gone and only the durable store
        remains, under contract names (succeeded/failed/stopped). SSE replay
        must treat both as terminal so a post-restart subscriber gets the
        backlog and an immediate stream-close instead of hanging on a live
        fan-out that no longer exists.
        """
        if run_id in self._run_event_terminal:
            return True
        status = self._run_statuses.get(run_id, {})
        if status.get("status") in {"completed", "failed", "cancelled"}:
            return True
        if self._broker_enabled() and getattr(self, "_durable_store", None) is not None:
            try:
                row = self._durable_store.get_run(run_id)
            except Exception:
                row = None
            if row is not None and row.get("status") in {
                "succeeded", "failed", "stopped",
                # Tolerate upstream names if ever persisted raw.
                "completed", "cancelled",
            }:
                return True
        return False

    # Upstream run-status name -> contract RunState name (contract matrix §1).
    # waiting_for_approval / stopping are sub-states of running, never top-level.
    _STATUS_TO_RUNSTATE = {
        "queued": "queued",
        "running": "running",
        "waiting_for_approval": "running",
        "stopping": "running",
        "completed": "succeeded",
        "failed": "failed",
        "cancelled": "stopped",
    }
    _PUBLIC_RUN_STATES = frozenset(
        {"queued", "running", "succeeded", "failed", "stopped"}
    )

    @classmethod
    def _public_run_status(cls, status: Dict[str, Any]) -> Dict[str, Any]:
        """Project internal lifecycle names onto the stable five-state contract."""
        projected = dict(status)
        raw_status = str(projected.get("status") or "")
        canonical = cls._STATUS_TO_RUNSTATE.get(raw_status)
        if canonical is None and raw_status in cls._PUBLIC_RUN_STATES:
            canonical = raw_status
        if canonical is None:
            raise ValueError("run status is outside the public five-state contract")
        projected["status"] = canonical
        if raw_status == "waiting_for_approval":
            projected["substate"] = "waiting_for_approval"
        elif raw_status == "stopping":
            projected["substate"] = "stopping"
        else:
            projected.pop("substate", None)
        return projected

    @staticmethod
    def _run_state_invalid_response(run_id: str) -> "web.Response":
        """Fail closed when an internal status cannot be projected faithfully."""
        logger.error(
            "[api_server] refusing to expose unrecognized run status for %s", run_id
        )
        return web.json_response(
            _openai_error(
                "Run state is unavailable because its lifecycle value is invalid",
                code="run_state_invalid",
            ),
            status=503,
        )

    @staticmethod
    def _durable_unavailable_response() -> "web.Response":
        """Return the stable fail-closed response for durable authority errors."""
        return web.json_response(
            _openai_error(
                "Durable run authority is unavailable; retry later",
                code="durable_unavailable",
            ),
            status=503,
            headers={"Retry-After": "1"},
        )

    def _stop_unadmitted_durable_run(self, run_id: str) -> None:
        """Best-effort terminal cleanup for a run rejected before task creation."""
        try:
            from gateway.durable_runs import RunState

            self._durable_store.finalize_run(
                run_id,
                terminal_state=RunState.STOPPED,
                event_type="run.stopped",
                event_payload={
                    "event": "run.stopped",
                    "run_id": run_id,
                    "reason": "admission_failed",
                    "timestamp": time.time(),
                },
            )
        except Exception:
            logger.exception(
                "[api_server] could not stop rejected durable run %s", run_id
            )

    def _sync_durable_status(self, run_id: str, upstream_status: str) -> None:
        """V2.8 (contract row 7): mirror the live status into the durable store.

        Maps the upstream status name to the contract RunState and applies it via
        a guarded transition.  Same-state writes are verified idempotently;
        unknown/missing/illegal transitions raise and stop provider admission.
        """
        if not self._broker_enabled():
            return
        contract = self._STATUS_TO_RUNSTATE.get(upstream_status)
        if contract is None:
            raise RuntimeError(f"durable status mapping is unknown: {upstream_status}")
        from gateway.durable_runs import RunState

        row = self._durable_store.get_run(run_id)
        if row is None:
            raise RuntimeError(f"durable status row is missing for {run_id}")
        if str(row.get("status")) == contract:
            return
        moved = self._durable_store.transition(run_id, RunState(contract))
        if moved:
            return
        verified = self._durable_store.get_run(run_id)
        if verified is None or str(verified.get("status")) != contract:
            raise RuntimeError(
                f"durable status transition failed for {run_id}: "
                f"{row.get('status')} -> {contract}"
            )

    def reconcile_durable_runs(self) -> int:
        """V2.8 (contract row 7): reconcile phantom in-flight runs on startup.

        After a restart, runs left in queued/running in the store have no live
        task — they are phantom-running. Reconcile them to the deterministic
        terminal state ``stopped``. Idempotent and terminal-safe (terminal runs
        are never rewritten). Returns the number of runs reconciled.
        """
        if not self._broker_enabled():
            return 0
        from gateway.durable_runs import RunState

        reconciled = 0
        for row in self._durable_store.list_non_terminal_runs():
            event = {
                "event": "run.stopped",
                "run_id": row["run_id"],
                "reason": "startup_reconcile",
                "timestamp": time.time(),
            }
            record = self._durable_store.finalize_run(
                row["run_id"],
                terminal_state=RunState.STOPPED,
                event_type="run.stopped",
                event_payload=event,
            )
            if record is not None:
                reconciled += 1
        if reconciled:
            logger.info(
                "[api_server] reconciled %d phantom run(s) to stopped on startup",
                reconciled,
            )
        return reconciled

    @staticmethod
    def _sse_frame(event: Dict[str, Any]) -> bytes:
        """One spec-compliant SSE frame: id + event + data."""
        seq = event.get("seq")
        event_type = event.get("event", "message")
        head = f"id: {seq}\nevent: {event_type}\n" if seq is not None else f"event: {event_type}\n"
        return f"{head}data: {json.dumps(event)}\n\n".encode()

    def _run_outcome_evidence(
        self,
        *,
        agent: Optional[Any] = None,
        provider_response_receipt: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Build actual policy only from a real provider response receipt."""
        if agent is None or not isinstance(provider_response_receipt, dict):
            return None, None
        receipt_model = provider_response_receipt.get("model")
        receipt_provider = provider_response_receipt.get("provider")
        actual_model = (
            receipt_model
            if isinstance(receipt_model, str) and receipt_model
            else None
        )
        actual_provider = (
            receipt_provider
            if isinstance(receipt_provider, str) and receipt_provider
            else None
        )
        fallback_activated = (
            getattr(agent, "_fallback_activated", False) is True
            and actual_model is not None
        )
        actual_policy: Dict[str, Any] = {}
        if actual_model is not None:
            actual_policy["model"] = actual_model
        if actual_provider is not None:
            actual_policy["provider"] = actual_provider

        fallback_reason: Optional[str] = None
        if fallback_activated:
            fallback_reason = f"fallback_activated:{actual_model}"
            if actual_provider:
                fallback_reason = (
                    f"fallback_activated:{actual_model} via {actual_provider}"
                )
        return actual_policy or None, fallback_reason

    def _finalize_durable_run(
        self,
        run_id: str,
        *,
        upstream_status: str,
        event: Dict[str, Any],
        outcome_usage: Optional[Dict[str, Any]],
        agent: Optional[Any] = None,
        provider_response_receipt: Optional[Dict[str, Any]] = None,
        **status_fields: Any,
    ) -> None:
        """Atomically finalize canonical truth, then project/fan out it."""
        from gateway.durable_runs import RunState

        contract = self._STATUS_TO_RUNSTATE[upstream_status]
        actual_policy, fallback_reason = self._run_outcome_evidence(
            agent=agent,
            provider_response_receipt=provider_response_receipt,
        )
        try:
            record = self._durable_store.finalize_run(
                run_id,
                terminal_state=RunState(contract),
                event_type=str(event["event"]),
                event_payload=dict(event),
                actual_policy=actual_policy,
                fallback_reason=fallback_reason,
                usage=outcome_usage,
            )
        except Exception:
            self._run_authority_unavailable.add(run_id)
            raise
        self._run_authority_unavailable.discard(run_id)
        if record is not None:
            self._broker_fanout_record(record)
        self._update_run_status_memory(
            run_id, upstream_status, **status_fields
        )
        self._broker_publish(run_id, None)
        self._run_streams.pop(run_id, None)
        self._run_streams_created.pop(run_id, None)

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            self._set_run_status(
                run_id,
                self._run_statuses.get(run_id, {}).get("status", "running"),
                last_event=event.get("event"),
            )
            if self._broker_enabled():
                def _publish() -> None:
                    try:
                        self._broker_publish(run_id, event)
                    except RuntimeError:
                        # _broker_publish already interrupted the run. Avoid an
                        # unhandled event-loop callback while the executor unwinds.
                        pass

                loop.call_soon_threadsafe(_publish)
                return
            q = self._run_streams.get(run_id)
            if q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

        def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            ts = time.time()
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "preview": preview,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                })
            # _thinking and subagent_progress are intentionally not forwarded

        return _callback

    def _resolve_managed_run_session(
        self,
        requested_session_id: Any,
    ) -> tuple[Optional[str], List[Dict[str, Any]], Optional["web.Response"]]:
        """Resolve one caller-supplied managed Session to its durable live tip.

        Hermes, not the platform client, owns transcript recovery.  A supplied
        Session id must already exist in SessionDB; compression continuations
        resolve to their live tip and replay includes the full root-to-tip
        lineage.  Every read failure is fail-closed before a Run is allocated.
        """
        from gateway.session import _is_path_unsafe

        if (
            not isinstance(requested_session_id, str)
            or not requested_session_id
            or len(requested_session_id) > self._MAX_SESSION_HEADER_LEN
            or re.search(r"[\r\n\x00]", requested_session_id)
            or _is_path_unsafe(requested_session_id)
        ):
            return None, [], web.json_response(
                _openai_error(
                    "Invalid managed session ID",
                    code="invalid_session_id",
                ),
                status=400,
            )

        db = self._ensure_session_db()
        if db is None:
            return None, [], web.json_response(
                _openai_error(
                    "Session database unavailable",
                    code="session_db_unavailable",
                ),
                status=503,
            )
        try:
            source = db.get_session(requested_session_id)
        except Exception:
            logger.exception(
                "[api_server] managed session lookup failed for %s",
                requested_session_id,
            )
            return None, [], web.json_response(
                _openai_error(
                    "Session database unavailable",
                    code="session_db_unavailable",
                ),
                status=503,
            )
        if source is None:
            return None, [], web.json_response(
                _openai_error(
                    f"Session not found: {requested_session_id}",
                    code="session_not_found",
                ),
                status=404,
            )

        try:
            resolved_session_id = db.resolve_resume_session_id(
                requested_session_id
            )
            resolved = db.get_session(resolved_session_id)
            if resolved is None:
                raise LookupError("resolved session is missing")
            if resolved.get("ended_at") is not None:
                return None, [], web.json_response(
                    _openai_error(
                        "Session is ended; fork it before continuing",
                        code="session_ended",
                    ),
                    status=409,
                )
            history = db.get_messages_as_conversation(
                resolved_session_id,
                include_ancestors=True,
                repair_alternation=True,
            )
        except Exception:
            logger.exception(
                "[api_server] managed session history recovery failed for %s",
                requested_session_id,
            )
            return None, [], web.json_response(
                _openai_error(
                    "Session history is unavailable",
                    code="session_history_unavailable",
                ),
                status=503,
            )
        return resolved_session_id, history, None

    @staticmethod
    def _managed_idempotency_store_key(
        managed_session_scope: str,
        client_key: str,
    ) -> str:
        """Scope a client action key to one managed Session without leaking ids."""
        scope = hashlib.sha256(
            managed_session_scope.encode("utf-8")
        ).hexdigest()
        return f"managed:{scope}:{client_key}"

    @staticmethod
    def _managed_session_scope(session_id: str) -> str:
        """Profile-qualified key for process-local managed Session ownership."""
        profile = _api_request_profile.get() or "default"
        return f"{profile}\x1f{session_id}"

    @staticmethod
    def _submission_headers(
        *,
        session_id: str,
        gateway_session_key: Optional[str],
    ) -> Dict[str, str]:
        headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        return headers

    def _durable_submission_replay_response(
        self,
        *,
        run_id: str,
        fallback_session_id: str,
        idempotency_key: str,
        gateway_session_key: Optional[str],
    ) -> "web.Response":
        """Return one canonical existing Run without creating provider work."""
        try:
            stored_run = self._durable_store.get_run(run_id)
        except Exception:
            logger.exception(
                "[api_server] durable replay lookup failed for %s",
                run_id,
            )
            return self._durable_unavailable_response()
        if stored_run is None:
            return self._durable_unavailable_response()
        replay_session_id = str(
            stored_run.get("session_id") or fallback_session_id or run_id
        )
        try:
            replay_body = self._public_run_status(
                {
                    "run_id": run_id,
                    "status": stored_run.get("status"),
                    "session_id": replay_session_id,
                    "created": False,
                    "idempotent_replay": True,
                    "idempotency_key": idempotency_key,
                }
            )
        except ValueError:
            return self._run_state_invalid_response(run_id)
        return web.json_response(
            replay_body,
            status=202,
            headers=self._submission_headers(
                session_id=replay_session_id,
                gateway_session_key=gateway_session_key,
            ),
        )

    def _managed_session_busy_response(
        self,
        *,
        managed_session_scope: str,
        session_id: str,
        idempotency_key: Optional[str],
        request_body: Dict[str, Any],
        gateway_session_key: Optional[str],
    ) -> Optional["web.Response"]:
        """Recover an in-flight identical submit or reject a divergent turn."""
        claim = self._managed_session_runs.get(managed_session_scope)
        if claim is None:
            return None
        if (
            self._broker_enabled()
            and idempotency_key is not None
            and claim.get("idempotency_key") == idempotency_key
        ):
            if (
                claim.get("request_digest")
                != self._request_digest_for_busy_check(request_body)
            ):
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used with a different request",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            return self._durable_submission_replay_response(
                run_id=claim["run_id"],
                fallback_session_id=session_id,
                idempotency_key=idempotency_key,
                gateway_session_key=gateway_session_key,
            )
        return web.json_response(
            _openai_error(
                "Another run is already mutating this managed session",
                code="session_busy",
            ),
            status=409,
            headers={"Retry-After": "1"},
        )

    @staticmethod
    def _request_digest_for_busy_check(request_body: Dict[str, Any]) -> str:
        from gateway.durable_runs import canonical_digest

        return canonical_digest(request_body)

    @_admit_api_agent_request
    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs — start an agent run, return run_id immediately."""
        # Long-term memory scope header (see chat_completions for details).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_input = body.get("input")
        if not raw_input:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        user_message = raw_input if isinstance(raw_input, str) else (raw_input[-1].get("content", "") if isinstance(raw_input, list) else "")
        if not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        requested_managed_session_id = (
            body.get("session_id") if "session_id" in body else None
        )

        # A caller-supplied Session id selects the managed contract: Hermes
        # loads its own canonical transcript.  Supplying a second transcript
        # authority alongside it would let a platform counterfeit context, so
        # reject that shape rather than choosing ambiguous precedence.
        conversation_history: List[Dict[str, Any]] = []
        stored_session_id = None
        if requested_managed_session_id is not None:
            if (
                "conversation_history" in body
                or previous_response_id is not None
                or (isinstance(raw_input, list) and len(raw_input) > 1)
            ):
                return web.json_response(
                    _openai_error(
                        "Hermes owns history for managed sessions; submit only the new input",
                        code="managed_session_history_owned_by_hermes",
                    ),
                    status=400,
                )
            (
                resolved_session_id,
                conversation_history,
                session_error,
            ) = self._resolve_managed_run_session(
                requested_managed_session_id
            )
            if session_error is not None:
                return session_error
        else:
            # Legacy/stateless clients may still provide explicit history or
            # previous_response_id.  This path does not claim managed-session
            # continuity and remains backwards compatible.
            raw_history = body.get("conversation_history")
            if raw_history:
                if not isinstance(raw_history, list):
                    return web.json_response(
                        _openai_error(
                            "'conversation_history' must be an array of message objects"
                        ),
                        status=400,
                    )
                for i, entry in enumerate(raw_history):
                    if (
                        not isinstance(entry, dict)
                        or "role" not in entry
                        or "content" not in entry
                    ):
                        return web.json_response(
                            _openai_error(
                                f"conversation_history[{i}] must have 'role' and 'content' fields"
                            ),
                            status=400,
                        )
                    conversation_history.append(
                        {
                            "role": str(entry["role"]),
                            "content": str(entry["content"]),
                        }
                    )
                if previous_response_id:
                    logger.debug(
                        "Both conversation_history and previous_response_id provided; using conversation_history"
                    )

            if not conversation_history and previous_response_id:
                stored = self._response_store.get(previous_response_id)
                if stored:
                    conversation_history = list(
                        stored.get("conversation_history", [])
                    )
                    stored_session_id = stored.get("session_id")
                    if instructions is None:
                        instructions = stored.get("instructions")

            # When input is a multi-message array, extract all but the last
            # message as conversation history (the last becomes user_message).
            if (
                not conversation_history
                and isinstance(raw_input, list)
                and len(raw_input) > 1
            ):
                for msg in raw_input[:-1]:
                    if (
                        isinstance(msg, dict)
                        and msg.get("role")
                        and msg.get("content")
                    ):
                        content = msg["content"]
                        if isinstance(content, list):
                            content = " ".join(
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict)
                                and part.get("type") == "text"
                            )
                        conversation_history.append(
                            {
                                "role": msg["role"],
                                "content": str(content),
                            }
                        )

            resolved_session_id = stored_session_id

        # V2.2 (contract matrix §2 rows 1/2): honor the caller's Idempotency-Key
        # with a durable submit-or-get against the canonical request digest.
        # Same identity (key + same semantic body) ⇒ recover the SAME Run without
        # re-executing; same key with a different digest ⇒ 409 (fail closed). When
        # no key is supplied, or no durable store is configured, a fresh run is
        # still minted below.
        idempotency_key = request.headers.get("Idempotency-Key") or None
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip() or None
        if idempotency_key is not None and len(idempotency_key) > self._MAX_IDEMPOTENCY_KEY_LEN:
            return web.json_response(
                _openai_error(
                    "Idempotency-Key exceeds maximum length",
                    code="idempotency_key_too_long",
                ),
                status=400,
            )

        managed_session = requested_managed_session_id is not None
        managed_session_scope = (
            self._managed_session_scope(str(resolved_session_id))
            if managed_session
            else None
        )
        if managed_session:
            busy_response = self._managed_session_busy_response(
                managed_session_scope=str(managed_session_scope),
                session_id=str(resolved_session_id),
                idempotency_key=idempotency_key,
                request_body=body,
                gateway_session_key=gateway_session_key,
            )
            if busy_response is not None:
                return busy_response

        idempotency_store_key = idempotency_key
        if managed_session and idempotency_key is not None:
            idempotency_store_key = self._managed_idempotency_store_key(
                self._managed_session_scope(
                    str(requested_managed_session_id)
                ),
                idempotency_key,
            )

        # Read-only exact recovery precedes provider concurrency admission.  A
        # completed Run replay remains available while another Session occupies
        # the last provider slot; it neither allocates a row nor starts work.
        if idempotency_key is not None and self._durable_store is not None:
            from gateway.durable_runs import ConflictError

            try:
                existing_submit = self._durable_store.find_submission(
                    idempotency_key=str(idempotency_store_key),
                    request_body=body,
                )
            except ConflictError:
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used with a different request",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            except Exception:
                logger.exception(
                    "[api_server] durable submission preflight failed"
                )
                return self._durable_unavailable_response()
            if existing_submit is not None:
                return self._durable_submission_replay_response(
                    run_id=existing_submit.run_id,
                    fallback_session_id=str(
                        resolved_session_id or existing_submit.run_id
                    ),
                    idempotency_key=idempotency_key,
                    gateway_session_key=gateway_session_key,
                )

        # Only a genuinely new provider turn consumes concurrency.  Idempotent
        # recovery returned above is control-plane read work, not agent work.
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited

        if idempotency_key is not None and self._durable_store is not None:
            from gateway.durable_runs import ConflictError

            try:
                submit = self._durable_store.submit_or_get(
                    idempotency_key=str(idempotency_store_key),
                    request_body=body,
                    session_id=resolved_session_id,
                )
            except ConflictError:
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used with a different request",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            except Exception:
                logger.exception("[api_server] durable submit_or_get failed")
                return web.json_response(
                    _openai_error(
                        "Durable run authority is unavailable; retry later",
                        code="durable_unavailable",
                    ),
                    status=503,
                    headers={"Retry-After": "1"},
                )
            if not submit.created:
                # Recovery: return the existing Run; never re-spawn it.
                return self._durable_submission_replay_response(
                    run_id=submit.run_id,
                    fallback_session_id=str(
                        resolved_session_id or submit.run_id
                    ),
                    idempotency_key=idempotency_key,
                    gateway_session_key=gateway_session_key,
                )
            run_id = submit.run_id
            session_id = resolved_session_id or run_id
            submission_created = True
        else:
            run_id = f"run_{uuid.uuid4().hex}"
            session_id = resolved_session_id or run_id
            submission_created = True
        # Ensure a durable run row exists whenever the broker is active so
        # events/approvals have a parent (contract row 9). Keyed submissions were
        # already persisted by submit_or_get; this registers server-minted runs.
        # Durable mode is fail-closed: no agent may start without its canonical
        # run row. Keyed submit_or_get already created that row; server-minted
        # runs must register it here before any transport/task state exists.
        if self._broker_enabled():
            try:
                if idempotency_key is None:
                    self._durable_store.register_run(
                        run_id=run_id,
                        session_id=session_id,
                        request_body=body,
                    )
            except Exception:
                logger.exception("[api_server] durable register_run failed for %s", run_id)
                return web.json_response(
                    _openai_error(
                        "Durable run authority is unavailable; retry later",
                        code="durable_unavailable",
                    ),
                    status=503,
                    headers={"Retry-After": "1"},
                )

        # Resolve the requested route and persist it before creating any live
        # task/transport state. A run without immutable requested-policy
        # evidence is not admitted to provider execution.
        route = self._resolve_route(body.get("model"))
        if self._broker_enabled():
            requested_policy: Dict[str, Any] = {}
            if body.get("model") is not None:
                requested_policy["model"] = body.get("model")
            if route:
                requested_policy["route"] = {
                    key: route[key]
                    for key in ("model", "provider")
                    if route.get(key) is not None
                }
            try:
                policy_recorded = self._durable_store.set_requested_policy(
                    run_id, requested_policy
                )
            except Exception:
                logger.exception(
                    "[api_server] set_requested_policy failed for %s", run_id
                )
                self._stop_unadmitted_durable_run(run_id)
                return self._durable_unavailable_response()
            if not policy_recorded:
                logger.error(
                    "[api_server] requested policy was not accepted for %s", run_id
                )
                self._stop_unadmitted_durable_run(run_id)
                return self._durable_unavailable_response()
            try:
                self._broker_seed(run_id)
            except Exception:
                logger.exception("[api_server] durable broker seed failed for %s", run_id)
                self._stop_unadmitted_durable_run(run_id)
                return self._durable_unavailable_response()

        if managed_session:
            self._managed_session_runs[str(managed_session_scope)] = {
                "run_id": run_id,
                "idempotency_key": idempotency_key or "",
                "request_digest": self._request_digest_for_busy_check(body),
            }

        # Approval queues gate host-side tool execution and must be isolated
        # per API run.  Managed Session turns are serialized above; approval
        # identity is still the Run because control decisions must never leak
        # between distinct runs or non-managed clients sharing another scope.
        approval_session_key = run_id
        ephemeral_system_prompt = instructions
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
        created_at = time.time()
        self._run_streams[run_id] = q
        self._run_streams_created[run_id] = created_at
        self._run_approval_sessions[run_id] = approval_session_key

        event_cb = self._make_run_event_callback(run_id, loop)

        def _put_event_if_active(event: Optional[Dict]) -> None:
            """Enqueue only while this run still owns live transport state."""
            if self._broker_enabled():
                # Broker path: publish is the single funnel for every event and
                # the close sentinel. Guard against post-close stragglers.
                if event is None or self._run_streams.get(run_id) is q:
                    self._broker_publish(run_id, event)
                return
            if self._run_streams.get(run_id) is q:
                q.put_nowait(event)

        def _put_callback_event_if_active(event: Dict) -> None:
            try:
                _put_event_if_active(event)
            except RuntimeError:
                # Persistence failure already interrupts the active agent.
                pass

        # Also wire stream_delta_callback so message.delta events flow through.
        def _text_cb(delta: Optional[str]) -> None:
            if delta is None:
                return
            if run_id not in self._run_streams:
                return
            try:
                loop.call_soon_threadsafe(_put_callback_event_if_active, {
                    "event": "message.delta",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "delta": delta,
                })
            except Exception:
                pass

        try:
            self._set_run_status(
                run_id,
                "queued",
                created_at=created_at,
                session_id=session_id,
                model=body.get("model", self._model_name),
            )
        except Exception:
            if not self._broker_enabled():
                raise
            logger.exception(
                "[api_server] initial durable status failed for %s", run_id
            )
            self._stop_unadmitted_durable_run(run_id)
            # No task exists yet. Remove every live/transport projection so a
            # rejected submission cannot remain as a phantom queued run.
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)
            self._run_approval_sessions.pop(run_id, None)
            self._run_statuses.pop(run_id, None)
            self._run_event_seq.pop(run_id, None)
            self._run_event_subscribers.pop(run_id, None)
            self._run_event_terminal.discard(run_id)
            self._active_run_agents.pop(run_id, None)
            self._active_run_tasks.pop(run_id, None)
            self._stopping_run_ids.discard(run_id)
            self._stop_intent_run_ids.discard(run_id)
            claim = self._managed_session_runs.get(str(managed_session_scope))
            if claim is not None and claim.get("run_id") == run_id:
                self._managed_session_runs.pop(
                    str(managed_session_scope),
                    None,
                )
            return self._durable_unavailable_response()

        # Background task outlives the HTTP response (and thus the middleware
        # profile scope). Capture now and re-enter inside the task/executor.
        request_profile = _api_request_profile.get()
        async def _run_and_close():
            provider_evidence: Dict[str, Optional[Dict[str, Any]]] = {
                "attempt": None,
                "response": None,
            }

            def _finish(
                upstream_status: str,
                event: Dict[str, Any],
                *,
                final_usage: Optional[Dict[str, Any]],
                final_agent: Optional[Any],
                **status_fields: Any,
            ) -> None:
                if self._broker_enabled():
                    self._finalize_durable_run(
                        run_id,
                        upstream_status=upstream_status,
                        event=event,
                        outcome_usage=final_usage,
                        agent=final_agent,
                        provider_response_receipt=provider_evidence["response"],
                        **status_fields,
                    )
                else:
                    _put_event_if_active(event)
                    self._set_run_status(
                        run_id, upstream_status, **status_fields
                    )

            try:
                self._set_run_status(run_id, "running")
                if run_id in self._stopping_run_ids:
                    cancelled = {
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    }
                    _finish(
                        "cancelled",
                        cancelled,
                        final_usage=None,
                        final_agent=None,
                        last_event="run.cancelled",
                    )
                    return
                with self._profile_scope(request_profile):
                    agent = self._create_agent(
                        ephemeral_system_prompt=ephemeral_system_prompt,
                        session_id=session_id,
                        stream_delta_callback=_text_cb,
                        tool_progress_callback=event_cb,
                        gateway_session_key=gateway_session_key,
                        route=route,
                    )
                self._active_run_agents[run_id] = agent

                def _record_provider_attempt(attempt: Dict[str, Any]) -> None:
                    # A new attempt invalidates any older response receipt. If
                    # this attempt fails before a response, public actual-policy
                    # evidence must remain empty rather than naming a stale
                    # primary/fallback route.
                    provider_evidence["attempt"] = dict(attempt)
                    provider_evidence["response"] = None

                def _record_provider_response(receipt: Dict[str, Any]) -> None:
                    provider_evidence["response"] = dict(receipt)

                agent._provider_attempt_callback = _record_provider_attempt
                agent._provider_response_callback = _record_provider_response

                # Stop can be accepted while agent construction is in flight,
                # after the earlier admission check but before provider work.
                # Recheck after publishing the live agent ref and immediately
                # before run_conversation so an accepted stop wins this window.
                if run_id in self._stopping_run_ids:
                    try:
                        agent.interrupt("Stop requested before provider dispatch")
                    except Exception:
                        pass
                    cancelled = {
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    }
                    _finish(
                        "cancelled",
                        cancelled,
                        final_usage=None,
                        final_agent=agent,
                        last_event="run.cancelled",
                    )
                    return

                def _approval_notify(approval_data: Dict[str, Any]) -> None:
                    event = dict(approval_data or {})
                    # Final API egress redaction covers every human-facing text
                    # field, including MCP elicitation descriptions/previews.
                    from gateway.run import _redact_approval_command

                    redacted_command = _redact_approval_command(event.get("command"))
                    event["command"] = redacted_command
                    for field in ("description", "preview", "target"):
                        if field in event:
                            event[field] = redact_sensitive_text(
                                str(event[field]), force=True
                            )
                    # Guard-pattern keys are server-internal authorization
                    # identifiers.  The action digest binds them inside the
                    # approval core; Web/SSE clients neither need nor receive
                    # their raw values.
                    event.pop("pattern_key", None)
                    event.pop("pattern_keys", None)
                    # V2.7 (contract row 6): issue an exact durable challenge bound
                    # to this run + a canonical digest of the gated action. The
                    # client must echo challenge_id + action_digest back to consume
                    # (single-use + TTL + CAS). Broker-only; legacy path unchanged.
                    if self._broker_enabled():
                        approval_id = str(event.get("approval_id") or "")
                        action_digest = str(event.get("action_digest") or "")
                        try:
                            if not approval_id or not action_digest:
                                raise ValueError(
                                    "approval core did not provide exact approval identity"
                                )
                            ch = self._durable_store.issue_approval_challenge(
                                run_id,
                                approval_id=approval_id,
                                action_digest=action_digest,
                                ttl_seconds=self._APPROVAL_CHALLENGE_TTL,
                            )
                            event["challenge_id"] = ch.challenge_id
                            event["approval_id"] = ch.approval_id
                            event["action_digest"] = action_digest
                            event["expires_at"] = ch.expires_at
                        except Exception as exc:
                            logger.exception(
                                "[api_server] issue_approval_challenge failed for %s", run_id
                            )
                            # Propagate into _await_gateway_decision: it removes
                            # the queue entry and returns notify_failed/deny.  An
                            # unanswerable request must never be published or
                            # move the run into waiting_for_approval.
                            raise RuntimeError(
                                "durable approval challenge unavailable"
                            ) from exc
                    event.update({
                        "event": "approval.request",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "choices": _approval_event_choices(
                            smart_denied=bool(event.get("smart_denied")),
                            allow_permanent=event.get("allow_permanent") is not False,
                        ),
                    })
                    self._set_run_status(
                        run_id,
                        "waiting_for_approval",
                        last_event="approval.request",
                    )
                    try:
                        if self._broker_enabled():
                            loop.call_soon_threadsafe(self._broker_publish, run_id, event)
                        else:
                            loop.call_soon_threadsafe(q.put_nowait, event)
                    except Exception:
                        pass

                def _run_sync():
                    from gateway.session_context import clear_session_vars
                    from tools.approval import (
                        register_gateway_notify,
                        reset_current_session_key,
                        set_current_session_key,
                        unregister_gateway_notify,
                    )

                    effective_task_id = session_id or run_id
                    approval_token = None
                    session_tokens = []
                    with self._profile_scope(request_profile):
                        try:
                            # Bind approval/session identity for this API run via
                            # contextvars so concurrent runs do not share process
                            # environment state.
                            approval_token = set_current_session_key(approval_session_key)
                            session_tokens = self._bind_api_server_session(
                                session_key=approval_session_key,
                            )
                            register_gateway_notify(approval_session_key, _approval_notify)
                            if run_id in self._stopping_run_ids:
                                return {
                                    "final_response": "",
                                    "stopped_before_provider_dispatch": True,
                                }, None
                            r = agent.run_conversation(
                                user_message=user_message,
                                conversation_history=conversation_history,
                                task_id=effective_task_id,
                            )
                        finally:
                            try:
                                unregister_gateway_notify(approval_session_key)
                            finally:
                                if approval_token is not None:
                                    try:
                                        reset_current_session_key(approval_token)
                                    except Exception:
                                        pass
                                if session_tokens:
                                    try:
                                        clear_session_vars(session_tokens)
                                    except Exception:
                                        pass
                        if provider_evidence["response"] is None:
                            u = None
                        else:
                            u = {
                                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                            }
                        return r, u

                result, usage = await asyncio.get_running_loop().run_in_executor(
                    None, _run_sync
                )
                if run_id in self._stopping_run_ids:
                    cancelled = {
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    }
                    _finish(
                        "cancelled",
                        cancelled,
                        final_usage=usage,
                        final_agent=agent,
                        last_event="run.cancelled",
                    )
                # Check for structured failure (non-retryable client errors like
                # 401/400 return failed=True instead of raising, so the except
                # block below never fires — issue #15561).
                elif isinstance(result, dict) and result.get("failed"):
                    error_msg = _redact_api_error_text(result.get("error") or "agent run failed")
                    failed_event = {
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": error_msg,
                    }
                    _finish(
                        "failed",
                        failed_event,
                        final_usage=usage,
                        final_agent=agent,
                        error=error_msg,
                        last_event="run.failed",
                    )
                else:
                    final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                    completed_event = {
                        "event": "run.completed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "output": final_response,
                        "usage": usage,
                    }
                    _finish(
                        "completed",
                        completed_event,
                        final_usage=usage,
                        final_agent=agent,
                        output=final_response,
                        usage=usage,
                        last_event="run.completed",
                    )
            except asyncio.CancelledError:
                try:
                    cancelled = {
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    }
                    _finish(
                        "cancelled",
                        cancelled,
                        final_usage=None,
                        final_agent=locals().get("agent"),
                        last_event="run.cancelled",
                    )
                except Exception:
                    logger.exception(
                        "[api_server] cancelled run %s could not finalize", run_id
                    )
                raise
            except Exception as exc:
                logger.exception("[api_server] run %s failed", run_id)
                try:
                    error_text = _redact_api_error_text(exc)
                    failed_event = {
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": error_text,
                    }
                    _finish(
                        "failed",
                        failed_event,
                        final_usage=None,
                        final_agent=locals().get("agent"),
                        error=error_text,
                        last_event="run.failed",
                    )
                except Exception:
                    logger.exception(
                        "[api_server] failed run %s could not finalize", run_id
                    )
            finally:
                # If the asyncio wrapper is cancelled (for example via
                # /stop), the executor thread can still be blocked waiting
                # on an approval Event.  Unregistering here releases those
                # waits immediately; the in-thread unregister is harmlessly
                # idempotent on normal completion.
                try:
                    from tools.approval import unregister_gateway_notify

                    unregister_gateway_notify(approval_session_key)
                except Exception:
                    pass
                # Sentinel: signal SSE stream to close
                try:
                    _put_event_if_active(None)
                except Exception:
                    pass
                # Durable subscribers replay from canonical storage, so their
                # live ownership can be dropped immediately.  Preserve the
                # legacy in-memory terminal queue until its SSE consumer/TTL
                # sweep observes the sentinel.
                if self._broker_enabled():
                    self._run_streams.pop(run_id, None)
                    self._run_streams_created.pop(run_id, None)
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)
                self._stopping_run_ids.discard(run_id)
                self._stop_intent_run_ids.discard(run_id)
                claim = self._managed_session_runs.get(
                    str(managed_session_scope)
                )
                if claim is not None and claim.get("run_id") == run_id:
                    self._managed_session_runs.pop(
                        str(managed_session_scope),
                        None,
                    )

        self._activate_admitted_request()
        task = asyncio.create_task(_run_and_close())
        self._active_run_tasks[run_id] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "run_id": run_id,
                "session_id": session_id,
                "created": submission_created,
                "status": "queued",
            },
            status=202,
            headers=self._submission_headers(
                session_id=str(session_id),
                gateway_session_key=gateway_session_key,
            ),
        )

    async def _handle_get_run(self, request: "web.Request") -> "web.Response":
        """GET /v1/runs/{run_id} — return pollable run status for external UIs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        status = self._run_statuses.get(run_id)
        if run_id in self._run_authority_unavailable:
            if not self._broker_enabled():
                self._run_authority_unavailable.discard(run_id)
            else:
                try:
                    recovery_row = self._durable_store.get_run(run_id)
                except Exception:
                    return self._durable_unavailable_response()
                if recovery_row is None or str(recovery_row.get("status")) not in {
                    "succeeded", "failed", "stopped"
                }:
                    return self._durable_unavailable_response()
                self._run_authority_unavailable.discard(run_id)
        if status is None:
            # V2.6: fall back to the durable store so evidence survives restart.
            if self._broker_enabled():
                try:
                    row = self._durable_store.get_run(run_id)
                except Exception:
                    logger.exception(
                        "[api_server] durable run lookup failed for %s", run_id
                    )
                    return self._durable_unavailable_response()
                if row is not None:
                    try:
                        projected = self._public_run_status(
                            self._run_status_from_store(row)
                        )
                    except DurableEvidenceError:
                        logger.exception(
                            "[api_server] durable evidence is corrupt for %s", run_id
                        )
                        return self._durable_unavailable_response()
                    except ValueError:
                        return self._run_state_invalid_response(run_id)
                    return web.json_response(projected)
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )
        if self._broker_enabled():
            # Merge durable evidence (requested/actual/usage/fallback) into the
            # in-memory status without mutating the live dict.
            try:
                row = self._durable_store.get_run(run_id)
            except Exception:
                logger.exception(
                    "[api_server] durable evidence lookup failed for %s", run_id
                )
                return self._durable_unavailable_response()
            if row is None:
                return self._durable_unavailable_response()
            # Canonical durable lifecycle wins over a stale live projection;
            # retain only additive live fields such as output/error.
            try:
                status = {**status, **self._run_status_from_store(row)}
            except DurableEvidenceError:
                logger.exception(
                    "[api_server] durable evidence is corrupt for %s", run_id
                )
                return self._durable_unavailable_response()
        try:
            projected = self._public_run_status(status)
        except ValueError:
            return self._run_state_invalid_response(run_id)
        return web.json_response(projected)

    @staticmethod
    def _evidence_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Project durable runs-row evidence columns onto the status response."""
        out: Dict[str, Any] = {}
        for column, public_name in (
            ("requested_policy", "requested_policy"),
            ("actual_policy", "actual_policy"),
            ("usage_json", "usage"),
        ):
            raw = row.get(column)
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise DurableEvidenceError(
                    f"invalid durable evidence JSON in {column}"
                ) from exc
            if not isinstance(decoded, dict):
                raise DurableEvidenceError(
                    f"durable evidence {column} must be a JSON object"
                )
            out[public_name] = decoded
        if row.get("fallback_reason") is not None:
            out["fallback_reason"] = row["fallback_reason"]
        return out

    def _run_status_from_store(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild a pollable status dict from a durable runs row (post-restart)."""
        status: Dict[str, Any] = {
            "object": "hermes.run",
            "run_id": row["run_id"],
            "status": row.get("status"),
            "session_id": row.get("session_id"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }
        status.update(self._evidence_from_row(row))
        return status

    async def _handle_run_events(self, request: "web.Request") -> "web.StreamResponse":
        """GET /v1/runs/{run_id}/events — SSE stream of structured agent lifecycle events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]

        if self._broker_enabled():
            return await self._handle_run_events_durable(request, run_id)

        # Allow subscribing slightly before the run is registered (race condition window)
        for _ in range(20):
            if run_id in self._run_streams:
                break
            await asyncio.sleep(0.05)
        else:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        q = self._run_streams[run_id]
        self._run_stream_subscribers.add(run_id)

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if event is None:
                    # Run finished — send final SSE comment and close
                    await response.write(b": stream closed\n\n")
                    break
                payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            self._run_stream_subscribers.discard(run_id)
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)

        return response

    async def _handle_run_events_durable(
        self, request: "web.Request", run_id: str
    ) -> "web.StreamResponse":
        """Durable, cursor-resumable SSE for runs (contract matrix §2 row 4).

        Canonical events live in the durable store, so SSE disconnect never
        deletes them and a fresh cursor (``?since={seq}`` or ``Last-Event-ID``)
        replays with no gap/dup. Each subscriber gets its own fan-out queue, so
        concurrent subscribers don't race one shared queue.
        """
        # Resolve the resume cursor BEFORE prepare() (headers/status are fixed
        # after). Last-Event-ID takes precedence over the ?since= query param.
        cursor = 0
        raw_cursor = request.headers.get("Last-Event-ID") or request.query.get("since")
        if raw_cursor:
            try:
                cursor = max(0, int(str(raw_cursor).strip()))
            except (TypeError, ValueError):
                return web.json_response(
                    _openai_error("Invalid resume cursor", code="invalid_cursor"),
                    status=400,
                )

        if run_id in self._run_authority_unavailable:
            return self._durable_unavailable_response()

        try:
            durable_run = self._durable_store.get_run(run_id)
        except Exception:
            logger.exception(
                "[api_server] durable SSE run lookup failed for %s", run_id
            )
            return self._durable_unavailable_response()

        if durable_run is None and (
            run_id in self._run_event_subscribers
            or run_id in self._run_streams
            or run_id in self._run_statuses
        ):
            # Live transport without its canonical parent is corruption, not a
            # normal pre-registration race.
            return self._durable_unavailable_response()

        terminal = self._broker_is_terminal(run_id) or bool(
            durable_run
            and str(durable_run.get("status")) in {"succeeded", "failed", "stopped"}
        )
        known = (
            terminal
            or run_id in self._run_event_subscribers
            or run_id in self._run_streams
            or run_id in self._run_statuses
            or durable_run is not None
        )
        if not known:
            # Allow subscribing slightly before the run is registered (race window).
            for _ in range(20):
                try:
                    durable_run = self._durable_store.get_run(run_id)
                except Exception:
                    logger.exception(
                        "[api_server] durable SSE registration lookup failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()
                if durable_run is not None:
                    known = True
                    terminal = str(durable_run.get("status")) in {
                        "succeeded", "failed", "stopped"
                    }
                    break
                await asyncio.sleep(0.05)
            if not known:
                return web.json_response(
                    _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                    status=404,
                )

        # Subscribe before reading the durable backlog. Events committed while
        # replay is in progress are then both durable and queued; ``last_sent``
        # below removes the intentional overlap without opening a gap.
        live_q = None
        if not terminal:
            live_q = self._broker_subscribe(run_id)
            self._run_stream_subscribers.add(run_id)

        # Durable backlog strictly after the cursor (no gap, no dup).
        try:
            backlog = self._durable_store.replay_events(run_id, since_seq=cursor)
        except Exception:
            logger.exception("[api_server] durable replay failed for run %s", run_id)
            if live_q is not None:
                self._broker_unsubscribe(run_id, live_q)
            self._run_stream_subscribers.discard(run_id)
            return web.json_response(
                _openai_error(
                    "Durable event replay is unavailable; retry later",
                    code="durable_unavailable",
                ),
                status=503,
                headers={"Retry-After": "1"},
            )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            # 1) Replay the durable backlog (spec-compliant frames with id:/event:).
            last_sent = cursor
            for rec in backlog:
                if rec.seq <= last_sent:
                    continue  # belt-and-suspenders: no duplicate across the seam
                frame = {
                    **rec.payload,
                    "seq": rec.seq,
                    "event": rec.event_type,
                    "event_id": rec.event_id,
                }
                await response.write(self._sse_frame(frame))
                last_sent = rec.seq

            # 2) Bridge to live fan-out for a still-running run.
            if live_q is not None:
                while True:
                    try:
                        event = await asyncio.wait_for(live_q.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        if self._broker_is_terminal(run_id):
                            await response.write(b": stream closed\n\n")
                            break
                        await response.write(b": keepalive\n\n")
                        continue
                    if event is None:
                        await response.write(b": stream closed\n\n")
                        break
                    ev_seq = event.get("seq")
                    if ev_seq is not None and ev_seq <= last_sent:
                        continue  # already replayed from the durable backlog
                    await response.write(self._sse_frame(event))
                    if ev_seq is not None:
                        last_sent = ev_seq
            else:
                # Terminal run: backlog fully replayed, close immediately.
                await response.write(b": stream closed\n\n")
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            if live_q is not None:
                self._broker_unsubscribe(run_id, live_q)
            self._run_stream_subscribers.discard(run_id)

        return response


    async def _handle_run_approval(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        status = self._run_statuses.get(run_id)
        if self._broker_enabled():
            try:
                durable_run = self._durable_store.get_run(run_id)
            except Exception:
                logger.exception(
                    "[api_server] durable approval run lookup failed for %s", run_id
                )
                return self._durable_unavailable_response()
            if durable_run is None:
                if status is not None:
                    return self._durable_unavailable_response()
                return web.json_response(
                    _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                    status=404,
                )
        elif status is None:
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_choice = str(body.get("choice", "")).strip().lower()
        aliases = {"approve": "once", "approved": "once", "allow": "once"}
        choice = aliases.get(raw_choice, raw_choice)
        allowed = {"once", "session", "always", "deny"}
        if choice not in allowed:
            return web.json_response(
                _openai_error(
                    "Invalid approval choice; expected one of: once, session, always, deny",
                    code="invalid_approval_choice",
                ),
                status=400,
            )

        # Validate exact challenge identity before any live queue mutation.
        challenge_id_str: Optional[str] = None
        action_digest_str: Optional[str] = None
        approval_id_str: Optional[str] = None
        decision_replay = False
        response_replay = False
        release_replay = False

        def _durable_approval_result(
            waiter_signal_status: str,
            *,
            idempotent_replay: bool = False,
        ) -> "web.Response":
            payload: Dict[str, Any] = {
                "object": "hermes.run.approval_response",
                "run_id": run_id,
                "choice": choice,
                "decision_status": "committed",
                "waiter_signal_status": waiter_signal_status,
                "challenge_id": challenge_id_str,
                "approval_id": approval_id_str,
                "action_digest": action_digest_str,
            }
            if idempotent_replay:
                payload["idempotent_replay"] = True
            return web.json_response(payload)

        if self._broker_enabled():
            challenge_id = body.get("challenge_id")
            action_digest = body.get("action_digest")
            if not challenge_id or not action_digest:
                return web.json_response(
                    _openai_error(
                        "Approval requires challenge_id and action_digest",
                        code="approval_challenge_required",
                    ),
                    status=409,
                )
            challenge_id_str = str(challenge_id)
            action_digest_str = str(action_digest)
            try:
                grant = self._durable_store.get_approval_challenge(challenge_id_str)
            except Exception:
                logger.exception(
                    "[api_server] durable approval challenge lookup failed for %s",
                    run_id,
                )
                return self._durable_unavailable_response()
            if grant is None or grant.get("run_id") != run_id:
                return web.json_response(
                    _openai_error(
                        "Approval challenge is invalid for this run",
                        code="approval_challenge_invalid",
                    ),
                    status=409,
                )
            approval_id_str = str(grant.get("approval_id") or "")
            if not approval_id_str:
                return web.json_response(
                    _openai_error(
                        "Approval challenge is not bound to a pending action",
                        code="approval_challenge_invalid",
                    ),
                    status=409,
                )
            if str(grant.get("action_digest") or "") != action_digest_str:
                return web.json_response(
                    _openai_error(
                        "Approval challenge is stale, expired, already used, or mismatched",
                        code="approval_challenge_invalid",
                    ),
                    status=409,
                )
            # An immutable committed decision may recover delivery only for the
            # exact same choice.  A second/different choice can never rewrite it.
            if grant.get("consumed"):
                try:
                    prior_decision = self._durable_store.get_approval_decision(
                        challenge_id_str
                    )
                except Exception:
                    logger.exception(
                        "[api_server] durable approval decision lookup failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()
                if (
                    prior_decision is None
                    or prior_decision.payload.get("approval_id") != approval_id_str
                    or prior_decision.payload.get("action_digest") != action_digest_str
                    or prior_decision.payload.get("choice") != choice
                ):
                    return web.json_response(
                        _openai_error(
                            "Approval challenge already has a different or invalid decision",
                            code="approval_challenge_invalid",
                        ),
                        status=409,
                    )
                try:
                    prior_response = self._durable_store.get_approval_response(
                        challenge_id_str
                    )
                except Exception:
                    logger.exception(
                        "[api_server] durable approval response lookup failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()
                if prior_response is not None:
                    response_payload = prior_response.payload
                    if (
                        response_payload.get("approval_id") != approval_id_str
                        or response_payload.get("action_digest") != action_digest_str
                        or response_payload.get("choice") != choice
                    ):
                        return web.json_response(
                            _openai_error(
                                "Approval response conflicts with the immutable decision",
                                code="approval_challenge_invalid",
                            ),
                            status=409,
                        )
                    response_replay = True
                try:
                    prior_release = self._durable_store.get_approval_release(
                        challenge_id_str
                    )
                    prior_delivery = self._durable_store.get_approval_delivery(
                        challenge_id_str
                    )
                except Exception:
                    logger.exception(
                        "[api_server] durable approval release/signal lookup failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()
                if prior_release is not None:
                    release_payload = prior_release.payload
                    if (
                        not response_replay
                        or release_payload.get("approval_id") != approval_id_str
                        or release_payload.get("action_digest") != action_digest_str
                        or release_payload.get("choice") != choice
                    ):
                        return web.json_response(
                            _openai_error(
                                "Approval release conflicts with canonical response",
                                code="approval_challenge_invalid",
                            ),
                            status=409,
                        )
                    release_replay = True
                if prior_delivery is not None:
                    delivery_payload = prior_delivery.payload
                    if (
                        not release_replay
                        or delivery_payload.get("approval_id") != approval_id_str
                        or delivery_payload.get("action_digest") != action_digest_str
                        or delivery_payload.get("choice") != choice
                    ):
                        return web.json_response(
                            _openai_error(
                                "Approval delivery conflicts with canonical response",
                                code="approval_challenge_invalid",
                            ),
                            status=409,
                        )
                    return _durable_approval_result(
                        "confirmed", idempotent_replay=True
                    )
                decision_replay = True
            try:
                expires_at = float(grant.get("expires_at") or 0.0)
            except (TypeError, ValueError):
                expires_at = 0.0
            if not decision_replay and expires_at <= time.time():
                return web.json_response(
                    _openai_error(
                        "Approval challenge is stale, expired, already used, or mismatched",
                        code="approval_challenge_invalid",
                    ),
                    status=409,
                )

        approval_session_key = self._run_approval_sessions.get(run_id)
        if not approval_session_key:
            if self._broker_enabled() and decision_replay:
                return _durable_approval_result(
                    "unknown", idempotent_replay=True
                )
            # Grant NOT consumed — client may retry when session is live again.
            return web.json_response(
                _openai_error(
                    f"Run has no active approval session: {run_id}",
                    code="approval_not_active",
                ),
                status=409,
            )

        from tools.approval import (
            claim_gateway_approval,
            finalize_gateway_approval_claim,
            resolve_gateway_approval,
            restore_gateway_approval_claim,
        )

        if self._broker_enabled():
            assert challenge_id_str is not None
            assert action_digest_str is not None
            assert approval_id_str is not None
            # Claim the exact live waiter before burning an immutable durable
            # decision.  The claim is unsignalled and temporarily suspends its
            # timeout; every failure below restores it for a safe retry.
            claim = claim_gateway_approval(
                approval_session_key, approval_id=approval_id_str
            )
            if claim is None:
                if decision_replay:
                    return _durable_approval_result(
                        "unknown", idempotent_replay=True
                    )
                return web.json_response(
                    _openai_error(
                        f"Run has no pending approval: {run_id}",
                        code="approval_not_pending",
                    ),
                    status=409,
                )
            if decision_replay:
                try:
                    delivery_allowed = self._durable_store.approval_delivery_allowed(
                        challenge_id_str,
                        approval_id=approval_id_str,
                        action_digest=action_digest_str,
                        choice=choice,
                    )
                except Exception:
                    restore_gateway_approval_claim(approval_session_key, claim)
                    logger.exception(
                        "[api_server] durable approval replay fence failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()
                if not delivery_allowed:
                    restore_gateway_approval_claim(approval_session_key, claim)
                    return _durable_approval_result(
                        "unknown", idempotent_replay=True
                    )
            if not decision_replay:
                try:
                    decision_record = self._durable_store.consume_approval_with_event(
                        challenge_id_str,
                        approval_id=approval_id_str,
                        action_digest=action_digest_str,
                        choice=choice,
                    )
                except Exception:
                    restore_gateway_approval_claim(approval_session_key, claim)
                    logger.exception(
                        "[api_server] durable approval decision commit failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()
                if decision_record is None:
                    restore_gateway_approval_claim(approval_session_key, claim)
                    return web.json_response(
                        _openai_error(
                            "Approval challenge is stale, expired, already used, or mismatched",
                            code="approval_challenge_invalid",
                        ),
                        status=409,
                    )
                # Preserve live SSE contiguity: the atomic store helper already
                # assigned seq/event_id, so fan it out without appending again.
                self._broker_fanout_record(decision_record)
            responded_event = {
                "event": "approval.responded",
                "run_id": run_id,
                "challenge_id": challenge_id_str,
                "approval_id": approval_id_str,
                "action_digest": action_digest_str,
                "timestamp": time.time(),
                "choice": choice,
                "decision_status": "committed",
                "waiter_signal_status": "unknown",
            }
            if not response_replay:
                try:
                    # Persist the human response first. It says nothing about
                    # the process-local waiter signal.
                    self._broker_publish(
                        run_id,
                        responded_event,
                        interrupt_on_failure=False,
                    )
                except Exception:
                    restore_gateway_approval_claim(approval_session_key, claim)
                    logger.exception(
                        "[api_server] durable approval response commit failed for %s",
                        run_id,
                    )
                    return self._durable_unavailable_response()

            release_event = {
                "event": "approval.release_committed",
                "run_id": run_id,
                "challenge_id": challenge_id_str,
                "approval_id": approval_id_str,
                "action_digest": action_digest_str,
                "timestamp": time.time(),
                "choice": choice,
                "decision_status": "committed",
                "waiter_signal_status": "unknown",
            }

            def _persist_release_before_signal() -> None:
                self._broker_publish(
                    run_id,
                    release_event,
                    interrupt_on_failure=False,
                )

            try:
                delivered = finalize_gateway_approval_claim(
                    claim,
                    choice,
                    before_signal=(
                        None if release_replay else _persist_release_before_signal
                    ),
                )
            except Exception:
                # A durable release commitment is intentionally not signal
                # proof.  If the exact live Event is still unset, restore the
                # waiter and report the committed/unknown crash gap.
                try:
                    committed_release = self._durable_store.get_approval_release(
                        challenge_id_str
                    )
                except Exception:
                    committed_release = None
                if claim.event.is_set():
                    delivered = True
                else:
                    restore_gateway_approval_claim(approval_session_key, claim)
                    logger.exception(
                        "[api_server] live approval delivery failed for %s", run_id
                    )
                    if committed_release is not None:
                        return _durable_approval_result(
                            "unknown", idempotent_replay=decision_replay
                        )
                    return self._durable_unavailable_response()
            if not delivered:
                restore_gateway_approval_claim(approval_session_key, claim)
                logger.error(
                    "[api_server] live approval delivery lost its exact waiter for %s",
                    run_id,
                )
                try:
                    committed_release = self._durable_store.get_approval_release(
                        challenge_id_str
                    )
                except Exception:
                    committed_release = None
                if committed_release is not None:
                    return _durable_approval_result(
                        "unknown", idempotent_replay=decision_replay
                    )
                return self._durable_unavailable_response()

            signalled_event = {
                "event": "approval.signalled",
                "run_id": run_id,
                "challenge_id": challenge_id_str,
                "approval_id": approval_id_str,
                "action_digest": action_digest_str,
                "timestamp": time.time(),
                "choice": choice,
                "decision_status": "committed",
                "waiter_signal_status": "confirmed",
            }
            signal_recorded = False
            try:
                self._broker_publish(
                    run_id,
                    signalled_event,
                    interrupt_on_failure=False,
                )
                signal_recorded = True
            except Exception:
                # Event.set already happened.  Never re-signal on retry; this
                # request can report its live observation, while a later
                # restart/replay without approval.signalled must say unknown.
                logger.exception(
                    "[api_server] approval signal observation was not persisted for %s",
                    run_id,
                )
            try:
                self._set_run_status(
                    run_id,
                    "running",
                    last_event=(
                        "approval.signalled"
                        if signal_recorded
                        else "approval.release_committed"
                    ),
                )
            except Exception:
                logger.exception(
                    "[api_server] approval signal status refresh failed for %s",
                    run_id,
                )
            return _durable_approval_result("confirmed")
        else:
            resolve_all = (
                _coerce_request_bool(body.get("all"), default=False)
                or _coerce_request_bool(body.get("resolve_all"), default=False)
            )
            try:
                resolved = resolve_gateway_approval(
                    approval_session_key, choice, resolve_all=resolve_all
                )
            except Exception as exc:
                logger.exception(
                    "[api_server] approval resolution failed for run %s", run_id
                )
                return web.json_response(_openai_error(str(exc)), status=500)
            if resolved <= 0:
                return web.json_response(
                    _openai_error(
                        f"Run has no pending approval: {run_id}",
                        code="approval_not_pending",
                    ),
                    status=409,
                )

        try:
            self._set_run_status(run_id, "running", last_event="approval.responded")
        except Exception:
            logger.exception(
                "[api_server] approval resolved but run status refresh failed for %s",
                run_id,
            )
        if not self._broker_enabled():
            q = self._run_streams.get(run_id)
            if q is not None:
                try:
                    q.put_nowait({
                        "event": "approval.responded",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "choice": choice,
                        "resolved": resolved,
                    })
                except Exception:
                    pass

        return web.json_response({
            "object": "hermes.run.approval_response",
            "run_id": run_id,
            "choice": choice,
            "resolved": resolved,
            **(
                {
                    "challenge_id": challenge_id_str,
                    "approval_id": approval_id_str,
                    "action_digest": action_digest_str,
                }
                if self._broker_enabled()
                else {}
            ),
        })

    async def _handle_stop_run(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/stop — interrupt a running agent.

        V2.8 (contract row 7): stop is idempotent when a durable store is
        configured. Repeating stop on a known terminal run returns the SAME
        result (200), never a 404 — the durable store is the authority on
        terminal state, so a cleaned-up live ref does not make a finished run
        unaddressable. A run that never existed still 404s.

        Idempotent responses report the **actual** terminal status from the
        store (succeeded/failed/stopped) — never coerce a succeeded/failed run
        into ``stopped`` (plan A6: stop must not rewrite the terminal fact).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        agent = self._active_run_agents.get(run_id)
        task = self._active_run_tasks.get(run_id)
        durable_row: Optional[Dict[str, Any]] = None

        # Canonical durable terminal truth wins even if stale live refs remain.
        # Conversely, a live ref with no canonical row is an authority failure.
        if self._broker_enabled():
            try:
                durable_row = self._durable_store.get_run(run_id)
            except Exception:
                logger.exception(
                    "[api_server] durable stop lookup failed for %s", run_id
                )
                return self._durable_unavailable_response()
            if durable_row is None:
                if agent is not None or task is not None or run_id in self._run_authority_unavailable:
                    return self._durable_unavailable_response()
                return web.json_response(
                    _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                    status=404,
                )
            store_status = str(durable_row.get("status") or "")
            try:
                projected = self._public_run_status(
                    {
                        "run_id": run_id,
                        "status": store_status,
                        "idempotent_replay": True,
                    }
                )
            except ValueError:
                return self._run_state_invalid_response(run_id)
            if projected["status"] in {"succeeded", "failed", "stopped"}:
                self._run_authority_unavailable.discard(run_id)
                return web.json_response(projected)

        if agent is None and task is None:
            # No live ref. If the store knows this run, return its actual status
            # (idempotent no-op for terminals; mark-stopped for phantoms).
            if self._broker_enabled():
                assert durable_row is not None
                if str(durable_row.get("status")) not in {"queued", "running"}:
                    return self._durable_unavailable_response()
                try:
                    from gateway.durable_runs import RunState

                    self._durable_store.finalize_run(
                        run_id,
                        terminal_state=RunState.STOPPED,
                        event_type="run.stopped",
                        event_payload={
                            "event": "run.stopped",
                            "run_id": run_id,
                            "reason": "stop_without_live_task",
                            "timestamp": time.time(),
                        },
                    )
                except Exception:
                    self._run_authority_unavailable.add(run_id)
                    logger.exception(
                        "[api_server] durable stop transition failed for %s", run_id
                    )
                    return self._durable_unavailable_response()
                self._run_authority_unavailable.discard(run_id)
                return web.json_response(
                    {"run_id": run_id, "status": "stopped", "idempotent_replay": True}
                )
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )

        if self._broker_enabled() and run_id not in self._stop_intent_run_ids:
            try:
                self._broker_publish(
                    run_id,
                    {
                        "event": "run.stop_requested",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    },
                )
            except Exception:
                logger.exception(
                    "[api_server] durable stop intent failed for %s", run_id
                )
                return self._durable_unavailable_response()
            self._stop_intent_run_ids.add(run_id)
            # Classification is derived from the committed intent and must be
            # set before any fallible projection/readback below.
            self._stopping_run_ids.add(run_id)

        try:
            self._set_run_status(run_id, "stopping", last_event="run.stopping")
        except Exception:
            logger.exception(
                "[api_server] durable stopping projection failed for %s", run_id
            )
            if agent is not None:
                try:
                    agent.interrupt("Stop accepted but durable status verification failed")
                except Exception:
                    pass
            return self._durable_unavailable_response()
        self._stopping_run_ids.add(run_id)

        if agent is not None:
            try:
                agent.interrupt("Stop requested via API")
            except Exception:
                logger.exception(
                    "[api_server] live stop delivery failed for %s", run_id
                )
                if self._broker_enabled():
                    return self._durable_unavailable_response()

        return web.json_response(
            {"run_id": run_id, "status": "running", "substate": "stopping"}
        )

    async def _sweep_orphaned_runs(self) -> None:
        """Periodically expire transport buffers and terminal status records."""
        while True:
            await asyncio.sleep(60)
            self._sweep_orphaned_runs_once(time.time())

    def _sweep_orphaned_runs_once(self, now: Optional[float] = None) -> None:
        """Expire old SSE buffers without treating transport age as run age."""
        if now is None:
            now = time.time()
        stale = [
            run_id
            for run_id, created_at in list(self._run_streams_created.items())
            if now - created_at > self._RUN_STREAM_TTL
            and run_id not in self._run_stream_subscribers
        ]
        for run_id in stale:
            logger.debug("[api_server] sweeping expired run transport %s", run_id)
            task = self._active_run_tasks.get(run_id)
            task_done = task is None or task.done()
            if task_done:
                try:
                    from tools.approval import unregister_gateway_notify

                    approval_session_key = self._run_approval_sessions.get(run_id)
                    if approval_session_key:
                        unregister_gateway_notify(approval_session_key)
                except Exception:
                    pass
            # The transport TTL always bounds buffering. Live control state is
            # independent and survives until the executor-backed task returns.
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)
            if task_done:
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)
                self._stopping_run_ids.discard(run_id)

        stale_statuses = [
            run_id
            for run_id, status in list(self._run_statuses.items())
            if status.get("status") in {"completed", "failed", "cancelled"}
            and now - float(status.get("updated_at", 0) or 0) > self._RUN_STATUS_TTL
        ]
        for run_id in stale_statuses:
            self._run_statuses.pop(run_id, None)

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    def _api_key_passes_startup_guard(self) -> bool:
        """Return True when API_SERVER_KEY is present and strong enough to start."""
        if not self._api_key:
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY is required for the API server, "
                "including loopback-only binds on %s.",
                self.name, self._host,
            )
            return False

        try:
            from hermes_cli.auth import has_usable_secret
            if not has_usable_secret(self._api_key, min_length=16):
                logger.error(
                    "[%s] Refusing to start: API_SERVER_KEY is a "
                    "placeholder or too short (<16 chars). This endpoint "
                    "dispatches terminal-capable agent work — a guessable "
                    "key is remote code execution. Generate a strong secret "
                    "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                    "before starting the API server on %s.",
                    self.name, self._host,
                )
                return False
        except ImportError:
            pass
        return True

    def _close_owned_durable_store_then_release_authority(self) -> None:
        """Close factory-owned SQLite while fenced, then release authority.

        Startup can fail before ``disconnect()`` is reached (bind, reconcile,
        or generic application setup).  A factory-owned store has already
        opened WAL and may have performed DDL, so the lock must remain held
        through its final checkpoint/close on every one of those paths.
        Borrowed stores remain open, but their adapter-owned fence is released.
        """
        durable_store = getattr(self, "_durable_store", None)
        if getattr(self, "_owns_durable_store", False) and durable_store is not None:
            self._durable_store = None
            self._owns_durable_store = False
            try:
                durable_store.close()
            except Exception:
                logger.debug(
                    "Failed to close durable run store for %s",
                    self.name,
                    exc_info=True,
                )
        authority_lock = getattr(self, "_durable_authority_lock", None)
        if authority_lock is not None:
            authority_lock.release()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        if not self._api_key_passes_startup_guard():
            return False

        if self._durable_store_required and self._durable_store is None:
            logger.error(
                "[%s] Refusing to reconnect with a closed durable store; "
                "construct a fresh adapter/store instead of degrading to legacy mode.",
                self.name,
            )
            return False

        if self._broker_enabled():
            from gateway.durable_runs import DurableAuthorityLock

            if self._durable_authority_lock is None:
                self._durable_authority_lock = DurableAuthorityLock(
                    self._durable_store.db_path
                )
            if not self._durable_authority_lock.acquire():
                self._set_fatal_error(
                    "durable_authority_held",
                    "Another API server process owns the durable run database; "
                    "refusing a second writer/reconciler.",
                    retryable=False,
                )
                logger.error(
                    "[%s] Refusing to start: durable authority already held for %s",
                    self.name,
                    self._durable_store.db_path,
                )
                return False

        try:
            mws = [
                mw
                for mw in (
                    self._make_profile_prefix_middleware(),
                    cors_middleware,
                    body_limit_middleware,
                    security_headers_middleware,
                )
                if mw is not None
            ]
            self._app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
            assert self._app is not None
            # Native routes + multiplex /p/<profile>/… mirrors. Same handlers;
            # the profile-prefix middleware validates the prefix and scopes
            # config/credentials to that profile when multiplexing is on.
            for method, path, handler in self._http_route_table():
                self._app.router.add_route(method, path, handler)
                self._app.router.add_route(method, f"/p/{{profile}}{path}", handler)
            # Store the adapter after native routes are registered. Local Hermes-Relay
            # bootstrap shims use this key as a feature-detection hook; registering
            # native routes first lets those shims no-op instead of shadowing the
            # upstream session-control handlers.
            self._app["api_server_adapter"] = self
            if self.gateway_runner is not None:
                self._app["gateway_runner"] = self.gateway_runner

            # Loud warning when a network-accessible API server runs against an
            # unsandboxed local terminal backend. The API server can drive the
            # agent's terminal/file tools as the host user; on a public bind
            # that is the exact surface the hermes-0day campaign abused to write
            # ~/.hermes/config.yaml and plant persistence. Sandboxing (Docker /
            # remote backend) contains the blast radius. Warn, don't refuse —
            # the operator may have an external firewall / strong key.
            if is_network_accessible(self._host):
                try:
                    from hermes_cli.config import load_config as _load_cfg
                    _backend = (
                        ((_load_cfg() or {}).get("terminal") or {}).get(
                            "backend", "local"
                        )
                    )
                except Exception:
                    _backend = "local"
                if str(_backend).lower() == "local":
                    logger.warning(
                        "[%s] API server is network-accessible (%s) AND the "
                        "terminal backend is 'local' (unsandboxed). Agent work "
                        "dispatched through this endpoint runs as the host user "
                        "with full terminal/file access. Strongly consider a "
                        "sandboxed backend (terminal.backend: docker) and "
                        "firewalling this port to trusted networks only.",
                        self.name, self._host,
                    )

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            # Bind directly instead of probing 127.0.0.1 first — the old
            # single-family pre-probe raced the real bind and reported a
            # TIME_WAIT socket as "in use" (#10297), failing gateway
            # restarts for up to ~60s.
            #
            # SO_REUSEADDR is platform-dependent (same rationale as the
            # webhook adapter, #65482):
            #   - macOS (BSD semantics): two sockets with SO_REUSEADDR can
            #     silently split traffic while both report success — disable.
            #   - Linux: SO_REUSEADDR only permits rebinding past TIME_WAIT
            #     (a second live listener needs SO_REUSEPORT, never set), so
            #     keep the default (enabled) for instant restart rebinds.
            self._site = web.TCPSite(
                self._runner,
                self._host,
                self._port,
                reuse_address=False if sys.platform == "darwin" else None,
            )
            try:
                await self._site.start()
            except OSError as exc:
                await self._runner.cleanup()
                self._runner = None
                self._site = None
                self._close_owned_durable_store_then_release_authority()
                if getattr(exc, "errno", None) == errno.EADDRINUSE:
                    # A port conflict is a configuration error, not a
                    # transient blip — another process holds the port for
                    # its lifetime. A bare ``return False`` makes the
                    # reconnect watcher in gateway.run treat it as retryable
                    # and loop forever at the backoff cap (observed: 1568+
                    # retries over 5 days across multi-profile setups all
                    # defaulting to the same port, #52132), filling
                    # errors.log and leaking the adapter's ResponseStore
                    # fds each retry. Non-retryable drops it from the
                    # reconnect queue; the operator recovers with
                    # ``/platform resume api_server`` after changing the port.
                    self._set_fatal_error(
                        "api_server_port_in_use",
                        f"Port {self._port} already in use. Set "
                        f"platforms.api_server.port in config.yaml to a "
                        f"different value, then `/platform resume api_server`.",
                        retryable=False,
                    )
                logger.error(
                    "[%s] Could not bind %s:%d: %s. Set a different port in "
                    "config.yaml: platforms.api_server.port",
                    self.name, self._host, self._port, exc,
                )
                return False

            # Mutate durable startup state only after this adapter owns the
            # listening socket. A failed bind must leave another instance's
            # queued/running facts untouched and create no background task.
            self.reconcile_durable_runs()

            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name,
            )
            return True

        except Exception as e:
            # A failure after TCPSite.start (notably startup reconciliation)
            # must tear down the listener before releasing DB authority.
            if self._site is not None:
                try:
                    await self._site.stop()
                except Exception:
                    logger.debug(
                        "Failed to stop partially started API site",
                        exc_info=True,
                    )
                self._site = None
            if self._runner is not None:
                try:
                    await self._runner.cleanup()
                except Exception:
                    logger.debug(
                        "Failed to clean partially started API runner",
                        exc_info=True,
                    )
                self._runner = None
            self._app = None
            self._close_owned_durable_store_then_release_authority()
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server and release all owned resources.

        Closes the ResponseStore SQLite connection in addition to stopping
        the aiohttp web server. Without this, every adapter instance leaks
        2 file descriptors (the database file and its WAL sidecar) — the
        reconnect loop in ``gateway.run`` constructs a fresh adapter on
        every retry, so 2 fds/retry × 300s backoff cap ≈ 12 fds/hour, which
        exhausts the default 2560 fd limit after ~12h of failed reconnects
        and turns the whole gateway into a zombie
        (OSError: [Errno 24] Too many open files, #37011).
        """
        self._mark_disconnected()
        if self._response_store is not None:
            try:
                self._response_store.close()
            except Exception:
                logger.debug(
                    "Failed to close response store for %s", self.name, exc_info=True,
                )
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._close_owned_durable_store_then_release_authority()
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
