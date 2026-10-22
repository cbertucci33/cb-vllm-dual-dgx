#!/usr/bin/env python3
"""OpenAI-compatible telemetry proxy with a stable client model alias."""

from __future__ import annotations

import argparse
import asyncio
import errno
import gzip
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import ClientOSError, ClientSession, ClientTimeout, TCPConnector, web

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}
SECRET_HEADER_PARTS = ("authorization", "api-key", "apikey", "token", "cookie")

REASONING_EFFORT_REWRITES = {
    "glm": {"high": "high"},
    "qwen": {"high": "xhigh"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def safe_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        result[key] = "[REDACTED]" if any(p in lower for p in SECRET_HEADER_PARTS) else value
    return result


def forward_headers(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS and k.lower() != "host"}


async def discover_backend_model(session: ClientSession, upstream: str) -> str:
    url = upstream.rstrip("/") + "/v1/models"
    async with session.get(url, headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        payload = await response.json()
    model_ids = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    }
    if len(model_ids) != 1:
        raise RuntimeError(
            f"backend must expose exactly one model through /v1/models; found {sorted(model_ids)!r}"
        )
    return model_ids.pop()


def backend_model_family(backend_model: str) -> str | None:
    normalized = backend_model.casefold()
    for family in REASONING_EFFORT_REWRITES:
        if family in normalized:
            return family
    return None


def rewrite_model(body: bytes, backend_model: str) -> tuple[bytes, str | None]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, None
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        return body, None
    client_model = payload["model"]
    payload["model"] = backend_model
    reasoning = payload.get("reasoning")
    family = backend_model_family(backend_model)
    if isinstance(reasoning, dict) and family is not None:
        effort = reasoning.get("effort")
        replacement = REASONING_EFFORT_REWRITES[family].get(effort)
        if replacement is not None:
            reasoning["effort"] = replacement
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), client_model


def rewrite_response_identity(
    payload: dict[str, Any], client_model: str, *, models_route: bool = False
) -> None:
    """Replace backend model identities at the public proxy boundary."""
    if isinstance(payload.get("model"), str):
        payload["model"] = client_model
    response = payload.get("response")
    if isinstance(response, dict) and isinstance(response.get("model"), str):
        response["model"] = client_model
    if models_route:
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("id"), str):
                item["id"] = client_model
            if isinstance(item.get("root"), str):
                item["root"] = client_model
            if "parent" in item:
                item["parent"] = None


def rewrite_json_response(body: bytes, client_model: str, *, models_route: bool = False) -> bytes:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    rewrite_response_identity(payload, client_model, models_route=models_route)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


class EventWriter:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.data_dir, 0o700)
        self.lock = asyncio.Lock()

    async def write(self, event: dict[str, Any]) -> None:
        event.setdefault("ts", utc_now())
        path = self.data_dir / f"events-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self.lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            os.chmod(path, 0o600)

    def capture(self, trace_id: str, label: str, body: bytes, limit: int) -> dict[str, Any]:
        capture_dir = self.data_dir / "captures" / datetime.now(timezone.utc).strftime("%Y%m%d")
        capture_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(capture_dir, 0o700)
        saved = body[:limit]
        path = capture_dir / f"{trace_id}.{label}.gz"
        with gzip.open(path, "wb", compresslevel=6) as handle:
            handle.write(saved)
        os.chmod(path, 0o600)
        return {
            "path": str(path),
            "bytes": len(body),
            "saved_bytes": len(saved),
            "truncated": len(saved) != len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }


class StreamSummary:
    def __init__(self) -> None:
        self.buffer = b""
        self.events = 0
        self.json_events = 0
        self.response_ids: set[str] = set()
        self.finish_reasons: set[str] = set()
        self.content_chars = 0
        self.reasoning_chars = 0
        self.tool_fragments = 0
        self.tool_calls: dict[str, dict[str, Any]] = {}
        self.raw_tool_markers = 0
        self.replacement_chars = 0
        self.done = False
        self.terminal_response_status: str | None = None
        self.response_statuses: set[str] = set()
        self.response_usage: dict[str, Any] | None = None
        self.response_output_count = 0
        self.response_output_types: dict[str, int] = {}

    def feed(self, chunk: bytes) -> None:
        self.raw_tool_markers += chunk.count(b"<tool_call>")
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self._line(line.rstrip(b"\r"))

    def finish(self) -> None:
        if self.buffer:
            self._line(self.buffer.rstrip(b"\r"))
            self.buffer = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        self.events += 1
        data = line[5:].strip()
        if data == b"[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        self.json_events += 1
        self._payload(payload)

    def _payload(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("id"), str):
            self.response_ids.add(payload["id"])
        event_type = payload.get("type")
        response = payload.get("response")
        if event_type == "response.completed":
            self.done = True
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str):
                self.response_ids.add(response_id)
            response_status = response.get("status")
            if isinstance(response_status, str):
                self.response_statuses.add(response_status)
                if event_type == "response.completed":
                    self.terminal_response_status = response_status
            usage = response.get("usage")
            if isinstance(usage, dict):
                self.response_usage = usage
            output = response.get("output")
            if isinstance(output, list):
                self.response_output_count = len(output)
                output_types: dict[str, int] = {}
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if isinstance(item_type, str):
                        output_types[item_type] = output_types.get(item_type, 0) + 1
                self.response_output_types = output_types
        for choice in payload.get("choices") or []:
            reason = choice.get("finish_reason")
            if reason:
                self.finish_reasons.add(str(reason))
            message = choice.get("delta") or choice.get("message") or {}
            content = message.get("content")
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if isinstance(content, str):
                self.content_chars += len(content)
                self.replacement_chars += content.count("\ufffd")
            if isinstance(reasoning, str):
                self.reasoning_chars += len(reasoning)
                self.replacement_chars += reasoning.count("\ufffd")
            self.tool_fragments += len(message.get("tool_calls") or [])
            for call in message.get("tool_calls") or []:
                key = str(call.get("index", len(self.tool_calls)))
                current = self.tool_calls.setdefault(key, {"id": None, "name": None, "arguments": ""})
                if call.get("id"):
                    current["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    current["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    current["arguments"] += function["arguments"]

    def as_dict(self) -> dict[str, Any]:
        tool_calls = []
        for index, call in sorted(self.tool_calls.items()):
            arguments = call["arguments"]
            try:
                json.loads(arguments)
                valid_json = True
            except (TypeError, json.JSONDecodeError):
                valid_json = False
            tool_calls.append({
                "index": index, "id": call["id"], "name": call["name"],
                "argument_chars": len(arguments),
                "arguments_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
                "arguments_json_valid": valid_json,
            })
        return {
            "sse_events": self.events,
            "sse_json_events": self.json_events,
            "sse_done": self.done,
            "terminal_response_status": self.terminal_response_status,
            "response_ids": sorted(self.response_ids),
            "response_statuses": sorted(self.response_statuses),
            "response_usage": self.response_usage,
            "response_output_count": self.response_output_count,
            "response_output_types": dict(sorted(self.response_output_types.items())),
            "finish_reasons": sorted(self.finish_reasons),
            "content_chars": self.content_chars,
            "reasoning_chars": self.reasoning_chars,
            "tool_fragments": self.tool_fragments,
            "tool_calls": tool_calls,
            "raw_tool_markers": self.raw_tool_markers,
            "replacement_chars": self.replacement_chars,
        }


class ResponsesIdentityNormalizer:
    """Build terminal output from the exact items completed by the stream."""

    def __init__(self, client_model: str) -> None:
        self.buffer = b""
        self.client_model = client_model
        self.completed_items: dict[int, dict[str, Any]] = {}

    def feed(self, chunk: bytes) -> bytes:
        self.buffer += chunk
        output = bytearray()
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            output.extend(self._line(line))
            output.extend(b"\n")
        return bytes(output)

    def finish(self) -> bytes:
        tail = self._line(self.buffer) if self.buffer else b""
        self.buffer = b""
        return tail

    def _line(self, line: bytes) -> bytes:
        suffix = b"\r" if line.endswith(b"\r") else b""
        content = line[:-1] if suffix else line
        if not content.startswith(b"data:"):
            return line
        data = content[5:].strip()
        if not data or data == b"[DONE]":
            return line
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return line
        if not isinstance(payload, dict):
            return line
        rewrite_response_identity(payload, self.client_model)
        event_type = payload.get("type")
        if event_type == "response.output_item.done":
            output_index = payload.get("output_index")
            item = payload.get("item")
            if isinstance(output_index, int) and isinstance(item, dict):
                self.completed_items[output_index] = item
        if event_type == "response.completed" and self.completed_items:
            response = payload.get("response")
            if isinstance(response, dict):
                last_index = max(self.completed_items)
                if set(self.completed_items) == set(range(last_index + 1)):
                    response["output"] = [
                        self.completed_items[output_index]
                        for output_index in range(last_index + 1)
                    ]
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        return b"data: " + encoded + suffix


def summarize_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"json": False}
    summary: dict[str, Any] = {"json": True}
    if isinstance(payload, dict):
        for key in ("id", "model", "object"):
            if key in payload:
                summary[key] = payload[key]
        if isinstance(payload.get("usage"), dict):
            summary["usage"] = payload["usage"]
        choices = payload.get("choices") or []
        summary["finish_reasons"] = sorted({str(c.get("finish_reason")) for c in choices if c.get("finish_reason")})
        summary["tool_calls"] = sum(len(((c.get("message") or {}).get("tool_calls") or [])) for c in choices)
        messages = payload.get("messages") or []
        if isinstance(messages, list):
            roles: dict[str, int] = {}
            tool_result_ids = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "unknown"))
                roles[role] = roles.get(role, 0) + 1
                if role == "tool" and message.get("tool_call_id"):
                    tool_result_ids.append(str(message["tool_call_id"]))
            summary["message_roles"] = roles
            summary["tool_result_ids"] = tool_result_ids
        if isinstance(payload.get("tools"), list):
            summary["declared_tools"] = [
                str((tool.get("function") or {}).get("name"))
                for tool in payload["tools"] if isinstance(tool, dict)
            ]
    return summary


async def create_app(args: argparse.Namespace) -> web.Application:
    writer = EventWriter(Path(args.data_dir))
    timeout = ClientTimeout(total=None, connect=args.connect_timeout, sock_read=None)
    session = ClientSession(
        timeout=timeout,
        connector=TCPConnector(
            limit=args.max_connections,
            keepalive_timeout=args.upstream_keepalive,
        ),
        auto_decompress=False,
    )
    app = web.Application(client_max_size=args.max_request_bytes)
    app["session"] = session
    app["writer"] = writer
    backend_model_state: dict[str, str | None] = {"value": None}
    backend_model_lock = asyncio.Lock()

    async def get_backend_model() -> str:
        backend_model = backend_model_state["value"]
        if backend_model is not None:
            return backend_model
        async with backend_model_lock:
            backend_model = backend_model_state["value"]
            if backend_model is None:
                backend_model = await discover_backend_model(session, args.upstream)
                backend_model_state["value"] = backend_model
            return backend_model

    async def close_session(_: web.Application) -> None:
        await session.close()

    app.on_cleanup.append(close_session)

    async def health(_: web.Request) -> web.Response:
        try:
            backend_model = await get_backend_model()
        except Exception as exc:
            return web.json_response({
                "ok": False,
                "upstream": args.upstream,
                "client_model": args.client_model,
                "backend_model": None,
                "error_type": type(exc).__name__,
            }, status=503)
        return web.json_response({
            "ok": True,
            "upstream": args.upstream,
            "client_model": args.client_model,
            "backend_model": backend_model,
        })

    async def proxy(request: web.Request) -> web.StreamResponse:
        trace_id = request.headers.get("X-Flight-Trace-Id") or uuid.uuid4().hex
        started_ns = time.monotonic_ns()
        body = await request.read()
        request_capture = writer.capture(trace_id, "request", body, args.capture_limit)
        request_summary = summarize_json(body)
        try:
            backend_model = await get_backend_model()
        except Exception as exc:
            await writer.write({
                "event": "backend_discovery_error", "trace_id": trace_id,
                "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_ms": (time.monotonic_ns() - started_ns) / 1_000_000,
            })
            raise web.HTTPBadGateway(
                text="upstream model discovery failed",
                headers={"X-Flight-Trace-Id": trace_id},
            )
        upstream_body, client_model = rewrite_model(body, backend_model)
        await writer.write({
            "event": "request_start", "trace_id": trace_id, "method": request.method,
            "path_qs": request.path_qs, "peer": request.remote,
            "headers": safe_headers(request.headers), "capture": request_capture,
            "request_summary": request_summary,
            "client_model": client_model,
            "backend_model": backend_model,
        })
        url = args.upstream.rstrip("/") + request.path_qs
        headers = forward_headers(request.headers)
        headers["X-Flight-Trace-Id"] = trace_id
        headers["Accept-Encoding"] = "identity"
        response_bytes = bytearray()
        stream_summary = StreamSummary()
        first_byte_ns: int | None = None
        client_disconnected = False
        try:
            upstream = None
            for attempt in range(2):
                try:
                    upstream = await session.request(
                        request.method,
                        url,
                        headers=headers,
                        data=upstream_body,
                        allow_redirects=False,
                    )
                    break
                except ClientOSError as exc:
                    if attempt > 0 or exc.errno != errno.EPIPE:
                        raise
                    await writer.write({
                        "event": "upstream_retry",
                        "trace_id": trace_id,
                        "attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
            assert upstream is not None
            async with upstream:
                response_headers = {
                    k: v for k, v in upstream.headers.items()
                    if k.lower() not in HOP_HEADERS and k.lower() != "content-encoding"
                }
                content_type = upstream.headers.get("Content-Type", "")
                is_event_stream = "text/event-stream" in content_type
                response_headers["X-Flight-Trace-Id"] = trace_id
                downstream = web.StreamResponse(status=upstream.status, reason=upstream.reason, headers=response_headers)
                await downstream.prepare(request)
                identity_normalizer = ResponsesIdentityNormalizer(args.client_model)
                buffered_json = bytearray()
                async for chunk in upstream.content.iter_any():
                    if first_byte_ns is None:
                        first_byte_ns = time.monotonic_ns()
                    if is_event_stream:
                        delivered = identity_normalizer.feed(chunk)
                    else:
                        buffered_json.extend(chunk)
                        delivered = b""
                    if not delivered:
                        continue
                    response_bytes.extend(delivered)
                    stream_summary.feed(delivered)
                    try:
                        await downstream.write(delivered)
                    except (ConnectionResetError, RuntimeError):
                        client_disconnected = True
                        break
                if is_event_stream:
                    tail = identity_normalizer.finish()
                else:
                    tail = rewrite_json_response(
                        bytes(buffered_json), args.client_model,
                        models_route=request.path == "/v1/models",
                    )
                if tail and not client_disconnected:
                    response_bytes.extend(tail)
                    stream_summary.feed(tail)
                    try:
                        await downstream.write(tail)
                    except (ConnectionResetError, RuntimeError):
                        client_disconnected = True
                stream_summary.finish()
                try:
                    await downstream.write_eof()
                except (ConnectionResetError, RuntimeError):
                    client_disconnected = True
                status = upstream.status
        except Exception as exc:
            await writer.write({
                "event": "proxy_error", "trace_id": trace_id,
                "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_ms": (time.monotonic_ns() - started_ns) / 1_000_000,
            })
            raise web.HTTPBadGateway(text="upstream request failed", headers={"X-Flight-Trace-Id": trace_id})

        response_body = bytes(response_bytes)
        response_capture = writer.capture(trace_id, "response", response_body, args.capture_limit)
        summary = stream_summary.as_dict() if "text/event-stream" in content_type else summarize_json(response_body)
        anomalies: list[str] = []
        if status >= 400:
            anomalies.append(f"http_{status}")
        if client_disconnected:
            anomalies.append("client_disconnected")
        if "text/event-stream" in content_type and not summary.get("sse_done"):
            anomalies.append("missing_sse_done")
        terminal_response_status = summary.get("terminal_response_status")
        if terminal_response_status in {"failed", "incomplete", "cancelled"}:
            anomalies.append(f"response_{terminal_response_status}")
        if summary.get("replacement_chars"):
            anomalies.append("unicode_replacement")
        if summary.get("raw_tool_markers") and not summary.get("tool_fragments"):
            anomalies.append("raw_tool_marker_without_structured_call")
        finished_ns = time.monotonic_ns()
        await writer.write({
            "event": "request_end", "trace_id": trace_id, "status": status,
            "content_type": content_type, "capture": response_capture, "response_summary": summary,
            "ttfb_ms": None if first_byte_ns is None else (first_byte_ns - started_ns) / 1_000_000,
            "elapsed_ms": (finished_ns - started_ns) / 1_000_000,
            "client_disconnected": client_disconnected, "anomalies": anomalies,
        })
        return downstream

    app.router.add_get("/recorder/health", health)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8320)
    parser.add_argument("--data-dir", default="/opt/project/model-flight-recorder-data/proxy")
    parser.add_argument("--capture-limit", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-request-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--max-connections", type=int, default=16)
    parser.add_argument("--upstream-keepalive", type=float, default=4.0)
    parser.add_argument("--client-model", default="model-main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    web.run_app(create_app(args), host=args.listen, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
