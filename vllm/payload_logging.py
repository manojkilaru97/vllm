# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in OpenAI-compatible request and response payload logging.

Payload capture lives at the ASGI boundary so it does not alter model
execution, token generation, parsing, or response construction. When payload
logging is disabled, the middleware is not registered.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import uuid
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_PAYLOAD_LOGGER_NAME = "vllm.payload"
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
_LOGGER_PROVIDER: Any | None = None


def payload_logging_enabled() -> bool:
    return os.getenv("VLLM_LOG_PAYLOADS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configure_payload_logging() -> None:
    """Configure exactly one structured payload-log exporter."""
    global _LOGGER_PROVIDER
    if not payload_logging_enabled() or _LOGGER_PROVIDER is not None:
        return

    payload_logger = logging.getLogger(_PAYLOAD_LOGGER_NAME)
    payload_logger.setLevel(logging.INFO)
    payload_logger.propagate = False

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if not endpoint:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonPayloadFormatter())
        payload_logger.addHandler(handler)
        _LOGGER_PROVIDER = False
        return

    try:
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        protocol = os.getenv(
            "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf"
        ).strip()
        if protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter,
            )
        elif protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter,
            )
        else:
            raise ValueError(
                "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL must be "
                f"'http/protobuf' or 'grpc', got {protocol!r}"
            )

        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", "vllm"),
                "service.instance.id": os.getenv(
                    "OTEL_SERVICE_INSTANCE_ID", os.getenv("HOSTNAME", "instance-0")
                ),
                "service.namespace": "vllm.openai.api_server",
            }
        )
        provider = LoggerProvider(resource=resource)
        provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        payload_logger.addHandler(
            LoggingHandler(level=logging.INFO, logger_provider=provider)
        )
        _LOGGER_PROVIDER = provider
        atexit.register(provider.shutdown)
    except Exception as exc:
        raise RuntimeError(
            "VLLM_LOG_PAYLOADS is enabled, but OpenTelemetry log export "
            "could not be configured"
        ) from exc


class _JsonPayloadFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "event": record.getMessage(),
            **{
                key: value
                for key, value in record.__dict__.items()
                if key
                not in {
                    "args",
                    "created",
                    "exc_info",
                    "exc_text",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "msg",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "thread",
                    "threadName",
                    "taskName",
                }
            },
        }
        return json.dumps(data, ensure_ascii=False, default=str)


def _headers(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in raw_headers
    }


def _request_summary(payload: dict[str, Any]) -> dict[str, Any]:
    image_count = 0
    video_count = 0
    audio_count = 0

    def count_parts(parts: Any) -> None:
        nonlocal image_count, video_count, audio_count
        if not isinstance(parts, list):
            return
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"image_url", "input_image"}:
                image_count += 1
            elif part_type == "video_url":
                video_count += 1
            elif part_type in {"audio_url", "input_audio"}:
                audio_count += 1

    for message in payload.get("messages") or []:
        if isinstance(message, dict):
            count_parts(message.get("content"))
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        count_parts([item])
        count_parts(item.get("content"))

    tools = payload.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            tool_choice_attr = "named"
        else:
            tool_choice_attr = str(tool_choice.get("type") or "named")
    else:
        tool_choice_attr = tool_choice
    response_format = payload.get("response_format")
    structured_kind = (
        response_format.get("type") if isinstance(response_format, dict) else None
    )
    return {
        "input_image_count": image_count,
        "input_video_count": video_count,
        "input_audio_count": audio_count,
        "input_tool_count": tool_count,
        "has_images": image_count > 0,
        "has_videos": video_count > 0,
        "has_audios": audio_count > 0,
        "has_tools": tool_count > 0,
        "has_tool_calls_enabled": tool_count > 0 and tool_choice != "none",
        "has_structured_output": structured_kind is not None,
        **({"tool_choice": tool_choice_attr} if tool_choice_attr is not None else {}),
        **(
            {"structured_output_kind": structured_kind}
            if structured_kind is not None
            else {}
        ),
    }


def _parse_json(data: bytes) -> Any:
    if not data:
        return None
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": data.decode("utf-8", errors="replace")}


def _merge_tool_calls(
    state: dict[int, dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> None:
    for position, tool_call in enumerate(tool_calls):
        index = int(tool_call.get("index", position))
        item = state.setdefault(
            index,
            {
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""},
            },
        )
        if tool_call.get("id"):
            item["id"] = tool_call["id"]
        if tool_call.get("type"):
            item["type"] = tool_call["type"]
        function = tool_call.get("function") or {}
        if function.get("name"):
            item["function"]["name"] = function["name"]
        if function.get("arguments"):
            item["function"]["arguments"] += str(function["arguments"])


def _aggregate_sse(data: bytes) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith("data:"):
            continue
        raw_event = raw_line[5:].strip()
        if not raw_event or raw_event == "[DONE]":
            continue
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)

    for event in reversed(events):
        response = event.get("response")
        if isinstance(response, dict):
            return response

    if not events:
        return {"events": []}

    first = events[0]
    object_type = str(first.get("object") or "")
    if object_type == "chat.completion.chunk":
        choices: dict[int, dict[str, Any]] = {}
        usage = None
        for event in events:
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices") or []:
                index = int(choice.get("index", 0))
                state = choices.setdefault(
                    index,
                    {
                        "content": [],
                        "reasoning_content": [],
                        "tool_calls": {},
                        "finish_reason": None,
                    },
                )
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    state["content"].append(str(delta["content"]))
                reasoning = delta.get("reasoning", delta.get("reasoning_content"))
                if reasoning:
                    state["reasoning_content"].append(str(reasoning))
                _merge_tool_calls(state["tool_calls"], delta.get("tool_calls") or [])
                if choice.get("finish_reason") is not None:
                    state["finish_reason"] = choice["finish_reason"]

        response_choices = []
        for index in sorted(choices):
            state = choices[index]
            message: dict[str, Any] = {"role": "assistant"}
            if state["content"]:
                message["content"] = "".join(state["content"])
            if state["reasoning_content"]:
                message["reasoning_content"] = "".join(state["reasoning_content"])
            if state["tool_calls"]:
                message["tool_calls"] = [
                    state["tool_calls"][i] for i in sorted(state["tool_calls"])
                ]
            response_choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": state["finish_reason"],
                }
            )
        return {
            "id": first.get("id"),
            "object": "chat.completion",
            "created": first.get("created"),
            "model": first.get("model"),
            "choices": response_choices,
            "usage": usage,
        }

    if object_type == "text_completion":
        choices: dict[int, dict[str, Any]] = {}
        usage = None
        for event in events:
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices") or []:
                index = int(choice.get("index", 0))
                state = choices.setdefault(index, {"text": [], "finish_reason": None})
                if choice.get("text"):
                    state["text"].append(str(choice["text"]))
                if choice.get("finish_reason") is not None:
                    state["finish_reason"] = choice["finish_reason"]
        return {
            "id": first.get("id"),
            "object": object_type,
            "created": first.get("created"),
            "model": first.get("model"),
            "choices": [
                {
                    "index": index,
                    "text": "".join(choices[index]["text"]),
                    "finish_reason": choices[index]["finish_reason"],
                }
                for index in sorted(choices)
            ],
            "usage": usage,
        }

    return {"events": events}


def _response_payload(data: bytes, content_type: str) -> dict[str, Any]:
    if "text/event-stream" in content_type:
        return _aggregate_sse(data)
    payload = _parse_json(data)
    return payload if isinstance(payload, dict) else {"body": payload}


def _response_rid(payload: dict[str, Any]) -> str | None:
    response_id = payload.get("id")
    if not isinstance(response_id, str):
        return None
    for prefix in ("chatcmpl-", "cmpl-", "resp_"):
        if response_id.startswith(prefix):
            return response_id[len(prefix) :]
    return None


def _emit_exchange(
    *,
    request_body: bytes,
    request_headers: dict[str, str],
    response_body: bytes,
    response_headers: dict[str, str],
    path: str,
    status_code: int | None,
    fallback_rid: str,
) -> None:
    request_payload = _parse_json(request_body)
    if not isinstance(request_payload, dict):
        request_payload = {"body": request_payload}
    response_payload = _response_payload(
        response_body, response_headers.get("content-type", "")
    )
    rid = (
        request_headers.get("x-request-id")
        or _response_rid(response_payload)
        or fallback_rid
    )
    payload_logger = logging.getLogger(_PAYLOAD_LOGGER_NAME)
    payload_logger.info(
        "openai.request",
        extra={
            "rid": rid,
            "request_id": rid,
            "endpoint": path,
            "payload": request_payload,
            "headers": request_headers,
            **_request_summary(request_payload),
        },
    )
    payload_logger.info(
        "openai.response",
        extra={
            "rid": rid,
            "request_id": rid,
            "endpoint": path,
            "http_status_code": status_code,
            "payload": response_payload,
        },
    )


def _schedule_background(func: Callable[[], None]) -> None:
    async def run() -> None:
        await asyncio.to_thread(func)

    task = asyncio.create_task(run())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


class PayloadLoggingMiddleware:
    """Capture complete HTTP exchanges without changing transmitted bytes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if not path.startswith(("/v1/", "/v2/", "/inference/")):
            await self.app(scope, receive, send)
            return

        request_body: list[bytes] = []
        response_body: list[bytes] = []
        request_headers = _headers(scope.get("headers") or [])
        response_headers: dict[str, str] = {}
        status_code: int | None = None
        fallback_rid = uuid.uuid4().hex

        async def receive_with_capture() -> Message:
            message = await receive()
            if message["type"] == "http.request" and message.get("body"):
                request_body.append(bytes(message["body"]))
            return message

        async def send_with_capture(message: Message) -> None:
            nonlocal response_headers, status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = _headers(message.get("headers") or [])
            elif message["type"] == "http.response.body" and message.get("body"):
                response_body.append(bytes(message["body"]))
            await send(message)

        try:
            await self.app(scope, receive_with_capture, send_with_capture)
        finally:
            request_bytes = b"".join(request_body)
            response_bytes = b"".join(response_body)
            _schedule_background(
                lambda: _emit_exchange(
                    request_body=request_bytes,
                    request_headers=request_headers,
                    response_body=response_bytes,
                    response_headers=response_headers,
                    path=path,
                    status_code=status_code,
                    fallback_rid=fallback_rid,
                )
            )
