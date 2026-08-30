# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, lossless Chat Completions request and response logging."""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm import envs

_PAYLOAD_LOGGER_NAME = "vllm.payload"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_SPOOL_MEMORY_LIMIT = 1024 * 1024
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
_LOGGER_PROVIDER: Any | None = None
_LOGGING_CONFIGURED = False
_MODULE_LOGGER = logging.getLogger(__name__)


def payload_logging_enabled() -> bool:
    return envs.VLLM_LOG_PAYLOADS


def _install_stream_handler(payload_logger: logging.Logger) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonPayloadFormatter())
    payload_logger.addHandler(handler)


def configure_payload_logging() -> None:
    """Configure payload export without preventing the server from starting."""
    global _LOGGER_PROVIDER, _LOGGING_CONFIGURED
    if not payload_logging_enabled() or _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    payload_logger = logging.getLogger(_PAYLOAD_LOGGER_NAME)
    payload_logger.setLevel(logging.INFO)
    payload_logger.propagate = False

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if not endpoint:
        _install_stream_handler(payload_logger)
        return

    provider = None
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
    except Exception:
        if provider is not None:
            with contextlib.suppress(Exception):
                provider.shutdown()
        _install_stream_handler(payload_logger)
        _MODULE_LOGGER.warning(
            "OpenTelemetry payload-log export could not be configured; "
            "falling back to stderr",
            exc_info=True,
        )


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


def _header_value(raw_headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    wanted = name.lower()
    for key, value in raw_headers:
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None


def _body_text(data: bytes) -> tuple[str, str]:
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return base64.b64encode(data).decode("ascii"), "base64"


def _json_payload(data: bytes) -> Any:
    if not data:
        return None
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        text, encoding = _body_text(data)
        return {"body": text, "encoding": encoding}


def _emit_exchange(
    *,
    request_file: BinaryIO | None,
    response_file: BinaryIO | None,
    request_id: str,
    status_code: int | None,
    content_type: str | None,
    response_complete: bool,
) -> None:
    try:
        request_body = _read_and_close(request_file)
        response_body = _read_and_close(response_file)
        request_raw, request_encoding = _body_text(request_body)
        response_raw, response_encoding = _body_text(response_body)
        is_stream = bool(content_type and "text/event-stream" in content_type)
        payload_logger = logging.getLogger(_PAYLOAD_LOGGER_NAME)
        payload_logger.info(
            "openai.request",
            extra={
                "rid": request_id,
                "request_id": request_id,
                "endpoint": _CHAT_COMPLETIONS_PATH,
                "payload": _json_payload(request_body),
                "payload_raw": request_raw,
                "payload_encoding": request_encoding,
            },
        )
        payload_logger.info(
            "openai.response",
            extra={
                "rid": request_id,
                "request_id": request_id,
                "endpoint": _CHAT_COMPLETIONS_PATH,
                "http_status_code": status_code,
                "content_type": content_type,
                "streaming": is_stream,
                "complete": response_complete,
                "payload": response_raw if is_stream else _json_payload(response_body),
                "payload_raw": response_raw,
                "payload_encoding": response_encoding,
            },
        )
    finally:
        _close_quietly(request_file)
        _close_quietly(response_file)


def _read_and_close(file: BinaryIO | None) -> bytes:
    if file is None:
        return b""
    try:
        file.seek(0)
        return file.read()
    finally:
        file.close()


def _close_quietly(file: BinaryIO | None) -> None:
    if file is None:
        return
    with contextlib.suppress(Exception):
        file.close()


def _background_done(task: asyncio.Task[None]) -> None:
    _BACKGROUND_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        _MODULE_LOGGER.exception("Payload logging failed in the background")


def _schedule_background(func: Callable[[], None]) -> bool:
    async def run() -> None:
        await asyncio.to_thread(func)

    try:
        task = asyncio.create_task(run())
    except Exception:
        _MODULE_LOGGER.exception("Could not schedule payload logging")
        return False
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_background_done)
    return True


def _new_spool() -> SpooledTemporaryFile[bytes] | None:
    try:
        return tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_LIMIT, mode="w+b")
    except Exception:
        _MODULE_LOGGER.exception("Could not create payload capture spool")
        return None


def _write_quietly(file: BinaryIO | None, data: bytes) -> BinaryIO | None:
    if file is None:
        return None
    try:
        file.write(data)
        return file
    except Exception:
        _MODULE_LOGGER.exception("Payload capture failed")
        _close_quietly(file)
        return None


class PayloadLoggingMiddleware:
    """Capture Chat Completions bodies without changing transmitted bytes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path") or "")
        if scope["type"] != "http" or path.rstrip("/") != _CHAT_COMPLETIONS_PATH:
            await self.app(scope, receive, send)
            return

        request_file: BinaryIO | None = _new_spool()
        response_file: BinaryIO | None = _new_spool()
        raw_headers = scope.get("headers") or []
        request_id = _header_value(raw_headers, b"x-request-id") or uuid.uuid4().hex
        status_code: int | None = None
        content_type: str | None = None
        response_complete = False

        async def receive_with_capture() -> Message:
            nonlocal request_file
            message = await receive()
            if message["type"] == "http.request" and (body := message.get("body")):
                request_file = _write_quietly(request_file, bytes(body))
            return message

        async def send_with_capture(message: Message) -> None:
            nonlocal content_type, response_complete, response_file, status_code
            await send(message)
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                content_type = _header_value(
                    message.get("headers") or [], b"content-type"
                )
            elif message["type"] == "http.response.body":
                if body := message.get("body"):
                    response_file = _write_quietly(response_file, bytes(body))
                if not message.get("more_body", False):
                    response_complete = True

        try:
            await self.app(scope, receive_with_capture, send_with_capture)
        finally:
            captured_request = request_file
            captured_response = response_file
            request_file = None
            response_file = None
            if not _schedule_background(
                lambda: _emit_exchange(
                    request_file=captured_request,
                    response_file=captured_response,
                    request_id=request_id,
                    status_code=status_code,
                    content_type=content_type,
                    response_complete=response_complete,
                )
            ):
                _close_quietly(captured_request)
                _close_quietly(captured_response)
