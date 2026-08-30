# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import logging
from collections import deque

import pytest

from vllm import payload_logging


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    def schedule(func):
        func()
        return True

    monkeypatch.setattr(payload_logging, "_schedule_background", schedule)


@pytest.mark.asyncio
async def test_streaming_payload_log_is_exact_and_omits_credentials(monkeypatch):
    _run_inline(monkeypatch)
    handler = _RecordHandler()
    logger = logging.getLogger("vllm.payload")
    monkeypatch.setattr(logger, "handlers", [handler])
    logger.setLevel(logging.INFO)

    request_parts = deque(
        [
            {
                "type": "http.request",
                "body": b'{"model":"m","messages":[',
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b'{"role":"user","content":"hello"}]}',
                "more_body": False,
            },
        ]
    )
    response_parts = (
        (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            b'"choices":[{"delta":{"reasoning":"think"}}]}\n\n'
        ),
        (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            b'"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,\n'
            b'"completion_tokens_details":{"reasoning_tokens":1},'
            b'"total_tokens":9}}\n\ndata: [DONE]\n\n'
        ),
    )
    sent = []

    async def app(scope, receive, send):
        while (message := await receive()).get("more_body", False):
            pass
        assert message["body"].endswith(b'"hello"}]}')
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_parts[0],
                "more_body": True,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_parts[1],
                "more_body": False,
            }
        )

    async def receive():
        return request_parts.popleft()

    async def send(message):
        sent.append(message)

    middleware = payload_logging.PayloadLoggingMiddleware(app)
    await middleware(
        {
            "type": "http",
            "path": "/v1/chat/completions",
            "headers": [
                (b"authorization", b"Bearer secret-token"),
                (b"x-api-key", b"secret-key"),
                (b"cookie", b"session=secret-cookie"),
                (b"x-request-id", b"request-1"),
            ],
        },
        receive,
        send,
    )

    transmitted = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert transmitted == b"".join(response_parts)
    assert [record.getMessage() for record in handler.records] == [
        "openai.request",
        "openai.response",
    ]
    request_record, response_record = handler.records
    assert request_record.payload_raw == (
        '{"model":"m","messages":[{"role":"user","content":"hello"}]}'
    )
    assert response_record.payload_raw == transmitted.decode()
    assert response_record.payload == transmitted.decode()
    assert response_record.complete is True
    assert response_record.streaming is True
    assert "[DONE]" in response_record.payload_raw
    record_text = repr(request_record.__dict__)
    assert "secret-token" not in record_text
    assert "secret-key" not in record_text
    assert "secret-cookie" not in record_text
    assert "headers" not in request_record.__dict__


@pytest.mark.asyncio
async def test_response_capture_failure_does_not_change_transmitted_body(
    monkeypatch,
):
    _run_inline(monkeypatch)

    class FailingFile(io.BytesIO):
        def write(self, data):
            raise OSError("capture failed")

    files = deque([io.BytesIO(), FailingFile()])
    monkeypatch.setattr(payload_logging, "_new_spool", files.popleft)
    sent = []
    body = b'{"id":"chatcmpl-1","choices":[],"usage":{}}'

    async def app(scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    await payload_logging.PayloadLoggingMiddleware(app)(
        {"type": "http", "path": "/v1/chat/completions", "headers": []},
        receive,
        send,
    )

    assert sent[-1]["body"] == body


def test_invalid_otel_configuration_falls_back_without_raising(monkeypatch):
    monkeypatch.setenv("VLLM_LOG_PAYLOADS", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://collector")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "invalid")
    monkeypatch.setattr(payload_logging, "_LOGGING_CONFIGURED", False)
    monkeypatch.setattr(payload_logging, "_LOGGER_PROVIDER", None)
    logger = logging.getLogger("vllm.payload")
    monkeypatch.setattr(logger, "handlers", [])

    payload_logging.configure_payload_logging()

    assert payload_logging._LOGGING_CONFIGURED is True
    assert any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    )


@pytest.mark.asyncio
async def test_non_chat_endpoint_is_not_captured(monkeypatch):
    scheduled = False

    def schedule(func):
        nonlocal scheduled
        scheduled = True
        return True

    monkeypatch.setattr(payload_logging, "_schedule_background", schedule)
    sent = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.body",
                "body": b"unchanged",
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await payload_logging.PayloadLoggingMiddleware(app)(
        {"type": "http", "path": "/v1/responses", "headers": []},
        receive,
        send,
    )

    assert sent[-1]["body"] == b"unchanged"
    assert scheduled is False
