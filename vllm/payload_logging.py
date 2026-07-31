# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

_ASYNC_ENV = "VLLM_PAYLOAD_LOG_ASYNC"
_MAX_QUEUE_ENV = "VLLM_PAYLOAD_LOG_MAX_QUEUE"
_DEFAULT_MAX_QUEUE = 8192
_MAX_RESPONSE_CHARS_ENV = "VLLM_PAYLOAD_LOG_MAX_RESPONSE_CHARS"
_DEFAULT_MAX_RESPONSE_CHARS = 16 * 1024 * 1024


def _max_queue_size() -> int:
    try:
        value = int(os.getenv(_MAX_QUEUE_ENV, str(_DEFAULT_MAX_QUEUE)))
    except (TypeError, ValueError):
        value = _DEFAULT_MAX_QUEUE
    return max(1, value)


_QUEUE: queue.Queue[
    tuple[logging.Logger, str, dict[str, Any] | Callable[[], dict[str, Any] | None]]
] = queue.Queue(maxsize=_max_queue_size())
_START_LOCK = threading.Lock()
_STARTED = False
_DROP_COUNT = 0
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def is_payload_logging_enabled() -> bool:
    """Return whether payload logging was explicitly enabled."""
    return os.getenv("VLLM_LOG_PAYLOADS", "0") == "1"


def payload_log_max_response_chars() -> int:
    """Return the bounded per-response payload accumulation budget."""
    try:
        value = int(
            os.getenv(
                _MAX_RESPONSE_CHARS_ENV,
                str(_DEFAULT_MAX_RESPONSE_CHARS),
            )
        )
    except (TypeError, ValueError):
        value = _DEFAULT_MAX_RESPONSE_CHARS
    return max(1, value)


def _async_enabled() -> bool:
    return os.getenv(_ASYNC_ENV, "1") != "0"


def _emit(
    payload_logger: logging.Logger,
    event: str,
    extra_or_builder: dict[str, Any] | Callable[[], dict[str, Any] | None],
) -> None:
    try:
        extra = extra_or_builder() if callable(extra_or_builder) else extra_or_builder
        if extra is not None:
            payload_logger.info(event, extra=extra)
    except Exception:
        logger.exception("Failed to emit payload log event=%s", event)


def _worker() -> None:
    while True:
        payload_logger, event, extra_or_builder = _QUEUE.get()
        _emit(payload_logger, event, extra_or_builder)


def _ensure_worker() -> bool:
    global _STARTED
    if _STARTED:
        return True
    with _START_LOCK:
        if _STARTED:
            return True
        thread = threading.Thread(
            target=_worker,
            name="vllm-payload-log-worker",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            return False
        _STARTED = True
        return True


def _enqueue(
    payload_logger: logging.Logger,
    event: str,
    extra_or_builder: dict[str, Any] | Callable[[], dict[str, Any] | None],
) -> None:
    global _DROP_COUNT
    try:
        _QUEUE.put_nowait((payload_logger, event, extra_or_builder))
    except queue.Full:
        _DROP_COUNT += 1
        if _DROP_COUNT == 1 or (_DROP_COUNT & (_DROP_COUNT - 1)) == 0:
            logger.warning(
                "Dropping payload log events because %s is full; dropped=%d",
                _MAX_QUEUE_ENV,
                _DROP_COUNT,
            )


def _flush_at_exit() -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            item = _QUEUE.get_nowait()
        except Exception:
            return
        _emit(*item)


atexit.register(_flush_at_exit)


def log_payload(
    payload_logger: logging.Logger,
    event: str,
    *,
    extra: dict[str, Any],
) -> None:
    """Emit a payload log record without blocking request handling on handlers."""
    if not _async_enabled():
        _emit(payload_logger, event, extra)
        return
    if not _ensure_worker():
        return
    _enqueue(payload_logger, event, extra)


def log_payload_lazy(
    payload_logger: logging.Logger,
    event: str,
    *,
    build_extra: Callable[[], dict[str, Any] | None],
) -> None:
    """Build and emit a payload log record off the request path."""
    if not _async_enabled():
        _emit(payload_logger, event, build_extra)
        return
    if not _ensure_worker():
        return
    _enqueue(payload_logger, event, build_extra)


def schedule_payload_log_task(
    coro: Coroutine[Any, Any, None],
    *,
    label: str,
    rid: str,
) -> None:
    """Run request-body capture outside the TTFT path and surface failures."""
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:
        try:
            coro.close()
        except Exception:
            pass
        logger.exception(
            "Failed to schedule payload log task label=%s rid=%s",
            label,
            rid,
        )
        return

    _BACKGROUND_TASKS.add(task)

    def _done(completed: asyncio.Task[Any]) -> None:
        _BACKGROUND_TASKS.discard(completed)
        try:
            exc = completed.exception()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(
                "Failed to inspect payload log task label=%s rid=%s",
                label,
                rid,
            )
            return
        if exc is not None:
            logger.error(
                "Payload log task failed label=%s rid=%s",
                label,
                rid,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(_done)
