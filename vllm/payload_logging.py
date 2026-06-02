# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_ASYNC_ENV = "VLLM_PAYLOAD_LOG_ASYNC"
_QUEUE: queue.SimpleQueue[
    tuple[logging.Logger, str, dict[str, Any] | Callable[[], dict[str, Any] | None]]
] = queue.SimpleQueue()
_START_LOCK = threading.Lock()
_STARTED = False


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


def _ensure_worker() -> None:
    global _STARTED
    if _STARTED:
        return
    with _START_LOCK:
        if _STARTED:
            return
        thread = threading.Thread(
            target=_worker,
            name="vllm-payload-log-worker",
            daemon=True,
        )
        thread.start()
        _STARTED = True


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
    _ensure_worker()
    _QUEUE.put((payload_logger, event, extra))


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
    _ensure_worker()
    _QUEUE.put((payload_logger, event, build_extra))
