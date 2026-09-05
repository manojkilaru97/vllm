# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import threading
import time
from typing import Any

_LOGGER = logging.getLogger("vllm.priority")
_DEFAULT_LIMIT = 200_000
_MAX_LIMIT = 1_000_000
_MAX_CANDIDATES = 64
_lock = threading.Lock()
_emitted = 0


def is_enabled() -> bool:
    return os.getenv("VLLM_PRIORITY_TRACE", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _limit() -> int:
    try:
        configured = int(os.getenv("VLLM_PRIORITY_TRACE_LIMIT", _DEFAULT_LIMIT))
    except ValueError:
        configured = _DEFAULT_LIMIT
    return min(max(configured, 0), _MAX_LIMIT)


def emit(
    stage: str,
    *,
    request_id: str,
    priority: int,
    scheduler_id: int | None = None,
    scheduler_step: int | None = None,
    scheduled_tokens: int | None = None,
    eligible_candidates: list[str] | None = None,
    x_request_id: str | None = None,
) -> None:
    """Emit one bounded scheduler observation without request payloads."""
    global _emitted
    if not is_enabled():
        return

    with _lock:
        if _emitted >= _limit():
            return
        _emitted += 1
        sequence = _emitted

    candidates = (eligible_candidates or [])[:_MAX_CANDIDATES]
    record: dict[str, Any] = {
        "schema": "vllm.priority.v1",
        "stage": stage,
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "process_id": os.getpid(),
        "request_id": request_id,
        "priority": priority,
    }
    if scheduler_step is not None:
        record["scheduler_step"] = scheduler_step
    if scheduler_id is not None:
        record["scheduler_id"] = scheduler_id
    if scheduled_tokens is not None:
        record["scheduled_tokens"] = scheduled_tokens
    if eligible_candidates is not None:
        record["eligible_candidates"] = candidates
        record["eligible_candidate_count"] = len(eligible_candidates)
        record["candidates_truncated"] = len(eligible_candidates) > _MAX_CANDIDATES
    if x_request_id is not None:
        record["x_request_id"] = x_request_id[:128]
    _LOGGER.info("priority_trace %s", json.dumps(record, separators=(",", ":")))


def _reset_for_test() -> None:
    global _emitted
    with _lock:
        _emitted = 0
