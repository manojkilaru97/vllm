# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

SUPPRESS_PAYLOAD_NCA_IDS_ENV = "VLLM_SUPPRESS_PAYLOAD_NCA_IDS"
PAYLOAD_SUPPRESSION_REASON_NCA_ID = "nca_id"

_NCA_HEADER_NAMES = ("nvcf-ncaid", "nvcf-nca-id")
_MAX_TRACKED_REQUESTS = 32768
_TRACKED_REQUESTS: OrderedDict[str, PayloadSuppressionContext] = OrderedDict()
_REQUEST_CONTEXTS: OrderedDict[str, RequestLoggingContext] = OrderedDict()
_TRACKED_REQUESTS_LOCK = threading.RLock()

_SAFE_REQUEST_FIELDS = {
    "best_of",
    "echo",
    "frequency_penalty",
    "ignore_eos",
    "include_reasoning",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "min_p",
    "min_tokens",
    "model",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_budget",
    "reasoning_effort",
    "repetition_penalty",
    "seed",
    "service_tier",
    "stream",
    "temperature",
    "top_k",
    "top_logprobs",
    "top_p",
    "truncate_prompt_tokens",
    "use_beam_search",
}
_SAFE_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class PayloadSuppressionContext:
    nca_id: str
    reason: str = PAYLOAD_SUPPRESSION_REASON_NCA_ID


@dataclass(frozen=True)
class RequestLoggingContext:
    nca_id: str = ""
    payload_suppressed: bool = False
    suppression_reason: str | None = None
    model: str | None = None
    stream: bool | None = None


def normalize_nca_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def resolve_nca_id(headers: Mapping[str, Any] | None) -> str:
    if headers is None:
        return ""

    for header_name in _NCA_HEADER_NAMES:
        value = _get_header_value(headers, header_name)
        nca_id = normalize_nca_id(value)
        if nca_id:
            return nca_id
    return ""


def payload_suppression_context_from_headers(
    headers: Mapping[str, Any] | None,
) -> PayloadSuppressionContext | None:
    nca_id = resolve_nca_id(headers)
    if not nca_id:
        return None
    if nca_id not in _suppressed_nca_ids():
        return None
    return PayloadSuppressionContext(nca_id=nca_id)


def request_logging_context_from_headers(
    headers: Mapping[str, Any] | None,
    *,
    model: str | None = None,
    stream: bool | None = None,
) -> RequestLoggingContext:
    nca_id = resolve_nca_id(headers)
    payload_suppressed = bool(nca_id and nca_id in _suppressed_nca_ids())
    return RequestLoggingContext(
        nca_id=nca_id,
        payload_suppressed=payload_suppressed,
        suppression_reason=PAYLOAD_SUPPRESSION_REASON_NCA_ID
        if payload_suppressed
        else None,
        model=model,
        stream=stream,
    )


def register_request_logging_context(
    request_id: str | None,
    context: RequestLoggingContext,
) -> RequestLoggingContext:
    if not request_id:
        return context

    with _TRACKED_REQUESTS_LOCK:
        request_key = str(request_id).strip()
        if request_key:
            _REQUEST_CONTEXTS[request_key] = context
            _REQUEST_CONTEXTS.move_to_end(request_key)
        while len(_REQUEST_CONTEXTS) > _MAX_TRACKED_REQUESTS:
            _REQUEST_CONTEXTS.popitem(last=False)

    if context.payload_suppressed:
        mark_payload_suppressed(
            request_id,
            PayloadSuppressionContext(
                nca_id=context.nca_id,
                reason=context.suppression_reason or PAYLOAD_SUPPRESSION_REASON_NCA_ID,
            ),
        )
    return context


def mark_payload_suppressed(
    request_id: str | None,
    context: PayloadSuppressionContext | None,
) -> None:
    if not request_id or context is None:
        return

    with _TRACKED_REQUESTS_LOCK:
        request_key = str(request_id).strip()
        if request_key:
            _TRACKED_REQUESTS[request_key] = context
            _TRACKED_REQUESTS.move_to_end(request_key)
        while len(_TRACKED_REQUESTS) > _MAX_TRACKED_REQUESTS:
            _TRACKED_REQUESTS.popitem(last=False)


def payload_suppression_context_for_request_id(
    request_id: str | None,
) -> PayloadSuppressionContext | None:
    if not request_id:
        return None

    with _TRACKED_REQUESTS_LOCK:
        for candidate in _request_id_candidates(request_id):
            request_context = _REQUEST_CONTEXTS.get(candidate)
            if request_context is not None:
                _REQUEST_CONTEXTS.move_to_end(candidate)
                if request_context.payload_suppressed:
                    return PayloadSuppressionContext(
                        nca_id=request_context.nca_id,
                        reason=request_context.suppression_reason
                        or PAYLOAD_SUPPRESSION_REASON_NCA_ID,
                    )
                return None
        for candidate in _request_id_candidates(request_id):
            context = _TRACKED_REQUESTS.get(candidate)
            if context is not None:
                _TRACKED_REQUESTS.move_to_end(candidate)
                return context
    return None


def request_logging_context_for_request_id(
    request_id: str | None,
) -> RequestLoggingContext | None:
    if not request_id:
        return None

    with _TRACKED_REQUESTS_LOCK:
        for candidate in _request_id_candidates(request_id):
            context = _REQUEST_CONTEXTS.get(candidate)
            if context is not None:
                _REQUEST_CONTEXTS.move_to_end(candidate)
                return context
    return None


def build_suppressed_request_payload(
    payload: Any,
    context: PayloadSuppressionContext,
) -> dict[str, Any]:
    safe_payload = _base_suppressed_payload(context)
    if isinstance(payload, Mapping):
        for key in _SAFE_REQUEST_FIELDS:
            if key in payload:
                safe_payload[key] = _safe_value(payload[key])
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, Mapping) and "effort" in reasoning:
            safe_payload["reasoning_effort"] = _safe_value(reasoning.get("effort"))
    else:
        for key in _SAFE_REQUEST_FIELDS:
            if hasattr(payload, key):
                safe_payload[key] = _safe_value(getattr(payload, key))
        reasoning = getattr(payload, "reasoning", None)
        if isinstance(reasoning, Mapping) and "effort" in reasoning:
            safe_payload["reasoning_effort"] = _safe_value(reasoning.get("effort"))
    return safe_payload


def build_suppressed_response_payload(
    payload: Mapping[str, Any] | None,
    context: PayloadSuppressionContext,
) -> dict[str, Any]:
    safe_payload = _base_suppressed_payload(context)
    if not isinstance(payload, Mapping):
        return safe_payload

    for key in ("created", "id", "model", "object", "status", "stream", "usage"):
        if key in payload:
            safe_payload[key] = _safe_value(payload[key])

    choices = payload.get("choices")
    if isinstance(choices, list):
        finish_reasons = [
            _safe_value(choice.get("finish_reason"))
            for choice in choices
            if isinstance(choice, Mapping) and choice.get("finish_reason") is not None
        ]
        if finish_reasons:
            safe_payload["finish_reasons"] = finish_reasons
            safe_payload["finish_reason"] = finish_reasons[0]
    return safe_payload


def build_suppressed_response_payload_from_obj(
    response: Any,
    context: PayloadSuppressionContext,
) -> dict[str, Any]:
    safe_payload = _base_suppressed_payload(context)
    for key in ("created", "id", "model", "object", "status"):
        if hasattr(response, key):
            safe_payload[key] = _safe_value(getattr(response, key))
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            with suppress(Exception):
                usage = usage.model_dump(mode="json")
        safe_payload["usage"] = _safe_value(usage)
    return safe_payload


def build_suppressed_error_payload(
    err_payload: Any,
    context: PayloadSuppressionContext,
) -> dict[str, Any]:
    safe_payload = _base_suppressed_payload(context)
    error = err_payload.get("error") if isinstance(err_payload, Mapping) else None
    if error is None:
        error = getattr(err_payload, "error", None)
    if isinstance(error, Mapping):
        safe_payload["error"] = {
            key: _safe_value(error.get(key))
            for key in ("type", "code")
            if error.get(key) is not None
        }
        return safe_payload
    if error is not None:
        error_payload = {
            key: _safe_value(getattr(error, key))
            for key in ("type", "code")
            if getattr(error, key, None) is not None
        }
        if error_payload:
            safe_payload["error"] = error_payload
    return safe_payload


def _base_suppressed_payload(
    context: PayloadSuppressionContext,
) -> dict[str, Any]:
    return {
        "payload_suppressed": True,
        "suppression_reason": context.reason,
        "nca_id": context.nca_id,
    }


def _suppressed_nca_ids() -> set[str]:
    raw = os.getenv(SUPPRESS_PAYLOAD_NCA_IDS_ENV, "")
    return {
        nca_id
        for nca_id in (normalize_nca_id(part) for part in raw.split(","))
        if nca_id
    }


def _get_header_value(headers: Mapping[str, Any], name: str) -> Any:
    try:
        value = headers.get(name)
    except Exception:
        value = None
    if value:
        return value

    needle = name.lower()
    try:
        items = headers.items()
    except Exception:
        return None
    for key, item in items:
        if str(key).lower() == needle:
            return item
    return None


def _request_id_candidates(request_id: str) -> list[str]:
    raw = str(request_id).strip()
    if not raw:
        return []

    candidates: list[str] = []

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(raw)
    if raw.startswith("chatcmpl-"):
        add(raw[len("chatcmpl-") :])
    else:
        add(f"chatcmpl-{raw}")

    for candidate in list(candidates):
        base, separator, suffix = candidate.rpartition("_")
        if separator and suffix.isdigit():
            add(base)
            if base.startswith("chatcmpl-"):
                add(base[len("chatcmpl-") :])
            else:
                add(f"chatcmpl-{base}")
    return candidates


def _safe_value(value: Any) -> Any:
    if isinstance(value, _SAFE_SCALAR_TYPES):
        return value
    if isinstance(value, list):
        return [_safe_value(item) for item in value if _is_safe_nested_value(item)]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if _is_safe_nested_value(item)
        }
    return str(value)


def _is_safe_nested_value(value: Any) -> bool:
    if isinstance(value, _SAFE_SCALAR_TYPES):
        return True
    if isinstance(value, list):
        return all(_is_safe_nested_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(_is_safe_nested_value(item) for item in value.values())
    return False
