"""Shared request counters and safe failure observability helpers.

This module intentionally avoids logging prompt/completion content. It emits
request-shape metadata only: sampling params, message roles, multimodal counts,
tool/structured-output usage, and a stable request-shape hash for searching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter as PyCounter
from collections import deque
from collections.abc import Mapping
from typing import Any

from prometheus_client import Counter

payload_logger = logging.getLogger("vllm.payload")

request_type_image = Counter(
    name="request_type_image_total",
    documentation="Total requests containing images",
)
request_type_video = Counter(
    name="request_type_video_total",
    documentation="Total requests containing videos",
)
request_type_tool_call = Counter(
    name="request_type_tool_call_total",
    documentation="Total requests with tool calls enabled",
)
request_type_structured_output = Counter(
    name="request_type_structured_output_total",
    documentation="Total requests with structured output "
    "(json_schema, json_object, structural_tag, regex, choice, or grammar)",
)
request_failures_total = Counter(
    name="request_failures_total",
    documentation="Total failed OpenAI-compatible requests by class",
    labelnames=("endpoint", "failure_class", "status_code", "probe"),
)
_RECENT_FAILURES: deque[dict[str, Any]] = deque(maxlen=512)


def _normalize_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        try:
            dumped = payload.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return None
    if isinstance(payload, dict):
        return payload
    return None


def _tool_choice_shape(tool_choice: Any) -> str:
    if tool_choice is None:
        return "none"
    if isinstance(tool_choice, str):
        return tool_choice
    if hasattr(tool_choice, "model_dump"):
        try:
            tool_choice = tool_choice.model_dump()
        except Exception:
            return "object"
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return f"named:{fn['name']}"
        return "object"
    return "object"


def _structured_output_shape(payload: Mapping[str, Any]) -> str:
    response_format = payload.get("response_format")
    if hasattr(response_format, "model_dump"):
        try:
            response_format = response_format.model_dump()
        except Exception:
            response_format = None
    if isinstance(response_format, dict):
        if isinstance(response_format.get("type"), str):
            return response_format["type"]
    structured_outputs = payload.get("structured_outputs")
    if hasattr(structured_outputs, "model_dump"):
        try:
            structured_outputs = structured_outputs.model_dump()
        except Exception:
            structured_outputs = None
    if isinstance(structured_outputs, dict):
        for key in (
            "json",
            "json_object",
            "regex",
            "choice",
            "grammar",
            "structural_tag",
        ):
            if structured_outputs.get(key) is not None:
                return key
        return "structured_outputs"
    if structured_outputs is not None:
        return "structured_outputs"
    return "none"


def summarize_chat_request(payload: Any) -> dict[str, Any]:
    data = _normalize_payload(payload) or {}
    messages = data.get("messages")
    role_counts: PyCounter[str] = PyCounter()
    part_counts: PyCounter[str] = PyCounter()
    image_count = 0
    video_count = 0

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "unknown"))
            role_counts[role] += 1
            content = msg.get("content")
            if isinstance(content, str):
                part_counts["text"] += 1
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type", "unknown"))
                    part_counts[part_type] += 1
                    if part_type == "image_url":
                        image_count += 1
                    elif part_type == "video_url":
                        video_count += 1

    tools = data.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    chat_kwargs = data.get("chat_template_kwargs")
    if hasattr(chat_kwargs, "model_dump"):
        try:
            chat_kwargs = chat_kwargs.model_dump()
        except Exception:
            chat_kwargs = None

    summary: dict[str, Any] = {
        "model": data.get("model"),
        "messages_count": len(messages) if isinstance(messages, list) else 0,
        "message_roles": dict(sorted(role_counts.items())),
        "content_types": dict(sorted(part_counts.items())),
        "images": image_count,
        "videos": video_count,
        "tools": tool_count,
        "tool_choice": _tool_choice_shape(data.get("tool_choice")),
        "parallel_tool_calls": bool(data.get("parallel_tool_calls"))
        if data.get("parallel_tool_calls") is not None
        else None,
        "structured_output": _structured_output_shape(data),
        "stream": bool(data.get("stream", False)),
        "sampling": {
            "temperature": data.get("temperature"),
            "top_p": data.get("top_p"),
            "top_k": data.get("top_k"),
            "min_p": data.get("min_p"),
            "max_tokens": data.get("max_tokens"),
            "max_completion_tokens": data.get("max_completion_tokens"),
            "n": data.get("n"),
            "presence_penalty": data.get("presence_penalty"),
            "frequency_penalty": data.get("frequency_penalty"),
            "repetition_penalty": data.get("repetition_penalty"),
            "seed": data.get("seed"),
            "ignore_eos": data.get("ignore_eos"),
        },
        "chat_template_kwargs_keys": (
            sorted(chat_kwargs.keys()) if isinstance(chat_kwargs, dict) else []
        ),
        "thinking": (
            chat_kwargs.get("enable_thinking")
            if isinstance(chat_kwargs, dict)
            else None
        ),
    }
    summary["shape_hash"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return summary


def classify_failure(error: str, *, exc_type: str = "", status_code: int = 0) -> str:
    msg = (error or "").lower()
    exc = (exc_type or "").lower()

    if "enginedeaderror" in exc:
        return "engine_dead_error"
    if "enginegenerateerror" in exc:
        return "engine_generate_error"
    if "requestvalidationerror" in exc:
        return "request_validation_error"
    if "templateerror" in exc and "system message must be at the beginning" in msg:
        return "chat_template_system_not_first"
    if "no user query found in messages" in msg:
        return "chat_template_no_user"
    if "system message must be at the beginning" in msg:
        return "chat_template_system_not_first"
    if "tool parser" in msg:
        return "tool_parser_error"
    if "reasoning parser" in msg:
        return "reasoning_parser_error"
    if "resolve nvcf" in msg or "asset" in msg:
        return "asset_resolution_error"
    if "model `" in msg and "does not exist" in msg:
        return "model_not_found"
    if "internal server error" in msg or status_code >= 500:
        return "internal_server_error"
    if status_code == 404:
        return "not_found"
    if status_code == 400:
        return "bad_request"
    return exc or "unknown_error"


def log_request_failure(
    *,
    endpoint: str,
    request_id: str,
    error: str,
    status_code: int,
    payload: Any = None,
    probe: bool = False,
    exc_type: str = "",
    error_type: str = "",
    path: str = "",
    method: str = "",
) -> None:
    failure_class = classify_failure(
        error, exc_type=exc_type or error_type, status_code=status_code
    )
    request_failures_total.labels(
        endpoint=endpoint,
        failure_class=failure_class,
        status_code=str(status_code),
        probe="true" if probe else "false",
    ).inc()

    summary = summarize_chat_request(payload) if payload is not None else {}
    _RECENT_FAILURES.append(
        {
            "ts": time.time(),
            "rid": request_id or "",
            "endpoint": endpoint,
            "failure_class": failure_class,
            "status_code": status_code,
            "probe": probe,
            "shape_hash": summary.get("shape_hash", ""),
            "error_type": error_type or exc_type or "",
        }
    )
    try:
        payload_logger.warning(
            "openai.request_failure",
            extra={
                "rid": request_id or "",
                "endpoint": endpoint,
                "path": path,
                "method": method,
                "probe": probe,
                "status_code": status_code,
                "failure_class": failure_class,
                "error_type": error_type or exc_type or "",
                "error": error,
                "request_shape": summary,
                "shape_hash": summary.get("shape_hash", ""),
            },
        )
    except Exception:
        return


def recent_failure_summary(*, window_s: float = 1800.0, limit: int = 12) -> dict[str, Any]:
    cutoff = time.time() - window_s
    recent = [item for item in _RECENT_FAILURES if item["ts"] >= cutoff]
    non_probe_recent = [item for item in recent if not item.get("probe")]
    counts = PyCounter(item["failure_class"] for item in non_probe_recent)
    latest = non_probe_recent[-limit:]
    return {
        "window_s": window_s,
        "total_recent_failures": len(non_probe_recent),
        "failure_classes": dict(sorted(counts.items())),
        "recent_items": [
            {
                "ts": item["ts"],
                "rid": item["rid"],
                "endpoint": item["endpoint"],
                "failure_class": item["failure_class"],
                "status_code": item["status_code"],
                "shape_hash": item.get("shape_hash", ""),
                "error_type": item.get("error_type", ""),
            }
            for item in latest
        ],
    }


def classify_chat_request(request) -> None:
    """Classify a ChatCompletionRequest and increment counters."""
    summary = summarize_chat_request(request)
    if summary["images"] > 0:
        request_type_image.inc()
    if summary["videos"] > 0:
        request_type_video.inc()
    if summary["tools"] > 0 and summary["tool_choice"] != "none":
        request_type_tool_call.inc()
    if summary["structured_output"] != "none":
        request_type_structured_output.inc()


def classify_completion_request(request) -> None:
    """Classify a CompletionRequest and increment counters."""
    _classify_structured_output_completion(request)


def classify_responses_request(request) -> None:
    """Classify a ResponsesRequest and increment counters."""
    _classify_responses_images(request)
    if getattr(request, "tools", None) and getattr(request, "tool_choice", None) != "none":
        request_type_tool_call.inc()
    text = getattr(request, "text", None)
    if text is not None:
        fmt = getattr(text, "format", None)
        if fmt is not None and hasattr(fmt, "type"):
            if fmt.type in ("json_schema", "json_object"):
                request_type_structured_output.inc()


def _classify_responses_images(request) -> None:
    input_items = getattr(request, "input", None)
    if not isinstance(input_items, list):
        return
    for item in input_items:
        item_type = (
            item.get("type", "") if isinstance(item, dict) else getattr(item, "type", "")
        )
        if item_type == "input_image":
            request_type_image.inc()
            return
        content = (
            item.get("content", None)
            if isinstance(item, dict)
            else getattr(item, "content", None)
        )
        if isinstance(content, list):
            for part in content:
                part_type = (
                    part.get("type", "")
                    if isinstance(part, dict)
                    else getattr(part, "type", "")
                )
                if part_type in ("input_image", "image_url"):
                    request_type_image.inc()
                    return


def _classify_structured_output_completion(request) -> None:
    response_format = getattr(request, "response_format", None)
    if (
        response_format is not None
        and hasattr(response_format, "type")
        and response_format.type
        in ("json_schema", "json_object", "structural_tag")
    ):
        request_type_structured_output.inc()
        return
    if getattr(request, "structured_outputs", None) is not None:
        request_type_structured_output.inc()
