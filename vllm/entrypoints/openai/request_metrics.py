"""Shared request counters and summaries for OpenAI-compatible endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter

request_type_image = Counter(
    name="request_type_image_total",
    documentation="Total requests containing images",
)
request_type_video = Counter(
    name="request_type_video_total",
    documentation="Total requests containing videos",
)
request_type_audio = Counter(
    name="request_type_audio_total",
    documentation="Total requests containing audio",
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
request_input_images = Counter(
    name="request_input_images_total",
    documentation="Total number of input images across requests",
)
request_input_videos = Counter(
    name="request_input_videos_total",
    documentation="Total number of input videos across requests",
)
request_input_audios = Counter(
    name="request_input_audios_total",
    documentation="Total number of input audios across requests",
)
request_input_tools = Counter(
    name="request_input_tools_total",
    documentation="Total number of declared tools across requests",
)
num_aborted_requests = Counter(
    name="num_aborted_requests_total",
    documentation="Total streaming requests aborted after client disconnects.",
)


@dataclass(frozen=True)
class RequestMetricsSummary:
    image_count: int = 0
    video_count: int = 0
    audio_count: int = 0
    tool_count: int = 0
    tool_choice: str | None = None
    structured_output_kind: str | None = None

    @property
    def has_images(self) -> bool:
        return self.image_count > 0

    @property
    def has_videos(self) -> bool:
        return self.video_count > 0

    @property
    def has_audios(self) -> bool:
        return self.audio_count > 0

    @property
    def has_tools(self) -> bool:
        return self.tool_count > 0

    @property
    def has_tool_calls_enabled(self) -> bool:
        return self.has_tools and self.tool_choice != "none"

    @property
    def has_structured_output(self) -> bool:
        return self.structured_output_kind is not None


def classify_chat_request(request: Any) -> None:
    """Classify a ChatCompletionRequest and increment counters."""
    _record_summary(summarize_chat_request(request))


def classify_completion_request(request: Any) -> None:
    """Classify a CompletionRequest and increment counters."""
    _record_summary(summarize_completion_request(request))


def classify_responses_request(request: Any) -> None:
    """Classify a ResponsesRequest and increment counters."""
    _record_summary(summarize_responses_request(request))


def summarize_request_payload(payload: Any) -> RequestMetricsSummary:
    """Summarize a raw OpenAI-compatible request payload for logging."""
    if not isinstance(payload, dict):
        return RequestMetricsSummary()
    if "messages" in payload:
        return summarize_chat_request(payload)
    if "input" in payload:
        return summarize_responses_request(payload)
    return summarize_completion_request(payload)


def summarize_chat_request(request: Any) -> RequestMetricsSummary:
    image_count = 0
    video_count = 0
    audio_count = 0
    for msg in _iter_obj_list(_get_obj_value(request, "messages")):
        content = _get_obj_value(msg, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            modality = _part_modality(part)
            if modality == "image":
                image_count += 1
            elif modality == "video":
                video_count += 1
            elif modality == "audio":
                audio_count += 1
    return RequestMetricsSummary(
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        tool_count=_count_tools(request),
        tool_choice=_normalize_tool_choice(_get_obj_value(request, "tool_choice")),
        structured_output_kind=_detect_structured_output_kind(request),
    )


def summarize_completion_request(request: Any) -> RequestMetricsSummary:
    return RequestMetricsSummary(
        tool_count=_count_tools(request),
        tool_choice=_normalize_tool_choice(_get_obj_value(request, "tool_choice")),
        structured_output_kind=_detect_structured_output_kind(request),
    )


def summarize_responses_request(request: Any) -> RequestMetricsSummary:
    image_count = 0
    video_count = 0
    audio_count = 0
    for item in _iter_obj_list(_get_obj_value(request, "input")):
        item_type = _normalize_part_type(_get_obj_value(item, "type"), item)
        if item_type in ("input_image", "image_url"):
            image_count += 1
            continue
        if item_type in ("input_audio", "audio_url"):
            audio_count += 1
            continue
        if item_type == "video_url":
            video_count += 1
            continue
        content = _get_obj_value(item, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            modality = _part_modality(part)
            if modality == "image":
                image_count += 1
            elif modality == "video":
                video_count += 1
            elif modality == "audio":
                audio_count += 1
    return RequestMetricsSummary(
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        tool_count=_count_tools(request),
        tool_choice=_normalize_tool_choice(_get_obj_value(request, "tool_choice")),
        structured_output_kind=_detect_structured_output_kind(request),
    )


def record_aborted_request() -> None:
    """Record a client-aborted request for OTEL/prometheus export."""
    num_aborted_requests.inc()


def _record_summary(summary: RequestMetricsSummary) -> None:
    if summary.has_images:
        request_type_image.inc()
        request_input_images.inc(summary.image_count)
    if summary.has_videos:
        request_type_video.inc()
        request_input_videos.inc(summary.video_count)
    if summary.has_audios:
        request_type_audio.inc()
        request_input_audios.inc(summary.audio_count)
    if summary.has_tools:
        request_input_tools.inc(summary.tool_count)
    if summary.has_tool_calls_enabled:
        request_type_tool_call.inc()
    if summary.has_structured_output:
        request_type_structured_output.inc()


def _count_tools(request: Any) -> int:
    tools = _get_obj_value(request, "tools")
    if not isinstance(tools, list):
        return 0
    return len(tools)


def _iter_obj_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _get_obj_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_part_type(part_type: Any, part: Any) -> str:
    if isinstance(part_type, str) and part_type:
        return part_type
    if not isinstance(part, dict):
        return ""
    if "image_url" in part or "image_pil" in part or "image_embeds" in part:
        return "image_url"
    if "video_url" in part:
        return "video_url"
    if "audio_url" in part or "input_audio" in part or "audio_embeds" in part:
        return "audio_url"
    return ""


def _part_modality(part: Any) -> str | None:
    part_type = _normalize_part_type(_get_obj_value(part, "type"), part)
    if part_type in ("image_url", "image_pil", "image_embeds", "input_image"):
        return "image"
    if part_type == "video_url":
        return "video"
    if part_type in ("audio_url", "input_audio", "audio_embeds"):
        return "audio"
    return None


def _normalize_tool_choice(tool_choice: Any) -> str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return "named"
        choice_type = tool_choice.get("type")
        return str(choice_type) if choice_type is not None else "named"
    choice_type = getattr(tool_choice, "type", None)
    function = getattr(tool_choice, "function", None)
    function_name = _get_obj_value(function, "name")
    if function_name:
        return "named"
    if isinstance(choice_type, str) and choice_type:
        return choice_type
    return str(tool_choice)


def _detect_structured_output_kind(request: Any) -> str | None:
    response_format = _get_obj_value(request, "response_format")
    response_format_type = _get_obj_value(response_format, "type")
    if response_format_type in ("json_schema", "json_object", "structural_tag"):
        return str(response_format_type)

    structured_outputs = _get_obj_value(request, "structured_outputs")
    if isinstance(structured_outputs, dict):
        for key in (
            "json",
            "json_object",
            "json_schema",
            "structural_tag",
            "regex",
            "choice",
            "grammar",
        ):
            if structured_outputs.get(key) is not None:
                return "json_schema" if key == "json" else key
        return "structured_outputs"
    if structured_outputs is not None:
        return "structured_outputs"

    text = _get_obj_value(request, "text")
    text_format = _get_obj_value(text, "format")
    text_format_type = _get_obj_value(text_format, "type")
    if text_format_type in ("json_schema", "json_object"):
        return str(text_format_type)

    return None
