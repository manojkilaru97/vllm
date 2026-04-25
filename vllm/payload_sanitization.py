# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from typing import Any

from vllm import envs

MM_INPUT_REDACTED = "[redacted-mm-input]"

_AUDIO_CONTAINER_KEYS = {
    "audio_url",
    "input_audio",
}
_VISUAL_CONTAINER_KEYS = {
    "image_url",
    "input_image",
    "input_video",
    "video_url",
}
_MM_CONTAINER_KEYS = _AUDIO_CONTAINER_KEYS | _VISUAL_CONTAINER_KEYS
_MM_SCALAR_KEYS = {
    "asset_id",
    "b64_json",
    "data",
    "file_id",
    "url",
}
_HTML_MM_SRC_RE = re.compile(
    r"(<(?P<tag>img|video|audio)\b[^>]*\bsrc\s*=\s*[\"'])(?P<src>[^\"']+)([\"'])",
    re.IGNORECASE,
)
_DATA_URI_RE = re.compile(
    r"data:(?P<mime>(?:image|video|audio)/[^,;]+)(?:;[^,]*)?,[^\s\"'<>]+",
    re.IGNORECASE,
)
_ASSET_ID_URI_RE = re.compile(
    r"(data:(?:image|video|audio)/[^;]+;asset_id,)([^\s\"'<>]+)",
    re.IGNORECASE,
)
_ASSET_ID_TEXT_RE = re.compile(r"(asset_id:)([^\s\"'<>]+)", re.IGNORECASE)


def log_mm_input_metadata_enabled() -> bool:
    return bool(envs.VLLM_LOG_MM_INPUT_METADATA)


def log_audio_input_metadata_enabled() -> bool:
    return log_mm_input_metadata_enabled() and bool(
        envs.VLLM_LOG_AUDIO_INPUT_METADATA
    )


def prepare_request_payload_for_logging(
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    allowed_local_media_path: str = "",
) -> Any:
    del headers, allowed_local_media_path
    if payload is None:
        return payload
    return maybe_redact_mm_payload(_normalize_logging_payload(payload))


def maybe_redact_mm_payload(payload: Any) -> Any:
    if log_mm_input_metadata_enabled() and log_audio_input_metadata_enabled():
        return payload
    return redact_mm_input_metadata(payload)


def maybe_redact_mm_text(text: str) -> str:
    if log_mm_input_metadata_enabled() and log_audio_input_metadata_enabled():
        return text
    return sanitize_mm_text(text)


def sanitize_mm_text(text: str) -> str:
    if not text:
        return text

    redact_visual = not log_mm_input_metadata_enabled()
    redact_audio = not log_audio_input_metadata_enabled()
    if not redact_visual and not redact_audio:
        return text

    def _replace_html_src(match: re.Match[str]) -> str:
        source = match.group("src") or ""
        if _should_redact_text_media(
            tag=match.group("tag") or "",
            source=source,
            redact_visual=redact_visual,
            redact_audio=redact_audio,
        ):
            return f"{match.group(1)}{MM_INPUT_REDACTED}{match.group(4)}"
        return match.group(0)

    def _replace_data_uri(match: re.Match[str]) -> str:
        mime = (match.group("mime") or "").lower()
        if _should_redact_mime(
            mime, redact_visual=redact_visual, redact_audio=redact_audio
        ):
            return MM_INPUT_REDACTED
        return match.group(0)

    def _replace_asset_uri(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        mime = prefix[5:].split(";", 1)[0].lower() if prefix.startswith("data:") else ""
        if _should_redact_mime(
            mime, redact_visual=redact_visual, redact_audio=redact_audio
        ):
            return f"{prefix}{MM_INPUT_REDACTED}"
        return match.group(0)

    sanitized = _HTML_MM_SRC_RE.sub(_replace_html_src, text)
    sanitized = _ASSET_ID_URI_RE.sub(_replace_asset_uri, sanitized)
    if redact_visual:
        sanitized = _ASSET_ID_TEXT_RE.sub(rf"\1{MM_INPUT_REDACTED}", sanitized)
    return _DATA_URI_RE.sub(_replace_data_uri, sanitized)


def redact_mm_input_metadata(payload: Any) -> Any:
    return _redact(payload, parent_path=())


def _normalize_logging_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            _normalize_logging_payload(key): _normalize_logging_payload(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_logging_payload(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _normalize_logging_payload(value.model_dump(mode="json"))
        except Exception:
            pass

    if isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, str)
    ):
        try:
            return [_normalize_logging_payload(item) for item in value]
        except Exception:
            pass

    try:
        return copy.deepcopy(value)
    except Exception:
        return str(value)


def _redact(value: Any, *, parent_path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            next_path = parent_path + (key_str,)
            if key_str in _MM_CONTAINER_KEYS and _container_requires_redaction(key_str):
                redacted[key] = _redact_mm_container(item, container_key=key_str)
            elif key_str in _MM_SCALAR_KEYS and _path_has_redacted_mm_container(
                parent_path
            ):
                redacted[key] = MM_INPUT_REDACTED
            else:
                redacted[key] = _redact(item, parent_path=next_path)
        return redacted

    if isinstance(value, list):
        return [_redact(item, parent_path=parent_path) for item in value]

    if isinstance(value, str):
        if _path_has_redacted_mm_container(parent_path):
            return MM_INPUT_REDACTED
        return sanitize_mm_text(value)

    return value


def _redact_mm_container(value: Any, *, container_key: str) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in _MM_SCALAR_KEYS:
                redacted[key] = MM_INPUT_REDACTED
            else:
                redacted[key] = _redact(item, parent_path=(container_key, key_str))
        return redacted

    if isinstance(value, list):
        return [_redact(item, parent_path=(container_key,)) for item in value]

    if isinstance(value, str):
        return MM_INPUT_REDACTED

    return value


def _path_has_redacted_mm_container(parent_path: tuple[str, ...]) -> bool:
    for part in reversed(parent_path):
        if part in _MM_CONTAINER_KEYS:
            return _container_requires_redaction(part)
    return False


def _container_requires_redaction(container_key: str) -> bool:
    if container_key in _AUDIO_CONTAINER_KEYS:
        return not log_audio_input_metadata_enabled()
    if container_key in _VISUAL_CONTAINER_KEYS:
        return not log_mm_input_metadata_enabled()
    return False


def _should_redact_text_media(
    *,
    tag: str,
    source: str,
    redact_visual: bool,
    redact_audio: bool,
) -> bool:
    tag = tag.strip().lower()
    source = source.strip().lower()
    if tag == "audio" or source.startswith("data:audio/"):
        return redact_audio
    if tag in {"img", "video"}:
        return redact_visual
    if source.startswith("data:image/") or source.startswith("data:video/"):
        return redact_visual
    return False


def _should_redact_mime(
    mime: str,
    *,
    redact_visual: bool,
    redact_audio: bool,
) -> bool:
    mime = mime.strip().lower()
    if mime.startswith("audio/"):
        return redact_audio
    if mime.startswith("image/") or mime.startswith("video/"):
        return redact_visual
    return False
