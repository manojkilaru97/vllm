# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from typing import Any

from vllm import envs

MM_INPUT_REDACTED = "[redacted-mm-input]"

_MM_CONTAINER_KEYS = {
    "audio_url",
    "image_url",
    "input_audio",
    "input_image",
    "input_video",
    "video_url",
}
_MM_SCALAR_KEYS = {
    "asset_id",
    "b64_json",
    "data",
    "file_id",
    "url",
}
_HTML_MM_SRC_RE = re.compile(
    r"(<(?:img|video|audio)\b[^>]*\bsrc\s*=\s*[\"'])([^\"']+)([\"'])",
    re.IGNORECASE,
)
_DATA_URI_RE = re.compile(
    r"data:(?:image|video|audio)/[^,;]+(?:;[^,]*)?,[^\s\"'<>]+",
    re.IGNORECASE,
)
_ASSET_ID_URI_RE = re.compile(
    r"(data:(?:image|video|audio)/[^;]+;asset_id,)([^\s\"'<>]+)",
    re.IGNORECASE,
)
_ASSET_ID_TEXT_RE = re.compile(r"(asset_id:)([^\s\"'<>]+)", re.IGNORECASE)


def log_mm_input_metadata_enabled() -> bool:
    return bool(envs.VLLM_LOG_MM_INPUT_METADATA)


def sanitize_mm_text(text: str) -> str:
    if not text:
        return text

    sanitized = _HTML_MM_SRC_RE.sub(
        lambda match: f"{match.group(1)}{MM_INPUT_REDACTED}{match.group(3)}",
        text,
    )
    sanitized = _ASSET_ID_URI_RE.sub(rf"\1{MM_INPUT_REDACTED}", sanitized)
    sanitized = _ASSET_ID_TEXT_RE.sub(rf"\1{MM_INPUT_REDACTED}", sanitized)
    sanitized = _DATA_URI_RE.sub(MM_INPUT_REDACTED, sanitized)
    return sanitized


def maybe_redact_mm_payload(payload: Any) -> Any:
    if log_mm_input_metadata_enabled():
        return payload
    return redact_mm_input_metadata(payload)


def maybe_redact_mm_text(text: str) -> str:
    if log_mm_input_metadata_enabled():
        return text
    return sanitize_mm_text(text)


def redact_mm_input_metadata(payload: Any) -> Any:
    return _redact(payload, parent_path=())


def _redact(value: Any, *, parent_path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            next_path = parent_path + (key_str,)
            if key_str in _MM_CONTAINER_KEYS:
                redacted[key] = _redact_mm_container(item, container_key=key_str)
            elif key_str in _MM_SCALAR_KEYS and _path_has_mm_container(parent_path):
                redacted[key] = MM_INPUT_REDACTED
            else:
                redacted[key] = _redact(item, parent_path=next_path)
        return redacted

    if isinstance(value, list):
        return [_redact(item, parent_path=parent_path) for item in value]

    if isinstance(value, str):
        if _path_has_mm_container(parent_path):
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


def _path_has_mm_container(parent_path: tuple[str, ...]) -> bool:
    return any(part in _MM_CONTAINER_KEYS for part in parent_path)
