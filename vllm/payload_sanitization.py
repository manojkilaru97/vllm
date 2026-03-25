# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import copy
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.request import url2pathname

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
_HTML_MEDIA_SRC_RE = re.compile(
    r"<(?P<tag>img|video|audio)\s+[^>]*src=\"(?P<src>[^\"]+)\"[^>]*\/?>",
    re.IGNORECASE,
)


def log_mm_input_metadata_enabled() -> bool:
    return bool(envs.VLLM_LOG_MM_INPUT_METADATA)


def log_audio_input_metadata_enabled() -> bool:
    return log_mm_input_metadata_enabled() and bool(
        envs.VLLM_LOG_AUDIO_INPUT_METADATA
    )


def sanitize_mm_text(text: str) -> str:
    if not text:
        return text

    redact_visual = not log_mm_input_metadata_enabled()
    redact_audio = not log_audio_input_metadata_enabled()
    if not redact_visual and not redact_audio:
        return text

    def _replace_html_src(match: re.Match[str]) -> str:
        src = match.group("src") or ""
        if _should_redact_text_media(
            tag=(match.group("tag") or ""),
            source=src,
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
    sanitized = _DATA_URI_RE.sub(_replace_data_uri, sanitized)
    return sanitized


def maybe_redact_mm_payload(payload: Any) -> Any:
    if log_mm_input_metadata_enabled() and log_audio_input_metadata_enabled():
        return payload
    return redact_mm_input_metadata(payload)


def maybe_redact_mm_text(text: str) -> str:
    if log_mm_input_metadata_enabled() and log_audio_input_metadata_enabled():
        return text
    return sanitize_mm_text(text)


def prepare_request_payload_for_logging(
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    allowed_local_media_path: str = "",
) -> Any:
    if payload is None:
        return payload
    payload_copy = copy.deepcopy(payload)
    payload_copy = _materialize_media_payload_for_logging(
        payload_copy,
        headers=headers or {},
        allowed_local_media_path=allowed_local_media_path,
    )
    return maybe_redact_mm_payload(payload_copy)


def redact_mm_input_metadata(payload: Any) -> Any:
    return _redact(payload, parent_path=())


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


def _path_has_mm_container(parent_path: tuple[str, ...]) -> bool:
    return any(part in _MM_CONTAINER_KEYS for part in parent_path)


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


def _materialize_media_payload_for_logging(
    payload: Any,
    *,
    headers: dict[str, str],
    allowed_local_media_path: str,
) -> Any:
    if not isinstance(payload, dict):
        return payload

    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = [
            _materialize_media_message_for_logging(
                msg,
                headers=headers,
                allowed_local_media_path=allowed_local_media_path,
            )
            for msg in messages
        ]

    return payload


def _materialize_media_message_for_logging(
    message: Any,
    *,
    headers: dict[str, str],
    allowed_local_media_path: str,
) -> Any:
    if not isinstance(message, dict):
        return message

    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [
            _materialize_media_part_for_logging(
                part,
                headers=headers,
                allowed_local_media_path=allowed_local_media_path,
            )
            for part in content
        ]
    elif isinstance(content, str):
        message["content"] = _materialize_media_text_for_logging(
            content,
            headers=headers,
            allowed_local_media_path=allowed_local_media_path,
        )

    return message


def _materialize_media_part_for_logging(
    part: Any,
    *,
    headers: dict[str, str],
    allowed_local_media_path: str,
) -> Any:
    if not isinstance(part, dict):
        return part

    for field_name, kind in (
        ("image_url", "image"),
        ("video_url", "video"),
        ("audio_url", "audio"),
    ):
        field_value = part.get(field_name)
        if isinstance(field_value, dict):
            url = field_value.get("url")
            if isinstance(url, str):
                field_value["url"] = _materialize_media_url_for_logging(
                    url,
                    kind=kind,
                    headers=headers,
                    allowed_local_media_path=allowed_local_media_path,
                )
        elif isinstance(field_value, str):
            part[field_name] = {
                "url": _materialize_media_url_for_logging(
                    field_value,
                    kind=kind,
                    headers=headers,
                    allowed_local_media_path=allowed_local_media_path,
                )
            }

    input_audio = part.get("input_audio")
    if isinstance(input_audio, dict) and log_audio_input_metadata_enabled():
        audio_format = str(input_audio.get("format") or "").strip().lower()
        audio_data = input_audio.get("data")
        if audio_format and isinstance(audio_data, str) and audio_data:
            input_audio["data"] = f"data:audio/{audio_format};base64,{audio_data}"

    return part


def _materialize_media_text_for_logging(
    text: str,
    *,
    headers: dict[str, str],
    allowed_local_media_path: str,
) -> str:
    def _replace(match: re.Match[str]) -> str:
        tag = (match.group("tag") or "").lower()
        if tag == "img":
            kind = "image"
        elif tag == "video":
            kind = "video"
        else:
            kind = "audio"
        src = match.group("src") or ""
        replaced = _materialize_media_url_for_logging(
            src,
            kind=kind,
            headers=headers,
            allowed_local_media_path=allowed_local_media_path,
        )
        return match.group(0).replace(src, replaced, 1)

    return _HTML_MEDIA_SRC_RE.sub(_replace, text)


def _materialize_media_url_for_logging(
    url: str,
    *,
    kind: str,
    headers: dict[str, str],
    allowed_local_media_path: str,
) -> str:
    if not _should_materialize_kind(kind):
        return url
    lowered = url.lower()
    if lowered.startswith("file://"):
        return _file_url_to_data_url_for_logging(
            url,
            kind=kind,
            allowed_local_media_path=allowed_local_media_path,
        )
    if kind in {"image", "video"} and ";asset_id," in lowered:
        return _asset_ref_to_data_url_for_logging(url, headers=headers)
    return url


def _file_url_to_data_url_for_logging(
    url: str,
    *,
    kind: str,
    allowed_local_media_path: str,
) -> str:
    if not allowed_local_media_path:
        return url

    allowed_root = Path(allowed_local_media_path).resolve()
    path_part = url[7:]
    filepath = Path(url2pathname(path_part)).resolve()
    if allowed_root not in filepath.parents:
        return url

    try:
        raw = filepath.read_bytes()
    except Exception:
        return url

    mime = mimetypes.guess_type(str(filepath))[0] or _fallback_mime(kind)
    data_b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{data_b64}"


def _asset_ref_to_data_url_for_logging(url: str, *, headers: dict[str, str]) -> str:
    asset_dir = headers.get("nvcf-asset-dir") or headers.get("NVCF-ASSET-DIR")
    allowed_ids_hdr = headers.get("nvcf-function-asset-ids") or headers.get(
        "NVCF-FUNCTION-ASSET-IDS"
    )
    if not asset_dir or not allowed_ids_hdr:
        return url

    match = re.match(
        r"^data:(?P<mime>(?:image|video)/[^;]+);asset_id,(?P<asset_id>.+)$",
        url,
        re.IGNORECASE,
    )
    if not match:
        return url

    asset_root = Path(asset_dir).resolve()
    if not asset_root.exists() or not asset_root.is_dir():
        return url

    asset_id = _normalize_asset_id(match.group("asset_id"))
    allowed_ids = {
        _normalize_asset_id(item)
        for item in allowed_ids_hdr.split(",")
        if item.strip()
    }
    if asset_id not in allowed_ids:
        return url

    filepath = (asset_root / asset_id).resolve()
    if asset_root not in filepath.parents and filepath != asset_root:
        return url

    try:
        raw = filepath.read_bytes()
    except Exception:
        return url

    mime = match.group("mime")
    data_b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{data_b64}"


def _normalize_asset_id(value: str) -> str:
    normalized = (value or "").strip().strip(",").strip()
    while len(normalized) >= 2 and normalized[0] in ("'", '"') and normalized[-1] == normalized[0]:
        normalized = normalized[1:-1].strip()
    return normalized


def _fallback_mime(kind: str) -> str:
    if kind == "audio":
        return "audio/wav"
    if kind == "video":
        return "video/mp4"
    return "image/png"


def _should_materialize_kind(kind: str) -> bool:
    if kind == "audio":
        return log_audio_input_metadata_enabled()
    if kind in {"image", "video"}:
        return log_mm_input_metadata_enabled()
    return False
