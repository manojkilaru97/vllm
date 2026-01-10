# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64

import pytest

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat


class _DummyRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def test_html_audio_base64_is_structured_without_nvcf_headers():
    audio_b64 = "AAAA"
    messages = [
        {
            "role": "user",
            "content": (
                "Transcribe this please: "
                f'<audio src="data:audio/wav;base64,{audio_b64}"/>'
                " thanks"
            ),
        }
    ]

    out = OpenAIServingChat._resolve_nvcf_image_assets(messages, _DummyRequest({}))
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "audio_url"
    assert content[1]["audio_url"]["url"] == f"data:audio/wav;base64,{audio_b64}"
    assert content[-1]["type"] == "text"


def test_html_audio_asset_id_is_resolved_with_headers(tmp_path):
    asset_id = "test.wav"
    asset_bytes = b"RIFFxxxxWAVEfmt "  # minimal-ish header bytes
    (tmp_path / asset_id).write_bytes(asset_bytes)

    messages = [
        {
            "role": "user",
            "content": f'<audio src="data:audio/wav;asset_id,{asset_id}"/>',
        }
    ]
    headers = {
        "NVCF-ASSET-DIR": str(tmp_path),
        "NVCF-FUNCTION-ASSET-IDS": asset_id,
    }

    out = OpenAIServingChat._resolve_nvcf_image_assets(messages, _DummyRequest(headers))
    part = out[0]["content"][0]
    assert part["type"] == "audio_url"
    assert part["audio_url"]["url"].startswith("data:audio/wav;base64,")
    expected_b64 = base64.b64encode(asset_bytes).decode("ascii")
    assert part["audio_url"]["url"] == f"data:audio/wav;base64,{expected_b64}"


def test_structured_audio_asset_id_is_resolved_with_headers(tmp_path):
    asset_id = "test.wav"
    asset_bytes = b"RIFFxxxxWAVEfmt "
    (tmp_path / asset_id).write_bytes(asset_bytes)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;asset_id,{asset_id}"}},
                {"type": "text", "text": "What is said?"},
            ],
        }
    ]
    headers = {
        "NVCF-ASSET-DIR": str(tmp_path),
        "NVCF-FUNCTION-ASSET-IDS": asset_id,
    }

    out = OpenAIServingChat._resolve_nvcf_image_assets(messages, _DummyRequest(headers))
    part0 = out[0]["content"][0]
    assert part0["type"] == "audio_url"
    expected_b64 = base64.b64encode(asset_bytes).decode("ascii")
    assert part0["audio_url"]["url"] == f"data:audio/wav;base64,{expected_b64}"


@pytest.mark.parametrize(
    "content",
    [
        '<audio src="data:audio/wav;asset_id,missing.wav"/>',
        [
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;asset_id,missing.wav"},
            }
        ],
    ],
)
def test_audio_asset_id_without_headers_is_left_as_text(content):
    messages = [{"role": "user", "content": content}]
    out = OpenAIServingChat._resolve_nvcf_image_assets(messages, _DummyRequest({}))
    # Backward-compatible behavior: without NVCF headers, don't attempt resolution.
    assert out[0]["content"] == content

