# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    _normalize_json_schema_for_backend,
)


def test_chat_completion_request_with_no_tools():
    # tools key is not present
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"

    # tools key is None
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": None,
        }
    )
    assert request.tool_choice == "none"

    # tools key present but empty
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [],
        }
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_chat_completion_request_with_tool_choice_but_no_tools(tool_choice):
    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
                "tools": None,
            }
        )


def test_normalize_json_schema_property_bag_root():
    normalized = _normalize_json_schema_for_backend(
        {
            "host": {"type": "string"},
            "port": {"type": "integer"},
            "required": ["host", "port"],
        }
    )
    assert normalized == {
        "type": "object",
        "properties": {
            "host": {"type": "string"},
            "port": {"type": "integer"},
        },
        "required": ["host", "port"],
        "additionalProperties": False,
    }


def test_normalize_json_schema_empty_format_and_not_number_root():
    normalized = _normalize_json_schema_for_backend(
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "format": "",
                }
            },
            "required": ["content"],
        }
    )
    assert normalized == {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
            }
        },
        "required": ["content"],
    }

    assert _normalize_json_schema_for_backend({"not": {"type": "number"}}) == {
        "anyOf": [
            {"type": "object"},
            {"type": "array"},
            {"type": "string"},
            {"type": "boolean"},
            {"type": "null"},
        ]
    }


def test_normalize_boolean_json_schema_roots():
    assert _normalize_json_schema_for_backend(True) == {}
    assert _normalize_json_schema_for_backend(False) == {
        "allOf": [
            {"type": "string"},
            {"type": "number"},
        ]
    }


def test_normalize_json_schema_preserves_nested_schema_maps_and_booleans():
    normalized = _normalize_json_schema_for_backend(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "payload": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {
                            "type": "string",
                            "format": "",
                        }
                    },
                    "required": ["name"],
                }
            },
            "required": ["payload"],
        }
    )
    assert normalized == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                    }
                },
                "required": ["name"],
            }
        },
        "required": ["payload"],
    }
