# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionToolsParam
from vllm.tool_parsers.abstract_tool_parser import ToolParser

pytestmark = pytest.mark.cpu_test


def test_tool_choice_required_does_not_apply_structured_outputs_grammar():
    tools = TypeAdapter(list[ChatCompletionToolsParam]).validate_python(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
    )

    req = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice="required",
    )
    assert req.structured_outputs is None

    parser = ToolParser(tokenizer=MagicMock())
    req2 = parser.adjust_request(request=req)

    # For robustness, required behaves like auto and must not set structured outputs.
    assert req2.tool_choice == "auto"
    assert req2.structured_outputs is None


