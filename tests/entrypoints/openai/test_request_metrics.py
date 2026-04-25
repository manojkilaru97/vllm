# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.request_metrics import summarize_chat_request


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        ("user-controlled-label", "unknown"),
        ({"type": "function", "function": {"name": "lookup"}}, "named"),
        ({"type": "custom-user-controlled-label"}, "unknown"),
        (SimpleNamespace(type="function"), "named"),
        (SimpleNamespace(type="custom-user-controlled-label"), "unknown"),
    ],
)
def test_tool_choice_summary_uses_bounded_labels(tool_choice, expected):
    summary = summarize_chat_request({"tool_choice": tool_choice})

    assert summary.tool_choice == expected
