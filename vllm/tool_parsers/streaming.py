# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import TYPE_CHECKING

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.utils import partial_json_loads
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object


def _bracket_level(s: str, opening: str = "{", closing: str = "}") -> int:
    """Calculate the current level of nested brackets in a string."""
    level = 0
    for char in s:
        if char == opening:
            level += 1
        elif char == closing:
            level -= 1
    return level


def filter_delta_text(
    delta_text: str,
    previous_text: str,
) -> tuple[str, bool]:
    """Trim trailing tool-list delimiters from required-tool streaming text."""
    bracket_level = _bracket_level(previous_text)
    updated_delta = ""
    passed_zero = False
    for char in delta_text:
        if char == "{":
            bracket_level += 1
            passed_zero = bracket_level == 0
        elif char == "}":
            bracket_level -= 1
            passed_zero = bracket_level == 0

        if bracket_level != 0:
            updated_delta += char
        else:
            if char == ",":
                break
    return updated_delta, passed_zero


def _extract_json_value_after_parameters(text: str) -> str:
    """Return the JSON value following the last generated parameters key."""
    param_match = re.search(r'.*"parameters":\s*', text, re.DOTALL)
    if param_match is None:
        return ""

    value = text[param_match.end():]
    if value == "":
        return value

    opening_to_closing = {"{": "}", "[": "]"}
    opening = value[0]
    closing = opening_to_closing.get(opening)
    if closing is None:
        return value

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return value[:index + 1]
    return value


def extract_named_tool_call_streaming(
    *,
    delta_text: str,
    function_name: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
    tokenizer: "TokenizerLike",
    tool_call_array_index: int = 0,
) -> tuple[DeltaMessage | None, bool]:
    """Build a streaming tool-call delta for forced named tool choice."""
    if function_name_returned:
        delta_tool_call = DeltaToolCall(
            function=DeltaFunctionCall(arguments=delta_text),
            index=tool_call_array_index,
        )
    else:
        if is_mistral_tokenizer(tokenizer):
            # Import mistral_common only if we need it.
            from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

            tool_call_id = MistralToolCall.generate_random_id()
        else:
            tool_call_id = make_tool_call_id(
                id_type=tool_call_id_type,
                func_name=function_name,
                idx=tool_call_idx,
            )
        delta_tool_call = DeltaToolCall(
            id=tool_call_id,
            type="function",
            function=DeltaFunctionCall(
                name=function_name,
                arguments=delta_text,
            ),
            index=tool_call_array_index,
        )
        function_name_returned = True
    return (
        DeltaMessage(tool_calls=[delta_tool_call]),
        function_name_returned,
    )


def extract_required_tool_call_streaming(
    *,
    previous_text: str,
    current_text: str | None,
    delta_text: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, bool]:
    if current_text is None or current_text == "":
        # if the current text is empty, we cannot parse it
        return None, function_name_returned
    try:
        flags = Allow.ALL
        obj, _ = partial_json_loads(current_text, flags)
    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
    ):
        obj = None

    # check if the current text is a valid array
    # containing a partial tool calling object
    # if not repeat
    if obj is None or not isinstance(obj, list) or not len(obj) > 0:
        function_name_returned = False
        delta_message = None
    else:
        _, finishes_previous_tool = filter_delta_text(delta_text, previous_text)
        # take the last tool call from the generated list
        current_tool_call = obj[-1]
        current_tool_call_index = len(obj) - 1

        # A single streamed chunk can finish tool N and already include the
        # start of tool N+1. In that case partial_json_loads exposes the new
        # incomplete object as obj[-1], but any argument delta still belongs to
        # the previous tool. Switch to the completed previous object before
        # deciding whether this is a new tool that needs name/id emission.
        if (
            finishes_previous_tool
            and "parameters" not in current_tool_call
            and len(obj) > 1
        ):
            current_tool_call = obj[-2]
            current_tool_call_index -= 1

        if (
            function_name_returned
            and tool_call_idx is not None
            and current_tool_call_index >= tool_call_idx
        ):
            # A new required-tool call has started. The single boolean state
            # tracks the currently streamed call, so reset it before emitting
            # arguments for the new array element.
            function_name_returned = False

        # once parameters have been generated the name is complete as well
        if not finishes_previous_tool and (
            "name" not in current_tool_call or "parameters" not in current_tool_call
        ):
            function_name_returned = False
            delta_message = None
        else:
            if not function_name_returned:
                # get partly generated arguments from the latest tool call
                arguments = _extract_json_value_after_parameters(current_text)
                arguments, _ = filter_delta_text(arguments, previous_text)

                function_name_returned = True
                tool_call_id = make_tool_call_id(
                    id_type=tool_call_id_type,
                    func_name=current_tool_call["name"],
                    idx=tool_call_idx,
                )
                delta_message = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            id=tool_call_id,
                            function=DeltaFunctionCall(
                                name=current_tool_call["name"], arguments=arguments
                            ),
                            index=current_tool_call_index,
                            type="function",
                        )
                    ]
                )

            else:
                current_arguments = _extract_json_value_after_parameters(current_text)
                previous_arguments = _extract_json_value_after_parameters(previous_text)
                if current_arguments.startswith(previous_arguments):
                    delta_text = current_arguments[len(previous_arguments) :]
                else:
                    delta_text, _ = filter_delta_text(delta_text, previous_text)

                if delta_text != "":
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                name=None,
                                arguments=delta_text,
                            ),
                            index=current_tool_call_index,
                        )
                    ]
                )
                else:
                    delta_message = None

    return delta_message, function_name_returned
