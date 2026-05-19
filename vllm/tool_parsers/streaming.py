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


def _find_matching_brace(text: str, opening_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for idx in range(opening_index, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _extract_complete_required_calls(text: str) -> list[tuple[str, str]]:
    """Return completed ``(name, parameters_json)`` calls from required JSON."""
    calls: list[tuple[str, str]] = []
    search_from = 0
    while True:
        name_match = re.search(
            r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', text[search_from:]
        )
        if name_match is None:
            break
        name_start = search_from + name_match.start()
        name_end = search_from + name_match.end()
        try:
            function_name = json.loads(f'"{name_match.group(1)}"')
        except json.JSONDecodeError:
            break

        params_match = re.search(r'"parameters"\s*:', text[name_end:])
        if params_match is None:
            break
        params_start = name_end + params_match.end()
        opening_index = text.find("{", params_start)
        if opening_index < 0:
            break
        closing_index = _find_matching_brace(text, opening_index)
        if closing_index is None:
            break
        parameters = text[opening_index : closing_index + 1]
        try:
            json.loads(parameters)
        except json.JSONDecodeError:
            break
        calls.append((function_name, parameters))
        search_from = closing_index + 1
        if search_from <= name_start:
            break
    return calls


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
    emitted_tool_calls = tool_call_idx or 0
    completed_calls = _extract_complete_required_calls(current_text or "")
    if len(completed_calls) > emitted_tool_calls:
        delta_tool_calls: list[DeltaToolCall] = []
        for index, (function_name, arguments) in enumerate(
            completed_calls[emitted_tool_calls:], start=emitted_tool_calls
        ):
            tool_call_id = make_tool_call_id(
                id_type=tool_call_id_type,
                func_name=function_name,
                idx=index,
            )
            delta_tool_calls.append(
                DeltaToolCall(
                    id=tool_call_id,
                    function=DeltaFunctionCall(
                        name=function_name,
                        arguments=arguments,
                    ),
                    index=index,
                    type="function",
                )
            )
        return DeltaMessage(tool_calls=delta_tool_calls), False

    return None, False

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
                param_match = re.search(
                    r'.*"parameters":\s*(.*)', current_text, re.DOTALL
                )
                arguments = param_match.group(1) if param_match else ""
                arguments, _ = filter_delta_text(arguments, previous_text)

                # if this iteration finishes a previous tool call but a
                # new incomplete tool is already generated, take the
                # previous from the list
                if finishes_previous_tool and "parameters" not in current_tool_call:
                    current_tool_call = obj[-2]
                    current_tool_call_index -= 1

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
