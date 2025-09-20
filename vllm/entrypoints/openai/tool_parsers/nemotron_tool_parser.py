"""
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project

Nemotron tool parser for models that emit tool calls wrapped in
<TOOLCALL> ... </TOOLCALL> with JSON array payloads, e.g.:
<TOOLCALL>[{"name": "fn", "arguments": {"k": "v"}}]</TOOLCALL>

Supports both full completion parsing and streaming parsing with
partial JSON handling.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Union

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("nemotron")
class NemotronToolParser(ToolParser):
    """
    Tool call parser for Nemotron models emitting <TOOLCALL> tags
    with JSON-encoded tool call arrays.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # State for streaming
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.last_complete_args: list[str] = []  # Track last complete arguments for each tool

        # Tokens and patterns
        self.bot_token = "<TOOLCALL>"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

        # Buffer for partial tag sequences in streaming
        self._pending_tag_buffer: str = ""

    @staticmethod
    def _sanitize_toolcall_wrappers(text: str) -> str:
        # Remove any stray TOOLCALL tags that might be leaked into arguments
        if not text:
            return text
        return text.replace("<TOOLCALL>", "").replace("</TOOLCALL>", "")

    @staticmethod
    def _extract_top_level_json_array_text(text: str) -> str | None:
        """Extract the first top-level JSON array substring from text.
        Returns None if '[' not found.
        """
        if not text:
            return None
        start = text.find('[')
        if start == -1:
            return None
        i = start
        depth = 0
        in_string = False
        escape = False
        n = len(text)
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                else:
                    if ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            i += 1
        # Fallback: return partial tail
        return text[start:]

    @staticmethod
    def _suffix_after_longest_common_prefix(current_text: str,
                                            streamed_prefix: str) -> str:
        """Return the incremental suffix of current_text after removing the
        longest common prefix with streamed_prefix.

        This is more robust than generic diff for streaming assembly and helps
        avoid duplicated substrings like repeated field prefixes.
        """
        if not streamed_prefix:
            return current_text
        i = 0
        max_i = min(len(current_text), len(streamed_prefix))
        while i < max_i and current_text[i] == streamed_prefix[i]:
            i += 1
        return current_text[i:]

    @staticmethod
    def _extract_raw_array_content(text: str, start_token: str, end_tag: str) -> str:
        """Return the substring inside <TOOLCALL> ... </TOOLCALL>.

        If the end tag is not present yet (streaming), return the tail after
        the start token.
        """
        after = text.split(start_token)[-1]
        end_idx = after.find(end_tag)
        return after[:end_idx] if end_idx != -1 else after

    @staticmethod
    def _extract_nth_object_in_array(arr_text: str, index: int) -> str | None:
        """Extract the raw substring of the index-th JSON object within a top-level array.

        Works on partial arrays during streaming. If the object hasn't closed
        yet, returns the substring from the opening brace to the current end.
        """
        if not arr_text:
            return None
        # Find the first '['
        start = arr_text.find('[')
        if start == -1:
            return None
        i = start + 1
        obj_idx = -1
        obj_start = None
        brace_count = 0
        in_string = False
        escape = False
        n = len(arr_text)
        while i < n:
            ch = arr_text[i]
            if in_string:
                if escape:
                    escape = False
                else:
                    if ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    if brace_count == 0:
                        obj_idx += 1
                        if obj_idx == index:
                            obj_start = i
                    brace_count += 1
                elif ch == '}':
                    if brace_count > 0:
                        brace_count -= 1
                        if brace_count == 0 and obj_start is not None and obj_idx == index:
                            return arr_text[obj_start:i + 1]
                elif ch == ']' and brace_count == 0:
                    break
            i += 1
        if obj_start is not None and obj_idx == index:
            return arr_text[obj_start:i]
        return None

    @staticmethod
    def _extract_arguments_raw(obj_text: str) -> str | None:
        """Extract the raw substring of the value of the "arguments" field.

        Returns the substring including surrounding braces if it's an object.
        Works on partial objects during streaming.
        """
        if not obj_text:
            return None
        key = '"arguments"'
        pos = obj_text.find(key)
        if pos == -1:
            return None
        i = pos + len(key)
        n = len(obj_text)
        # Skip whitespace
        while i < n and obj_text[i].isspace():
            i += 1
        if i >= n or obj_text[i] != ':':
            return None
        i += 1
        while i < n and obj_text[i].isspace():
            i += 1
        if i >= n:
            return None
        # Object value
        if obj_text[i] == '{':
            start = i
            brace = 0
            in_string = False
            escape = False
            while i < n:
                ch = obj_text[i]
                if in_string:
                    if escape:
                        escape = False
                    else:
                        if ch == '\\':
                            escape = True
                        elif ch == '"':
                            in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{':
                        brace += 1
                    elif ch == '}':
                        brace -= 1
                        if brace == 0:
                            return obj_text[start:i + 1]
                i += 1
            return obj_text[start:i]
        # String value
        if obj_text[i] == '"':
            start = i
            in_string = True
            escape = False
            i += 1
            while i < n:
                ch = obj_text[i]
                if escape:
                    escape = False
                else:
                    if ch == '\\':
                        escape = True
                    elif ch == '"':
                        return obj_text[start:i + 1]
                i += 1
            return obj_text[start:i]
        # Primitive value (number, true/false/null)
        start = i
        while i < n and obj_text[i] not in ',}':
            i += 1
        return obj_text[start:i].strip()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # Only for auto tool parsing should we expose <TOOLCALL> tokens.
        # For named/required tool_choice, keep default behavior so guided
        # decoding and postprocessors see clean JSON without wrappers.
        if request.tools and (request.tool_choice is None
                              or request.tool_choice == "auto"):
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # No tool call tag -> return content as-is
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        # Remove the leading tag sequence
        sanitized = self._sanitize_toolcall_wrappers(model_output).strip()
        # Extract the JSON array payload best-effort
        tool_content = self._extract_top_level_json_array_text(sanitized) or sanitized

        try:
            # Prefer direct JSON load; fall back to regex extraction of the array
            try:
                function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # Try partial parser first
                try:
                    function_call_arr = partial_json_parser.loads(tool_content, Allow.ALL)
                except Exception:
                    # Regex fallback: find the first [ ... ] segment if present
                    matches = self.tool_call_regex.findall(tool_content)
                    if not matches:
                        raise
                    function_call_arr = json.loads(matches[0])

            tool_calls: list[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        arguments=(json.dumps(raw_function_call["arguments"], ensure_ascii=False)
                                   if not isinstance(raw_function_call.get("arguments"), str)
                                   else raw_function_call["arguments"]),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            content = model_output.split(self.bot_token, 1)[0]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if len(content) > 0 else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=tool_content)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if not previous_text:
            # First chunk of a new generation, reset state
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self.last_complete_args = []

        # Handle potential partial tag buffering (e.g., "<", "<T", ...)
        try:
            start_token = self.bot_token
            end_token = "</TOOLCALL>"

            if delta_text == "<" and not self._pending_tag_buffer:
                self._pending_tag_buffer = "<"
                return None

            if self._pending_tag_buffer:
                self._pending_tag_buffer += delta_text

                if "<TOOLCALL>" in self._pending_tag_buffer:
                    buffered = self._pending_tag_buffer
                    self._pending_tag_buffer = ""
                    current_text = previous_text + buffered
                    delta_text = buffered
                elif self._pending_tag_buffer.startswith("</"):
                    return None
                else:
                    alphas = ""
                    for i in range(1, len(self._pending_tag_buffer)):
                        ch = self._pending_tag_buffer[i]
                        if ch.isalpha():
                            alphas += ch.upper()
                        else:
                            break
                    if alphas and "TOOLCALL".startswith(alphas) and len(alphas) < 8:
                        return None
                    if len(alphas) > 0 and not "TOOLCALL".startswith(alphas):
                        content_to_flush = self._pending_tag_buffer
                        self._pending_tag_buffer = ""
                        return DeltaMessage(content=content_to_flush)
                    return None

            if any(current_text.endswith(start_token[:k]) for k in range(1, len(start_token))):
                return None
            if any(current_text.endswith(end_token[:k]) for k in range(1, len(end_token))):
                return None
        except Exception:
            if (current_text.endswith("<") or current_text.endswith("<T") or
                    current_text.endswith("<TO") or current_text.endswith("<TOOL") or
                    current_text.endswith("<TOOLCALL")):
                return None

        # If tag not seen, just stream content
        if self.bot_token not in current_text:
            if self._pending_tag_buffer:
                content_to_flush = self._pending_tag_buffer + delta_text
                self._pending_tag_buffer = ""
                return DeltaMessage(content=content_to_flush)
            return DeltaMessage(content=delta_text)

        # Partial JSON parsing flags
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        end_of_call = False

        try:
            # Work on raw substring inside <TOOLCALL> ... </TOOLCALL>
            parsable_arr = self._extract_raw_array_content(
                current_text, self.bot_token, "</TOOLCALL>")
            parsable_arr = self._sanitize_toolcall_wrappers(parsable_arr)

            if "</TOOLCALL>" in parsable_arr:
                end_of_call = True
                parsable_arr = parsable_arr.split("</TOOLCALL>")[0]

            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                return None
            except json.JSONDecodeError:
                raw = parsable_arr
                if raw.endswith("</") or raw.endswith("</TO") or raw.endswith("</TOOL") or raw.endswith("</TOOLCALL"):
                    return None
                if "</TOOLCALL>" in raw:
                    end_of_call = True
                    raw = raw.split("</TOOLCALL>")[0]
                lt_index = raw.find("<")
                if lt_index != -1:
                    raw = raw[:lt_index]
                raw = raw.strip()
                raw = self._sanitize_toolcall_wrappers(raw)
                try:
                    tool_call_arr = json.loads(raw)
                except json.JSONDecodeError:
                    rb_index = raw.rfind("]")
                    if rb_index != -1:
                        try:
                            tool_call_arr = json.loads(raw[:rb_index + 1])
                        except json.JSONDecodeError:
                            return None
                    else:
                        return None

            # Current tool index & raw object substring for robust streaming diffs
            current_index = len(tool_call_arr) - 1
            current_tool_call: dict = tool_call_arr[current_index] if len(tool_call_arr) > 0 else {}

            if len(tool_call_arr) == 0:
                return None

            # Starting a new tool in the array
            elif (len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1):
                if self.current_tool_id >= 0:
                    diff: Union[str, None] = current_tool_call.get("arguments")
                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], "")
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                            )
                        ])
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = current_index
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.last_complete_args.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # Send name if not yet sent
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                        )
                    ])
                    self.current_tool_name_sent = True
                else:
                    delta = None

            else:
                # Robust raw-substring assembly of the current tool's arguments
                raw_arr = parsable_arr
                raw_obj = self._extract_nth_object_in_array(raw_arr, current_index)
                if raw_obj is not None:
                    raw_args = self._extract_arguments_raw(raw_obj)
                    if raw_args is not None:
                        streamed_prefix = self.streamed_args_for_tool[self.current_tool_id]
                        # Compute suffix to avoid duplication, based on already streamed raw content
                        arguments_delta = self._suffix_after_longest_common_prefix(raw_args, streamed_prefix)
                        if arguments_delta:
                            delta = DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True),
                                )
                            ])
                            self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                        else:
                            delta = None
                    else:
                        delta = None
                else:
                    delta = None

            self.prev_tool_call_arr = tool_call_arr

            # End-of-call flush - ensure we've sent all arguments
            if end_of_call and self.current_tool_id >= 0:
                try:
                    # Recompute raw args and flush any remaining suffix
                    raw_arr = parsable_arr
                    raw_obj = self._extract_nth_object_in_array(raw_arr, current_index)
                    if raw_obj is not None:
                        raw_args = self._extract_arguments_raw(raw_obj)
                        if raw_args is not None:
                            streamed_prefix = self.streamed_args_for_tool[self.current_tool_id]
                            remaining_suffix = self._suffix_after_longest_common_prefix(raw_args, streamed_prefix)
                            if remaining_suffix and remaining_suffix.strip():
                                extra = DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=remaining_suffix).model_dump(exclude_none=True),
                                )
                                if delta is None:
                                    delta = DeltaMessage(tool_calls=[extra])
                                else:
                                    if getattr(delta, "tool_calls", None):
                                        delta.tool_calls.append(extra)
                                    else:
                                        delta.tool_calls = [extra]
                                self.streamed_args_for_tool[self.current_tool_id] += remaining_suffix
                except Exception:
                    logger.debug("NemotronToolParser: end-of-call flush failed", exc_info=True)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None


