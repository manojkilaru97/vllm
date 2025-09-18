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

from vllm.entrypoints.chat_utils import random_tool_call_id
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

        # Tokens and patterns
        self.bot_token = "<TOOLCALL>"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

        # Buffer for partial tag sequences in streaming
        self._pending_tag_buffer: str = ""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # Ensure special tokens are not skipped so <TOOLCALL> tags are visible
        if request.tools and request.tool_choice != "none":
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
        tool_content = model_output.replace(self.bot_token, "").strip()

        try:
            # Prefer direct JSON load; fall back to regex extraction of the array
            try:
                function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            tool_calls: list[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        arguments=json.dumps(
                            raw_function_call["arguments"], ensure_ascii=False,
                        ),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            content = model_output.split(self.bot_token)[0]
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
            parsable_arr = current_text.split(self.bot_token)[-1]

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

            # Current tool
            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

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
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
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
                            id=random_tool_call_id(),
                            function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                        )
                    ])
                    self.current_tool_name_sent = True
                else:
                    delta = None

            else:
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("\'", '"')
                if '"}' in new_text:
                    new_text = new_text[:new_text.rindex('"}')]

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error("INVARIANT - impossible to have arguments reset mid-arguments")
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    streamed_prefix = self.streamed_args_for_tool[self.current_tool_id]
                    if cur_arguments_json.startswith(streamed_prefix):
                        arguments_delta = cur_arguments_json[len(streamed_prefix):]
                    else:
                        arguments_delta = extract_intermediate_diff(cur_arguments_json, streamed_prefix)

                    if (not self.streamed_args_for_tool[self.current_tool_id]
                            and not end_of_call and arguments_delta and arguments_delta.endswith('}')):
                        arguments_delta = arguments_delta[:-1]

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

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    argument_diff = extract_intermediate_diff(cur_args_json, prev_args_json)
                    if argument_diff:
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                            )
                        ])
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None

            self.prev_tool_call_arr = tool_call_arr

            # End-of-call flush of remaining suffix
            if end_of_call and self.current_tool_id >= 0:
                try:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments is not None:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        streamed_prefix = self.streamed_args_for_tool[self.current_tool_id]
                        if cur_args_json.startswith(streamed_prefix):
                            remaining_suffix = cur_args_json[len(streamed_prefix):]
                        else:
                            remaining_suffix = extract_intermediate_diff(cur_args_json, streamed_prefix)

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


