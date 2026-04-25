# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging
from collections.abc import Callable
from json import JSONDecodeError

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Content,
    ResponseReasoningItem,
)
from pydantic import TypeAdapter, ValidationError

from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.entrypoints.constants import MCP_PREFIX
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.outputs import CompletionOutput
from vllm.parser.abstract_parser import Parser as BaseParser
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


class ResponsesParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
        response_messages: list[ResponseInputOutputItem],
        request: ResponsesRequest,
        tool_parser_cls: type[ToolParser] | None,
    ):
        self.response_messages: list[ResponseInputOutputItem] = (
            # TODO: initial messages may not be properly typed
            response_messages
        )
        self.num_init_messages = len(response_messages)
        self.tokenizer = tokenizer
        self.request = request

        self.reasoning_parser_instance = reasoning_parser_cls(tokenizer)
        self.tool_parser_instance = None
        if tool_parser_cls is not None:
            self.tool_parser_instance = tool_parser_cls(tokenizer, request.tools)

        # Store the last finish_reason to determine response status
        self.finish_reason: str | None = None
        self._buffered_non_auto_text = ""
        self._buffered_stream_text = ""

    @staticmethod
    def _forced_tool_name(request: ResponsesRequest) -> str | None:
        tool_choice = request.tool_choice
        name = getattr(tool_choice, "name", None)
        if isinstance(name, str) and name:
            return name
        function = getattr(tool_choice, "function", None)
        function_name = getattr(function, "name", None)
        if isinstance(function_name, str) and function_name:
            return function_name
        return None

    def _parse_non_auto_tool_calls(
        self, content: str | None
    ) -> tuple[list[ResponseFunctionToolCall], str | None]:
        content = content or ""
        function_calls: list[ResponseFunctionToolCall] = []
        forced_tool_name = self._forced_tool_name(self.request)

        def append_call(name: str, arguments: dict) -> None:
            function_calls.append(
                ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()}",
                    call_id=f"call_{random_uuid()}",
                    type="function_call",
                    status="completed",
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            )

        if forced_tool_name is not None:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "name" not in parsed:
                append_call(forced_tool_name, parsed)
                return function_calls, None

        if self.request.tool_choice == "required" or forced_tool_name is not None:
            try:
                tool_defs = TypeAdapter(list[FunctionDefinition]).validate_json(content)
                for tool_def in tool_defs:
                    name = tool_def.name or forced_tool_name
                    if not name:
                        continue
                    append_call(name, tool_def.parameters or {})
                if function_calls:
                    return function_calls, None
            except (ValidationError, JSONDecodeError, ValueError):
                recovered_calls = BaseParser._recover_required_tool_calls(
                    self.request, content
                )
                if recovered_calls:
                    for call in recovered_calls:
                        function_calls.append(
                            ResponseFunctionToolCall(
                                id=f"fc_{random_uuid()}",
                                call_id=call.id or f"call_{random_uuid()}",
                                type="function_call",
                                status="completed",
                                name=call.name,
                                arguments=call.arguments,
                            )
                        )
                    return function_calls, None

        return function_calls, content

    def process(self, output: CompletionOutput) -> "ResponsesParser":
        # Store the finish_reason from the output
        self.finish_reason = output.finish_reason
        self._buffered_stream_text += output.text
        if self.request.tool_choice != "auto" and self.request.tools:
            self._buffered_non_auto_text += output.text

        if self.request.stream and output.finish_reason is None:
            return self

        if self.request.tool_choice != "auto" and self.request.tools:
            model_text = self._buffered_non_auto_text
        elif self.request.stream:
            model_text = self._buffered_stream_text
        else:
            model_text = output.text

        reasoning, content = self.reasoning_parser_instance.extract_reasoning(
            model_text, request=self.request
        )
        if self.request.reasoning is not None and self.request.reasoning.effort == "none":
            if reasoning:
                content = f"{reasoning}{content or ''}"
                reasoning = None

        if reasoning:
            self.response_messages.append(
                ResponseReasoningItem(
                    type="reasoning",
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    content=[
                        Content(
                            type="reasoning_text",
                            text=reasoning,
                        )
                    ],
                )
            )

        function_calls: list[ResponseFunctionToolCall] = []
        if self.request.tool_choice != "auto" and self.request.tools:
            function_calls, content = self._parse_non_auto_tool_calls(content)
        elif self.tool_parser_instance is not None:
            tool_call_info = self.tool_parser_instance.extract_tool_calls(
                content if content is not None else "",
                request=self.request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                # extract_tool_calls() returns a list of tool calls.
                function_calls.extend(
                    ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()}",
                        call_id=f"call_{random_uuid()}",
                        type="function_call",
                        status="completed",
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                content = tool_call_info.content
                if content and content.strip() == "":
                    content = None

        if content:
            self.response_messages.append(
                ResponseOutputMessage(
                    type="message",
                    id=f"msg_{random_uuid()}",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            annotations=[],  # TODO
                            type="output_text",
                            text=content,
                            logprobs=None,  # TODO
                        )
                    ],
                )
            )
        if len(function_calls) > 0:
            self.response_messages.extend(function_calls)

        return self

    def make_response_output_items_from_parsable_context(
        self,
    ) -> list[ResponseOutputItem]:
        """Given a list of sentences, construct ResponseOutput Items."""
        response_messages = self.response_messages[self.num_init_messages :]
        output_messages: list[ResponseOutputItem] = []
        for message in response_messages:
            if not isinstance(message, ResponseFunctionToolCallOutputItem):
                output_messages.append(message)
            else:
                if len(output_messages) == 0:
                    raise ValueError(
                        "Cannot have a FunctionToolCallOutput before FunctionToolCall."
                    )
                if isinstance(output_messages[-1], ResponseFunctionToolCall):
                    mcp_message = McpCall(
                        id=f"{MCP_PREFIX}{random_uuid()}",
                        arguments=output_messages[-1].arguments,
                        name=output_messages[-1].name,
                        server_label=output_messages[
                            -1
                        ].name,  # TODO: store the server label
                        type="mcp_call",
                        status="completed",
                        output=message.output,
                        # TODO: support error output
                    )
                    output_messages[-1] = mcp_message

        return output_messages


def get_responses_parser_for_simple_context(
    *,
    tokenizer: TokenizerLike,
    reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
    tool_parser_cls,
) -> ResponsesParser:
    """Factory function to create a ResponsesParser with
    optional reasoning parser.

    Returns:
        ResponsesParser instance configured with the provided parser
    """
    return ResponsesParser(
        tokenizer=tokenizer,
        reasoning_parser_cls=reasoning_parser_cls,
        response_messages=response_messages,
        request=request,
        tool_parser_cls=tool_parser_cls,
    )
