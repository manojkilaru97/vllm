# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class NemotronJsonToolParser(ToolParser):
    """Tool parser for Nemotron Nano v2 JSON tool calls.

    The model emits tool calls as:
    `<TOOLCALL>[{"name": "...", "arguments": {...}}]</TOOLCALL>`

    For streaming, this parser buffers tool text until the closing tag is
    present, then emits the completed tool calls in one delta.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tool_call_start_token = "<TOOLCALL>"
        self.tool_call_end_token = "</TOOLCALL>"
        self._pending_tag_buffer = ""
        self._streamed_content = ""
        self._emitted_tool_calls = False
        self._stream_tool_ids: list[str] = []
        self._inside_tool_block = False
        self._tool_block_buffer = ""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _normalize_arguments(arguments) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

    def _parse_tool_payload(self, payload: str) -> list[ToolCall]:
        payload = payload.strip()
        if not payload:
            return []
        if not payload.startswith("["):
            payload = "[" + payload
        if not payload.endswith("]"):
            payload = payload + "]"

        json_tool_calls = json.loads(payload)
        tool_calls: list[ToolCall] = []
        for item in json_tool_calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(name, str):
                continue
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=self._normalize_arguments(arguments),
                    ),
                )
            )
        return tool_calls

    def _decode_ids(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=False
            )
        except Exception:
            return ""

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        del request
        start_idx = model_output.find(self.tool_call_start_token)
        if start_idx < 0:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        end_idx = model_output.find(
            self.tool_call_end_token, start_idx + len(self.tool_call_start_token)
        )
        if end_idx < 0:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        payload = model_output[
            start_idx + len(self.tool_call_start_token) : end_idx
        ].strip()
        try:
            tool_calls = self._parse_tool_payload(payload)
        except Exception:
            logger.exception("Error in extracting Nemotron tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        prefix_content = model_output[:start_idx] or None
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=prefix_content,
        )

    def _visible_delta_outside_tool(self, delta_text: str) -> str:
        if not delta_text:
            return ""

        visible: list[str] = []
        for ch in delta_text:
            if self._pending_tag_buffer or ch == "<":
                self._pending_tag_buffer += ch
                if self.tool_call_start_token.startswith(self._pending_tag_buffer):
                    if self._pending_tag_buffer == self.tool_call_start_token:
                        self._pending_tag_buffer = ""
                    continue
                if self.tool_call_end_token.startswith(self._pending_tag_buffer):
                    if self._pending_tag_buffer == self.tool_call_end_token:
                        self._pending_tag_buffer = ""
                    continue
                visible.append(self._pending_tag_buffer)
                self._pending_tag_buffer = ""
            else:
                visible.append(ch)

        return "".join(visible)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del request

        decoded_previous = self._decode_ids(previous_token_ids)
        decoded_current = self._decode_ids(current_token_ids)
        decoded_delta = self._decode_ids(delta_token_ids)

        if (
            self.tool_call_start_token in decoded_previous
            or self.tool_call_end_token in decoded_previous
        ) and (
            self.tool_call_start_token not in previous_text
            and self.tool_call_end_token not in previous_text
        ):
            previous_text = decoded_previous

        if (
            self.tool_call_start_token in decoded_current
            or self.tool_call_end_token in decoded_current
        ) and (
            self.tool_call_start_token not in current_text
            and self.tool_call_end_token not in current_text
        ):
            current_text = decoded_current

        if (
            self.tool_call_start_token in decoded_delta
            or self.tool_call_end_token in decoded_delta
        ) and (
            self.tool_call_start_token not in delta_text
            and self.tool_call_end_token not in delta_text
        ):
            delta_text = decoded_delta

        if self._inside_tool_block:
            self._tool_block_buffer += delta_text
            end_idx = self._tool_block_buffer.find(self.tool_call_end_token)
            if end_idx < 0:
                return None

            payload = self._tool_block_buffer[:end_idx].strip()
            trailing_content = self._tool_block_buffer[
                end_idx + len(self.tool_call_end_token) :
            ]
            self._inside_tool_block = False
            self._tool_block_buffer = ""

            try:
                parsed_tool_calls = self._parse_tool_payload(payload)
            except Exception:
                logger.exception(
                    "Error in extracting buffered Nemotron tool call. payload=%r",
                    payload,
                )
                return None

            if not parsed_tool_calls:
                if trailing_content and trailing_content.strip():
                    self._streamed_content += trailing_content
                    return DeltaMessage(content=trailing_content)
                return None

            if len(self._stream_tool_ids) < len(parsed_tool_calls):
                self._stream_tool_ids.extend(
                    make_tool_call_id()
                    for _ in range(len(parsed_tool_calls) - len(self._stream_tool_ids))
                )

            delta_tool_calls = [
                DeltaToolCall(
                    id=self._stream_tool_ids[idx],
                    type="function",
                    index=idx,
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for idx, tool_call in enumerate(parsed_tool_calls)
            ]

            self._emitted_tool_calls = True
            if trailing_content and trailing_content.strip():
                self._streamed_content += trailing_content
                return DeltaMessage(
                    content=trailing_content,
                    tool_calls=delta_tool_calls,
                )
            return DeltaMessage(tool_calls=delta_tool_calls)

        start_idx = current_text.find(self.tool_call_start_token)
        if start_idx < 0:
            visible_delta = self._visible_delta_outside_tool(delta_text)
            if (
                visible_delta
                and not visible_delta.strip()
                and not self._streamed_content
                and not self._emitted_tool_calls
            ):
                return None
            if visible_delta:
                self._streamed_content += visible_delta
                return DeltaMessage(content=visible_delta)
            return None

        prefix_content = current_text[:start_idx]
        prefix_delta = ""
        if prefix_content.startswith(self._streamed_content):
            prefix_delta = prefix_content[len(self._streamed_content) :]
        elif prefix_content != self._streamed_content:
            prefix_delta = prefix_content

        end_idx = current_text.find(
            self.tool_call_end_token, start_idx + len(self.tool_call_start_token)
        )
        if end_idx < 0:
            delta_start_idx = delta_text.find(self.tool_call_start_token)
            if delta_start_idx >= 0:
                self._inside_tool_block = True
                self._tool_block_buffer = delta_text[
                    delta_start_idx + len(self.tool_call_start_token) :
                ]
            else:
                # The opening tag can straddle chunk boundaries. In that case
                # derive the buffered payload from the full accumulated text
                # rather than slicing the current delta at a mismatched index.
                self._inside_tool_block = True
                self._tool_block_buffer = current_text[
                    start_idx + len(self.tool_call_start_token) :
                ]
            if (
                prefix_delta
                and not prefix_delta.strip()
                and not self._streamed_content
                and not self._emitted_tool_calls
            ):
                return None
            if prefix_delta:
                self._streamed_content += prefix_delta
                return DeltaMessage(content=prefix_delta)
            return None

        trailing_content = current_text[end_idx + len(self.tool_call_end_token) :]
        content_delta = None
        final_plain_content = prefix_content + trailing_content
        if final_plain_content.startswith(self._streamed_content):
            content_delta = final_plain_content[len(self._streamed_content) :]
        elif final_plain_content != self._streamed_content:
            content_delta = final_plain_content

        if self._emitted_tool_calls:
            if content_delta and not content_delta.strip():
                self._streamed_content = final_plain_content
                return None
            if content_delta:
                self._streamed_content = final_plain_content
                return DeltaMessage(content=content_delta)
            return None

        payload = current_text[
            start_idx + len(self.tool_call_start_token) : end_idx
        ].strip()
        try:
            parsed_tool_calls = self._parse_tool_payload(payload)
        except Exception:
            logger.exception(
                "Error in extracting streaming Nemotron tool call. payload=%r",
                payload,
            )
            return None

        if not parsed_tool_calls:
            if content_delta:
                self._streamed_content = final_plain_content
                return DeltaMessage(content=content_delta)
            return None

        if len(self._stream_tool_ids) < len(parsed_tool_calls):
            self._stream_tool_ids.extend(
                make_tool_call_id()
                for _ in range(len(parsed_tool_calls) - len(self._stream_tool_ids))
            )

        delta_tool_calls = [
            DeltaToolCall(
                id=self._stream_tool_ids[idx],
                type="function",
                index=idx,
                function=DeltaFunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for idx, tool_call in enumerate(parsed_tool_calls)
        ]

        self._emitted_tool_calls = True
        self._streamed_content = final_plain_content
        return DeltaMessage(
            content=(content_delta if content_delta and content_delta.strip() else None),
            tool_calls=delta_tool_calls,
        )
