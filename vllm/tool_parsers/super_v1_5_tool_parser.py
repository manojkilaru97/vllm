# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


class SuperV15ToolParser(ToolParser):
    """Parser for v1.5 JSON tool calls wrapped in <TOOLCALL>...</TOOLCALL>."""

    TOOL_START = "<TOOLCALL>"
    TOOL_END = "</TOOLCALL>"
    TOOL_BLOCK_RE = re.compile(r"<TOOLCALL>\s*(.*?)\s*</TOOLCALL>", re.DOTALL)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self._stream_emitted = False
        self._plain_content_emitted_len = 0

    def _suffix_prefix_overlap_len(self, text: str, marker: str) -> int:
        max_overlap = min(len(text), len(marker) - 1)
        for size in range(max_overlap, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def _normalize_json_block(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("["):
            stripped = "[" + stripped
        if not stripped.endswith("]"):
            stripped = stripped + "]"
        return stripped

    def _parse_tool_text(self, text: str) -> list[ToolCall]:
        json_tool_calls = json.loads(self._normalize_json_block(text))
        parsed_tool_calls: list[ToolCall] = []
        for tool_call in json_tool_calls:
            try:
                arguments = tool_call["arguments"]
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                parsed_tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_call["name"],
                            arguments=arguments,
                        ),
                    )
                )
            except Exception:
                logger.exception("Skipping malformed v1.5 tool call: %s", tool_call)
        return parsed_tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        match = self.TOOL_BLOCK_RE.search(model_output)
        if match is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        prefix = model_output[: match.start()].strip() or None
        try:
            tool_calls = self._parse_tool_text(match.group(1))
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=prefix,
            )
        except Exception:
            logger.exception("Error in extracting Super v1.5 tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

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
        if self._stream_emitted:
            return None

        start_idx = current_text.find(self.TOOL_START)
        if start_idx == -1:
            overlap = self._suffix_prefix_overlap_len(current_text, self.TOOL_START)
            safe_text = current_text[:-overlap] if overlap else current_text
            if len(safe_text) <= self._plain_content_emitted_len:
                return None
            content_delta = safe_text[self._plain_content_emitted_len :]
            self._plain_content_emitted_len = len(safe_text)
            return DeltaMessage(content=content_delta or None)

        end_idx = current_text.find(self.TOOL_END, start_idx + len(self.TOOL_START))
        if end_idx == -1:
            return None

        prefix = current_text[:start_idx]
        prefix_delta = None
        if len(prefix) > self._plain_content_emitted_len:
            prefix_delta = prefix[self._plain_content_emitted_len :]
            self._plain_content_emitted_len = len(prefix)

        tool_text = current_text[start_idx + len(self.TOOL_START) : end_idx]
        try:
            tool_calls = self._parse_tool_text(tool_text)
        except Exception:
            logger.exception("Error while streaming Super v1.5 tool call.")
            return None

        delta_tool_calls = []
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        for idx, tool_call in enumerate(tool_calls):
            fn = tool_call.function
            args = fn.arguments if fn and fn.arguments is not None else ""
            name = fn.name if fn else None
            self.prev_tool_call_arr.append({"name": name or "", "arguments": args})
            self.streamed_args_for_tool.append(args)
            delta_tool_calls.append(
                DeltaToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    index=idx,
                    function=DeltaFunctionCall(name=name, arguments=args),
                )
            )

        self._stream_emitted = True
        return DeltaMessage(content=prefix_delta or None, tool_calls=delta_tool_calls)
