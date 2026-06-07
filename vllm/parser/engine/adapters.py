# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapters that expose :class:`ParserEngine` through the legacy
:class:`ReasoningParser` and :class:`ToolParser` interfaces.

This lets parser engines flow through the existing serving-layer code
paths that expect separate reasoning and tool parser instances, without
any changes to the serving layer itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from vllm.parser.engine.parser_engine_config import ParserState
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage,
        ExtractedToolCallInformation,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.parser.engine.parser_engine import ParserEngine
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.utils import Tool


class ParserEngineReasoningAdapter(ReasoningParser):
    """Adapts a :class:`ParserEngine` to the :class:`ReasoningParser`
    interface so parser engines can be used as reasoning parsers in the
    existing serving code.

    Subclasses set :attr:`_parser_engine_cls` to the concrete
    :class:`ParserEngine` class.
    """

    _parser_engine_cls: type[ParserEngine]
    engine_based_streaming: bool = True

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._parser_engine = self._parser_engine_cls(tokenizer, **kwargs)  # type: ignore[call-arg]

    @contextmanager
    def _skip_tool_parsing(self) -> Iterator[None]:
        saved = self._parser_engine.skip_tool_parsing
        self._parser_engine.skip_tool_parsing = True
        try:
            yield
        finally:
            self._parser_engine.skip_tool_parsing = saved

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser_engine.is_reasoning_end(list(input_ids))

    def _decode_token_ids(self, token_ids: Sequence[int]) -> str:
        tokenizer = self._parser_engine.model_tokenizer
        try:
            return tokenizer.decode(list(token_ids), skip_special_tokens=False)
        except TypeError:
            return tokenizer.decode(list(token_ids))
        except Exception:
            try:
                tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
                return tokenizer.convert_tokens_to_string(tokens)
            except Exception:
                return ""

    def strip_reasoning_boundary_content(
        self, previous_content: str, content_delta: str
    ) -> str:
        end_marker = self._parser_engine.reasoning_end_str
        if end_marker and end_marker in content_delta:
            _, _, content_delta = content_delta.rpartition(end_marker)
        if not previous_content or not previous_content.lstrip("\n"):
            content_delta = content_delta.lstrip("\n")
        return content_delta

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        delta_ids = list(delta_ids)
        if self.is_reasoning_end(delta_ids):
            return True
        if not delta_ids:
            return False

        # Structured-output gating only has token IDs. Fall back to decoded
        # text so grammar constraints start when the model emits the
        # reasoning end marker as ordinary token pieces.
        end_marker = self._parser_engine.reasoning_end_str
        if not end_marker:
            return False
        delta_start = max(len(input_ids) - len(delta_ids), 0)
        tail_start = max(delta_start - 32, 0)
        previous_tail = self._decode_token_ids(input_ids[tail_start:delta_start])
        current_tail = self._decode_token_ids(input_ids[tail_start:])
        return end_marker in current_tail and end_marker not in previous_tail

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        self._parser_engine.adjust_initial_state_from_prompt(prompt_token_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser_engine.extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        with self._skip_tool_parsing():
            return self._parser_engine.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        with self._skip_tool_parsing():
            return self._parser_engine.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

    @property
    def reasoning_start_str(self) -> str | None:
        return self._parser_engine.reasoning_start_str

    @property
    def reasoning_end_str(self) -> str | None:
        return self._parser_engine.reasoning_end_str

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return self._parser_engine.adjust_request(request)

    def has_engine_confirmed_reasoning_end(self) -> bool:
        return self._parser_engine.reasoning_ended

    def finish_streaming(self) -> DeltaMessage | None:
        return self._parser_engine.finish_streaming()

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        return self._parser_engine.get_streaming_fallback_content(text, request)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self._parser_engine.count_reasoning_tokens(token_ids)


class ParserEngineToolAdapter(ToolParser):
    """Adapts a :class:`ParserEngine` to the :class:`ToolParser` interface.

    :meth:`extract_tool_calls` starts the parser engine in ``CONTENT``
    state so it can parse reasoning-stripped content (i.e. the output of
    :meth:`ReasoningParser.extract_reasoning`).

    Subclasses set :attr:`_parser_engine_cls` to the concrete
    :class:`ParserEngine` class.
    """

    _parser_engine_cls: type[ParserEngine]
    engine_based_streaming: bool = True

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools)
        self._parser_engine = self._parser_engine_cls(tokenizer, tools, **kwargs)  # type: ignore[call-arg]

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        return self._parser_engine.adjust_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        return self._parser_engine.extract_tool_calls_from_content(
            model_output, request
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
        engine = self._parser_engine
        engine.initialize_streaming(initial_state=ParserState.CONTENT)
        return engine.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    def finish_streaming(self) -> DeltaMessage | None:
        return self._parser_engine.finish_streaming()


def make_adapters(
    parser_engine_cls: type[ParserEngine],
) -> tuple[type[ParserEngineReasoningAdapter], type[ParserEngineToolAdapter]]:
    reasoning_adapter = type(
        f"{parser_engine_cls.__name__}ReasoningAdapter",
        (ParserEngineReasoningAdapter,),
        {"_parser_engine_cls": parser_engine_cls},
    )
    tool_adapter = type(
        f"{parser_engine_cls.__name__}ToolAdapter",
        (ParserEngineToolAdapter,),
        {"_parser_engine_cls": parser_engine_cls},
    )
    # Let the serving layer find the adapters and call adjust_request(),
    # which sets skip_special_tokens=False for the detokenizer.
    parser_engine_cls.reasoning_parser_cls = reasoning_adapter  # type: ignore[attr-defined]
    parser_engine_cls.tool_parser_cls = tool_adapter  # type: ignore[attr-defined]
    return reasoning_adapter, tool_adapter
