# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypedDict

import pytest
import regex as re

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import NemotronV3ParserReasoningAdapter
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "nemotron_v3"


class ReasoningCase(TypedDict):
    output: str
    reasoning: str | None
    content: str | None


class FakeNemotronTokenizer:
    def __init__(self):
        self._vocab = {
            "<think>": 1,
            "</think>": 2,
            "</": 3,
            "think": 4,
            ">": 5,
            "x": 6,
        }
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        self._pattern = re.compile(r"(<think>|</think>)")

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for part in self._pattern.split(text):
            if part:
                tokens.append(part)
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        return self.convert_tokens_to_string(
            [self._reverse_vocab[token_id] for token_id in token_ids]
        )


@pytest.fixture
def tokenizer():
    return FakeNemotronTokenizer()


@pytest.mark.parametrize(
    "streaming,param_dict",
    [
        pytest.param(
            False,
            {
                "output": "This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="without_start_token",
        ),
        pytest.param(
            True,
            {
                "output": "This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="without_start_token_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "<think>This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="with_start_token",
        ),
        pytest.param(
            True,
            {
                "output": "<think>This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="with_start_token_streaming",
        ),
    ],
)
def test_nemotron_v3_reasoning(
    tokenizer: FakeNemotronTokenizer,
    streaming: bool,
    param_dict: ReasoningCase,
):
    output = tokenizer.tokenize(param_dict["output"])
    model_output = [tokenizer.convert_tokens_to_string([token]) for token in output]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, model_output, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


def test_nemotron_v3_without_thinking_moves_into_content(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is plain content"],
        request=request,
        streaming=False,
    )

    # No real content followed the reasoning, so the trace is moved into
    # content (reasoning left empty) — matching main's behavior.
    assert reasoning is None
    assert content == "This is plain content"


def test_nemotron_v3_streaming_handles_split_end_marker(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is reasoning</thi", "nk>This is content"],
        streaming=True,
    )

    assert reasoning == "This is reasoning"
    assert content == "This is content"


@pytest.mark.parametrize("streaming", [False, True])
def test_nemotron_v3_strips_post_think_separator_newlines(
    tokenizer: FakeNemotronTokenizer,
    streaming: bool,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        (
            ["This is reasoning</think>\n\nThis is content"]
            if not streaming
            else ["This is reasoning", "</think>\n\nThis is content"]
        ),
        streaming=streaming,
    )

    assert reasoning == "This is reasoning"
    assert content == "This is content"


def test_nemotron_v3_streaming_strips_split_post_think_newlines(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is reasoning</think>", "\n", "\nThis is content"],
        streaming=True,
    )

    assert reasoning == "This is reasoning"
    assert content == "This is content"


def test_nemotron_v3_streaming_drops_replayed_end_marker_from_content(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    delta = parser.extract_reasoning_streaming(
        previous_text="This is reasoning</think>",
        current_text="This is reasoning</think></think>\n\nThis is content",
        delta_text="</think>\n\nThis is content",
        previous_token_ids=[2],
        current_token_ids=[2, 2],
        delta_token_ids=[2],
    )

    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "This is content"


def test_nemotron_v3_streaming_preserves_false_end_marker_prefix(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is reasoning</thi", "s is not an end marker"],
        streaming=True,
    )

    assert reasoning == "This is reasoning</this is not an end marker"
    assert content is None


def test_nemotron_v3_streaming_end_detection_handles_token_pieces(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    assert parser.is_reasoning_end_streaming([3, 4, 5], [5])
    assert not parser.is_reasoning_end_streaming([3, 4], [4])
    assert not parser.is_reasoning_end_streaming([3, 4, 5, 6], [6])


def test_nemotron_v3_force_nonempty_content_returns_content(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>This is plain content"],
        request=request,
        streaming=False,
    )

    assert reasoning == "This is plain content"
    assert content == "This is plain content"


def test_nemotron_v3_force_nonempty_keeps_real_content(
    tokenizer: FakeNemotronTokenizer,
):
    # When real content follows the closing tag nothing is promoted: the
    # content after </think> is returned as-is and reasoning stays separate.
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>reasoning here</think>real answer"],
        request=request,
        streaming=False,
    )

    assert reasoning == "reasoning here"
    assert content == "real answer"


def test_nemotron_v3_with_thinking_replays_truncated_reasoning_as_content(
    tokenizer: FakeNemotronTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is truncated reasoning"],
        request=request,
        streaming=False,
    )

    assert reasoning == "This is truncated reasoning"
    assert content == "This is truncated reasoning"


_SPECIAL_TOKEN_IDS = {"<think>": 1, "</think>": 2}


def _token_id(token: str) -> int:
    # Only the think markers need stable ids; everything else is non-special.
    return _SPECIAL_TOKEN_IDS.get(token, 0)


def _make_reasoning_parser(tokenizer):
    class _NemotronParser(DelegatingParser):
        reasoning_parser_cls = NemotronV3ParserReasoningAdapter
        tool_parser_cls = None

    return _NemotronParser(tokenizer)


def _run_parse_delta(parser, tokenizer, text, request):
    tokens = tokenizer.tokenize(text)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for i, token in enumerate(tokens):
        delta = parser.parse_delta(
            delta_text=token,
            delta_token_ids=[_token_id(token)],
            request=request,
            prompt_token_ids=[] if i == 0 else None,
            finished=(i == len(tokens) - 1),
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
    return "".join(reasoning_parts), "".join(content_parts)


def test_nemotron_v3_streaming_promotes_reasoning_to_content(
    tokenizer: FakeNemotronTokenizer,
):
    # Model never closes <think>: reasoning streams normally AND is duplicated
    # into content on the terminal delta.
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(parser, tokenizer, "<think>4", request)

    assert reasoning == "4"
    assert content == "4"


def test_nemotron_v3_streaming_no_promotion_with_real_content(
    tokenizer: FakeNemotronTokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(
        parser, tokenizer, "<think>reason</think>real answer", request
    )

    # Real content followed </think>, so nothing is duplicated.
    assert reasoning == "reason"
    assert content == "real answer"


def test_nemotron_v3_streaming_no_promotion_without_opt_in(
    tokenizer: FakeNemotronTokenizer,
):
    # Without enable_thinking=False / force_nonempty_content the fallback must
    # stay disabled: the response stays reasoning-only, content empty.
    request = ChatCompletionRequest(model="test-model", messages=[])
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(parser, tokenizer, "<think>4", request)

    assert reasoning == "4"
    assert content == ""
