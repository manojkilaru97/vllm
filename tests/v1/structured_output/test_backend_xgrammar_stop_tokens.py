# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Issue #60580: xgrammar only treats tokens that decode to an empty string as
special, so a tokenizer's control tokens (e.g. ``<|im_start|>``) matched
grammar text and could be sampled inside a JSON string. Structural tags and
Lark grammars, which can name such tokens, must still be able to emit them.
"""

import json

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

QWEN_TOKENIZER = "Qwen/Qwen3-0.6B"
QWEN_VOCAB_SIZE = 151936

# Qwen3 special added tokens.
ENDOFTEXT = 151643  # <|endoftext|>: a generation_config stop token
IM_START = 151644  # <|im_start|>: never a stop token
IM_END = 151645  # <|im_end|>: the tokenizer's eos


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


@pytest.fixture(scope="module")
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained(QWEN_TOKENIZER)


@pytest.fixture(scope="module")
def qwen_backend(qwen_tokenizer) -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config, tokenizer=qwen_tokenizer, vocab_size=QWEN_VOCAB_SIZE
    )


@pytest.mark.parametrize(
    "request_type, grammar_spec",
    [
        (StructuredOutputOptions.JSON, '{"type": "string"}'),
        (StructuredOutputOptions.REGEX, '"[^"]*"'),
        (StructuredOutputOptions.GRAMMAR, 'root ::= "\\"" [^"]* "\\""'),
    ],
)
def test_special_tokens_masked_until_grammar_ends(
    qwen_backend: XgrammarBackend, qwen_tokenizer, request_type, grammar_spec
):
    grammar = qwen_backend.compile_grammar(request_type, grammar_spec)
    bitmask = qwen_backend.allocate_token_bitmask(1)

    text = qwen_tokenizer.encode('"The company is', add_special_tokens=False)
    assert grammar.accept_tokens("req", text)
    grammar.fill_bitmask(bitmask, 0)
    for token_id in (IM_START, IM_END, ENDOFTEXT):
        assert not _token_allowed(bitmask[0], token_id)

    quote = qwen_tokenizer.encode('"', add_special_tokens=False)
    assert grammar.accept_tokens("req", quote)
    grammar.fill_bitmask(bitmask, 0)
    assert not _token_allowed(bitmask[0], IM_START)
    assert _token_allowed(bitmask[0], IM_END)
    # Request stop tokens are not forwarded to xgrammar (#49227 reverted).
    assert not _token_allowed(bitmask[0], ENDOFTEXT)


@pytest.mark.parametrize(
    "request_type, grammar_spec",
    [
        (
            StructuredOutputOptions.STRUCTURAL_TAG,
            json.dumps(
                {
                    "type": "structural_tag",
                    "format": {
                        "type": "tag",
                        "begin": "<|im_start|>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": {"type": "string"},
                        },
                        "end": "\n",
                    },
                }
            ),
        ),
        (StructuredOutputOptions.GRAMMAR, f'start: <[{IM_START}]> "assistant"'),
    ],
)
def test_token_naming_grammars_can_emit_special_tokens(
    qwen_backend: XgrammarBackend, request_type, grammar_spec
):
    grammar = qwen_backend.compile_grammar(request_type, grammar_spec)
    bitmask = qwen_backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    assert _token_allowed(bitmask[0], IM_START)
    assert grammar.accept_tokens("req", [IM_START])
