# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "openai-community/gpt2"
SCHEMA = json.dumps(
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
)


@pytest.mark.parametrize(
    ("max_whitespace", "disable_any_whitespace", "text", "expected"),
    [
        ("128", False, "{" + "\n" * 128 + '"a": "x"}', True),
        ("128", False, "{" + "\n" * 129 + '"a": "x"}', False),
        ("128", False, "{\n" + " " * 127 + '"a": "x"}', True),
        ("128", False, '{"a": "' + " " * 300 + '"}', True),
        ("0", False, "{" + "\n" * 5000 + '"a": "x"}', True),
        ("128", True, '{\n"a": "x"}', False),
    ],
    ids=[
        "run-at-limit",
        "run-over-limit",
        "indent-at-limit",
        "spaces-inside-string",
        "limit-disabled",
        "any-whitespace-disabled",
    ],
)
def test_xgrammar_json_whitespace_run_limit(
    monkeypatch, max_whitespace, disable_any_whitespace, text, expected
):
    """Unbounded JSON whitespace lets models loop on newlines until max_tokens."""
    monkeypatch.setenv("VLLM_STRUCTURED_OUTPUTS_MAX_WHITESPACE", max_whitespace)
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar", disable_any_whitespace=disable_any_whitespace
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=50257)
    for request_type, spec in (
        (StructuredOutputOptions.JSON, SCHEMA),
        (StructuredOutputOptions.JSON_OBJECT, ""),
    ):
        grammar = backend.compile_grammar(request_type, spec)
        tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
        assert grammar.accept_tokens("", tokens) is expected, (request_type, text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{\n  "a": [1, 2.5e-3],\n  "b": {"c": null, "d": "x\\"y\\u00e9"}\n}', True),
        ('{"a": {' + "\n" * 129 + '"b": [1]}}', False),
        ('{"a": [' + " " * 129 + "1]}", False),
        ("[1]", False),
    ],
    ids=["pretty-nested", "nested-object-run", "array-run", "not-an-object"],
)
def test_xgrammar_json_object_bounds_nested_whitespace(monkeypatch, text, expected):
    monkeypatch.setenv("VLLM_STRUCTURED_OUTPUTS_MAX_WHITESPACE", "128")
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=50257)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON_OBJECT, "")
    tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
    assert grammar.accept_tokens("", tokens) is expected
