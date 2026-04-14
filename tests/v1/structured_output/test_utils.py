# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

import vllm.v1.structured_output.backend_xgrammar as backend_xgrammar
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    has_xgrammar_unsupported_json_features,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def unsupported_string_schemas():
    return [
        {"type": "string", "format": "non_existing_format"},
    ]


@pytest.fixture
def unsupported_integer_schemas():
    return [
        {"type": "integer", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_number_schemas():
    return [
        {"type": "number", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 5},
    ]


@pytest.fixture
def unsupported_object_schemas():
    return [
        {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}},
        {"type": "object", "patternProperties": {"^S": {"type": "string"}}},
    ]


@pytest.fixture
def supported_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
            "status": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
            "car_type": {"type": "string", "enum": ["sedan", "suv", "truck"]},
            "car_brand": {"type": "string", "pattern": "^[a-zA-Z]+$"},
            "short_description": {"type": "string", "maxLength": 50},
            "mileage": {"type": "number", "minimum": 0, "maximum": 1000000},
            "model_year": {
                "type": "integer",
                "exclusiveMinimum": 1900,
                "exclusiveMaximum": 2100,
            },
            "long_description": {"type": "string", "minLength": 50, "maxLength": 2000},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
        },
        "minProperties": 1,
        "maxProperties": 100,
    }


@pytest.mark.parametrize(
    "schema_type",
    [
        "unsupported_string_schemas",
        "unsupported_integer_schemas",
        "unsupported_number_schemas",
        "unsupported_array_schemas",
        "unsupported_object_schemas",
    ],
)
def test_unsupported_json_features_by_type(schema_type, request):
    schemas = request.getfixturevalue(schema_type)
    for schema in schemas:
        assert has_xgrammar_unsupported_json_features(schema), (
            f"Schema should be unsupported: {schema}"
        )


def test_supported_json_features(supported_schema):
    assert not has_xgrammar_unsupported_json_features(supported_schema), (
        "Schema should be supported"
    )


def test_structural_tag_uses_uncached_compile_path(monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeCompiler:
        def __init__(self, tokenizer_info, *, max_threads=8, cache_enabled=True, **_):
            calls.append(
                (
                    "compiler_init",
                    {
                        "tokenizer_info": tokenizer_info,
                        "max_threads": max_threads,
                        "cache_enabled": cache_enabled,
                    },
                )
            )

        def compile_grammar(self, grammar, *, root_rule_name="root"):
            calls.append(
                (
                    "compile_grammar",
                    {"grammar": grammar, "root_rule_name": root_rule_name},
                )
            )
            return "compiled-ctx"

        def compile_structural_tag(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("compile_structural_tag should not be used")

    class FakeGrammar:
        @staticmethod
        def from_structural_tag(*args):
            calls.append(("from_structural_tag", args))
            return "structural-grammar"

    class FakeGrammarMatcher:
        def __init__(self, ctx, *, max_rollback_tokens=-1, **_):
            calls.append(
                (
                    "matcher_init",
                    {"ctx": ctx, "max_rollback_tokens": max_rollback_tokens},
                )
            )

    fake_xgr = SimpleNamespace(
        GrammarCompiler=FakeCompiler,
        Grammar=FakeGrammar,
        GrammarMatcher=FakeGrammarMatcher,
        StructuralTagItem=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(backend_xgrammar, "xgr", fake_xgr)

    backend = object.__new__(XgrammarBackend)
    backend.tokenizer_info = "tokinfo"
    backend.num_speculative_tokens = 0
    backend.vocab_size = 32000
    backend.vllm_config = SimpleNamespace(speculative_config=None)

    grammar = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "tag", "begin": "", "content": {"type": "any_text"}, "end": "</think>"},
                {"type": "json_schema", "json_schema": {"type": "object"}},
            ],
        },
    }

    result = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        json.dumps(grammar),
    )

    assert result.ctx == "compiled-ctx"
    assert ("from_structural_tag", (json.dumps(grammar),)) in calls
    assert (
        "compiler_init",
        {"tokenizer_info": "tokinfo", "max_threads": 8, "cache_enabled": False},
    ) in calls
    assert (
        "compile_grammar",
        {"grammar": "structural-grammar", "root_rule_name": "root"},
    ) in calls
