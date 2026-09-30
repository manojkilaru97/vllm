# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-time validation of structured output requests."""

import json

import pytest

from vllm.config import StructuredOutputsConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

pytestmark = pytest.mark.cpu_test

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "customer": {"type": "string"},
    },
    "required": ["invoice_id", "customer"],
    "additionalProperties": False,
}


class _StubModelConfig:
    def __init__(self, is_diffusion: bool):
        self.is_diffusion = is_diffusion


def test_structured_outputs_rejected_for_diffusion_models():
    """Diffusion LLMs denoise the canvas in parallel, which is incompatible
    with the token-by-token grammar FSM. The request must fail with a clear
    validation error instead of an FSM rejection mid-generation (#45436)."""
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
    )
    with pytest.raises(VLLMValidationError, match="not yet supported for diffusion"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=True),
            StructuredOutputsConfig(),
            tokenizer=None,
        )


def test_plain_request_allowed_for_diffusion_models():
    """Requests without structured outputs are unaffected by the guard."""
    params = SamplingParams()
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=True),
        StructuredOutputsConfig(),
        tokenizer=None,
    )


@pytest.mark.parametrize(
    "structured_outputs, match",
    [
        (StructuredOutputsParams(json_object=False), "json_object must be True"),
        (StructuredOutputsParams(json=""), "json cannot be an empty string"),
    ],
)
def test_degenerate_structured_outputs_rejected(structured_outputs, match):
    """json_object=False and an empty json schema pass the `is not None`
    exclusivity check but resolve to no structured-output key, so they must be
    rejected at request validation (-> 400) instead of reaching and crashing
    the engine."""
    params = SamplingParams(structured_outputs=structured_outputs)
    with pytest.raises(VLLMValidationError, match=match):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(),
            tokenizer=object(),
        )


@pytest.mark.parametrize("backend", ["auto", "xgrammar", "guidance"])
def test_disable_any_whitespace_allowed_for_whitespace_aware_backends(backend):
    config = StructuredOutputsConfig(backend=backend, disable_any_whitespace=True)
    assert config.disable_any_whitespace


@pytest.mark.parametrize("backend", ["outlines", "lm-format-enforcer"])
def test_disable_any_whitespace_rejected_for_other_backends(backend):
    with pytest.raises(ValueError, match="disable_any_whitespace"):
        StructuredOutputsConfig(backend=backend, disable_any_whitespace=True)


def test_disable_any_whitespace_keeps_outlines_fallback():
    """outlines-core JSON regexes allow at most one space between tokens, so the
    outlines fallback stays bounded under disable_any_whitespace."""
    import re

    from outlines_core import json_schema

    schema = {"type": "object", "patternProperties": {"^a": {"type": "string"}}}
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="auto", disable_any_whitespace=True),
        tokenizer=object(),
    )
    assert params.structured_outputs._backend == "outlines"
    regex = json_schema.build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(regex, '{"ab": "x"}')
    assert not re.fullmatch(regex, '{\n"ab": "x"}')
    assert not re.fullmatch(regex, '{"ab":  "x"}')


def test_disable_any_whitespace_keeps_non_json_outlines_fallback(monkeypatch):
    """Non-JSON requests keep the outlines fallback under disable_any_whitespace."""
    import vllm.sampling_params as sampling_params_module
    import vllm.v1.structured_output.backend_outlines as backend_outlines

    monkeypatch.setattr(
        sampling_params_module, "_is_non_tekken_mistral", lambda _: True
    )
    monkeypatch.setattr(
        backend_outlines, "validate_structured_output_request_outlines", lambda _: None
    )
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(regex="(?i:abc)")
    )
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="auto", disable_any_whitespace=True),
        tokenizer=object(),
    )
    assert params.structured_outputs._backend == "outlines"


def test_disable_any_whitespace_keeps_guidance_fallback():
    """xgrammar-unsupported schemas that guidance supports stay compilable."""
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json={"type": "integer", "multipleOf": 2}
        )
    )
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="auto", disable_any_whitespace=True),
        tokenizer=object(),
    )
    assert params.structured_outputs._backend == "guidance"
