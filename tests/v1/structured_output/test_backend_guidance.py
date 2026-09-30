# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
from concurrent.futures import Future

import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "openai-community/gpt2"


@pytest.fixture(scope="module")
def mistral_tokenizer():
    return get_tokenizer(
        tokenizer_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        tokenizer_mode="mistral",
    )


def test_backend_guidance_rollback_terminated():
    # Test that the backend guidance successfully rollbacks from a
    # terminated state. This can happen with speculative decoding,
    # where the draft model proposes EOS and it is verified by the
    # guidance backend. In that case we are in a stopped state, but
    # it should be reverted in case EOS is not accepted by the target
    # model.
    structured_outputs_config = StructuredOutputsConfig(backend="guidance")
    vllm_config = VllmConfig(structured_outputs_config=structured_outputs_config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    backend = GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    prompt = tokenizer.encode('{"a": "b"}')
    assert len(prompt) > 1
    dummy_wrong = tokenizer.encode('{"a"}')
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Giving any other token should also be accepted
    assert grammar.accept_tokens("", dummy_wrong)
    # Rollback is done from where state was terminated, so from '}' not EOS
    grammar.rollback(len(prompt) - 1)
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
    assert grammar.validate_tokens(dummy_wrong) != dummy_wrong
    assert grammar.accept_tokens("", prompt[1:])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Rollback of <= 0 should not change the terminated state
    grammar.rollback(0)
    assert grammar.is_terminated()
    grammar.rollback(-1)
    assert grammar.is_terminated()


def test_grammar_bitmask_with_specdec():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    for i in range(1, 2):
        sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                json='{"type": "object"}',
            ),
        )
        sampling_params.structured_outputs._backend = "guidance"
        sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

        my_req_id = f"my_req_id_{i}"
        request = Request(
            my_req_id,
            prompt_token_ids=prompt[:i],
            sampling_params=sampling_params,
            pooling_params=None,
        )

        structured_output_manager.grammar_init(request)

        def grammar_bitmask(req: Request, tokens: list[int]) -> None:
            structured_output_manager.grammar_bitmask(
                requests={req.request_id: req},
                structured_output_request_ids={req.request_id: 0},
                scheduled_spec_decode_tokens={req.request_id: tokens},
            )
            # At this point, we rolled-back, so should not be terminated
            assert not req.structured_output_request.grammar.is_terminated()

        # The grammar might not yet be compiled, so we wait for it
        while not request.structured_output_request._check_grammar_completion():
            continue

        assert request.structured_output_request.grammar.accept_tokens(
            request.request_id, prompt[:i]
        )

        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])
        grammar_bitmask(
            request, prompt[i:] + [tokenizer.eos_token_id] + prompt
        )  # EOS not the final token
        grammar_bitmask(request, prompt[i:])  # EOS not present
        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_init_async_and_sync(async_grammar):
    """Test grammar initialization works correctly in both async and sync modes.

    This test validates that the distributed_executor_backend config option
    correctly controls whether grammar compilation happens asynchronously
    (via executor.submit) or synchronously. When set to "external_launcher",
    grammar compilation is synchronous to avoid deadlocks.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')

    # Use "external_launcher" for sync mode, None for async mode
    executor_backend = None if async_grammar else "external_launcher"
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        parallel_config=ParallelConfig(distributed_executor_backend=executor_backend),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json='{"type": "object"}',
        ),
    )
    sampling_params.structured_outputs._backend = "guidance"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "test_request",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )

    structured_output_manager.grammar_init(request)

    # Check the internal _grammar type immediately after init
    # Before _check_grammar_completion is called, async mode should have a Future
    raw_grammar = request.structured_output_request._grammar
    if async_grammar:
        assert isinstance(raw_grammar, Future), (
            "Async mode should store a Future before completion"
        )
    else:
        assert not isinstance(raw_grammar, Future), (
            "Sync mode should store the grammar directly, not a Future"
        )

    # Wait for grammar to be ready (handles both async and sync cases)
    start_time = time.time()
    while not request.structured_output_request._check_grammar_completion():
        if time.time() - start_time > 5:  # 5-second timeout
            pytest.fail("Grammar compilation timed out")
        time.sleep(0.01)

    # After completion, _grammar should no longer be a Future
    assert not isinstance(request.structured_output_request._grammar, Future)

    # Verify grammar is properly initialized and functional
    grammar = request.structured_output_request.grammar
    assert grammar is not None
    assert not grammar.is_terminated()

    # Verify the grammar can accept valid tokens
    assert grammar.accept_tokens(request.request_id, prompt)


def test_disable_any_whitespace_ignores_schema_whitespace_options():
    """A request schema's x-guidance must not re-enable flexible whitespace."""
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(
            backend="guidance", disable_any_whitespace=True
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    backend = GuidanceBackend(vllm_config, tokenizer=tokenizer, vocab_size=50257)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "x-guidance": {
                "whitespace_flexible": True,
                "whitespace_pattern": "[\\n ]*",
            },
        }
    )
    spaced = tokenizer.encode("{" + " " * 200 + '"a":"x"}')
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    assert len(grammar.validate_tokens(spaced)) < len(spaced)
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    assert grammar.accept_tokens("", tokenizer.encode('{"a":"x"}'))


def _structured_request(
    request_id: str, backend: str, params: StructuredOutputsParams, tokenizer
) -> Request:
    sampling_params = SamplingParams(structured_outputs=params)
    sampling_params.structured_outputs._backend = backend
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)
    return Request(
        request_id,
        prompt_token_ids=tokenizer.encode("x"),
        sampling_params=sampling_params,
        pooling_params=None,
    )


def _auto_manager(disable_any_whitespace: bool = True) -> StructuredOutputManager:
    return StructuredOutputManager(
        VllmConfig(
            model_config=ModelConfig(tokenizer=TOKENIZER),
            structured_outputs_config=StructuredOutputsConfig(
                backend="auto", disable_any_whitespace=disable_any_whitespace
            ),
            parallel_config=ParallelConfig(
                distributed_executor_backend="external_launcher"
            ),
        )
    )


# Compact separators differ: xgrammar emits ", " / ": ", guidance "," / ":".
@pytest.mark.parametrize(
    "params,spaced,compact",
    [
        pytest.param(
            {"json": '{"type": "object"}'},
            '{\n"a": "b"}',
            {"xgrammar": '{"a": "b"}', "guidance": '{"a":"b"}'},
            id="json",
        ),
        pytest.param(
            {"json_object": True},
            '{\n"a": "b"}',
            {"xgrammar": '{"a": "b"}', "guidance": '{"a":"b"}'},
            id="json_object",
        ),
        pytest.param(
            {"json": '{"type": "object", "properties": {"m": {}}, "required": ["m"]}'},
            '{"m":\n{"n": 1}}',
            {"xgrammar": '{"m": {"n": 1}}', "guidance": '{"m":{"n":1}}'},
            id="free_form",
        ),
    ],
)
def test_manager_compiles_each_request_with_its_selected_backend(
    params, spaced, compact
):
    """auto can select different backends per request; one engine must honour each,
    and disable_any_whitespace must hold on both."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    manager = _auto_manager()
    requests = {}
    for backend in ("xgrammar", "guidance"):
        request = _structured_request(
            backend, backend, StructuredOutputsParams(**params), tokenizer
        )
        manager.grammar_init(request)
        assert request.structured_output_request._check_grammar_completion()
        requests[backend] = request

    grammars = {k: r.structured_output_request.grammar for k, r in requests.items()}
    assert type(grammars["xgrammar"]).__name__ == "XgrammarGrammar"
    assert type(grammars["guidance"]).__name__ == "GuidanceGrammar"
    assert manager.grammar_bitmask(requests, list(requests), {}) is not None
    spaced_tokens = tokenizer.encode(spaced)
    for request_id, grammar in grammars.items():
        assert len(grammar.validate_tokens(spaced_tokens)) < len(spaced_tokens)
        assert grammar.accept_tokens(
            request_id, tokenizer.encode(compact[request_id])
        ), request_id


def test_guidance_fills_a_narrower_shared_bitmask():
    """guidance's bitmask can be wider than the engine's (tokenizer larger than
    the model vocabulary); it must fill the shared words exactly."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance")
    )
    backend = GuidanceBackend(vllm_config, tokenizer=tokenizer, vocab_size=50257 + 64)
    schema = (
        '{"type": "object", "properties": {"a": {"type": "integer", "multipleOf": 2}}}'
    )
    native = backend.allocate_token_bitmask(1)
    assert native.shape[1] == 1573
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    prefix = tokenizer.encode('{"a":')
    assert grammar.accept_tokens("", prefix)
    grammar.fill_bitmask(native, 0)
    shared = torch.full((2, 1571), -1, dtype=torch.int32)
    grammar.fill_bitmask(shared, 1)
    assert torch.equal(shared[1], native[0, :1571])
    assert torch.equal(shared[0], torch.full((1571,), -1, dtype=torch.int32))


def test_manager_falls_back_for_incompatible_backends(monkeypatch):
    """xgrammar anchors the bitmask under auto; a backend with a different bitmask
    layout is served by it, and a construction error is retried."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    manager = _auto_manager()
    created = []

    class FakeBackend:
        def __init__(self, name, width):
            self.name = name
            self.width = width

        def allocate_token_bitmask(self, n):
            return torch.zeros((n, self.width), dtype=torch.int32)

        def compile_grammar(self, request_type, grammar_spec):
            raise RuntimeError(f"compiled by {self.name}")

    def create(name):
        created.append(name)
        if name == "outlines":
            raise RuntimeError("transient")
        return FakeBackend(name, 8 if name == "xgrammar" else 9)

    monkeypatch.setattr(manager, "_create_backend", create)
    params = StructuredOutputsParams(json='{"type": "object"}')
    for _ in range(2):
        request = _structured_request("b", "guidance", params, tokenizer)
        manager.grammar_init(request)
        error = request.structured_output_request.grammar
        assert isinstance(error, RuntimeError) and "by xgrammar" in str(error)
        with pytest.raises(RuntimeError, match="transient"):
            manager.grammar_init(
                _structured_request("c", "outlines", params, tokenizer)
            )
    assert created == ["xgrammar", "guidance", "outlines", "outlines"]


@pytest.mark.parametrize(
    "request_type,grammar_spec",
    [
        pytest.param(
            StructuredOutputOptions.JSON,
            '{"type": "object"}',
            id="json",
        ),
        pytest.param(
            StructuredOutputOptions.GRAMMAR,
            'start: "hello" | "world"',
            id="lark",
        ),
    ],
)
def test_mistral_tokenizer_compile_grammar(
    mistral_tokenizer,
    request_type: StructuredOutputOptions,
    grammar_spec: str,
) -> None:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
        vllm_config,
        tokenizer=mistral_tokenizer,
        vocab_size=mistral_tokenizer.vocab_size,
    )
    assert backend.ll_tokenizer is mistral_tokenizer.llg_tokenizer

    grammar = backend.compile_grammar(request_type, grammar_spec)
    assert grammar is not None
    assert not grammar.is_terminated()
