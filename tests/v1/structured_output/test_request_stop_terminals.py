# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request stop tokens as xgrammar terminals, as shipped in Super/Ultra prod.

Request stops (``all_stop_token_ids``) reach xgrammar as override stop tokens
(#49227). Stops the tokenizer does not know are additionally re-checked in
``fill_bitmask`` (the prod image-only mask fix for xgrammar < 0.2.8). The
named-tool + MTP case that once forced a revert is covered end to end: drafts
past a request stop must not fail the request.
"""

import json
import logging
import os

import pytest
import torch
from transformers import AutoTokenizer

from tests.v1.core.utils import create_scheduler
from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    XgrammarGrammar,
)

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257
EOS = 50256  # the tokenizer's only default stop token
QUOTE = 1  # `"`
LETTER = 55  # `X`: string content unless registered as a stop
BACKTICK = 63  # "`": string content; bit 31 of its bitmask word

NAMED_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
)
NUM_SPEC_TOKENS = 3


def _allowed(bitmask: torch.Tensor, row: int, token_id: int) -> bool:
    word = int(bitmask[row, token_id // 32]) & 0xFFFFFFFF
    return bool(word >> (token_id % 32) & 1)


@pytest.fixture(scope="module")
def backend() -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config,
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER),
        vocab_size=VOCAB_SIZE,
    )


class _LeakyMatcher:
    """Mimics xgrammar < 0.2.8: leaves one override stop samplable everywhere."""

    def __init__(self, matcher, leaked: int):
        self._matcher = matcher
        self._leaked = leaked

    def fill_next_token_bitmask(self, bitmask, idx):
        self._matcher.fill_next_token_bitmask(bitmask, idx)
        word, bit = divmod(self._leaked, 32)
        value = (int(bitmask[idx, word]) & 0xFFFFFFFF) | (1 << bit)
        bitmask[idx, word] = value - (1 << 32) if value >= 1 << 31 else value

    def __getattr__(self, name):
        return getattr(self._matcher, name)


def test_only_unknown_request_stops_are_rechecked(backend):
    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "string"}', {EOS, LETTER}
    )
    assert isinstance(grammar, XgrammarGrammar)
    assert grammar.extra_stop_token_ids == (LETTER,)
    plain = backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "string"}')
    assert plain.extra_stop_token_ids == ()


@pytest.mark.parametrize("stop", [LETTER, BACKTICK])
def test_mask_fix_keeps_stop_only_where_grammar_can_end(backend, stop):
    compiled = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "string"}', {EOS, stop}
    )
    grammar = XgrammarGrammar(
        matcher=_LeakyMatcher(compiled.matcher, stop),
        vocab_size=VOCAB_SIZE,
        ctx=compiled.ctx,
        extra_stop_token_ids=compiled.extra_stop_token_ids,
    )
    bitmask = backend.allocate_token_bitmask(1)

    # Mid-string the leaked stop would end the output early: masked.
    assert grammar.accept_tokens("req", [QUOTE])
    grammar.fill_bitmask(bitmask, 0)
    assert not _allowed(bitmask, 0, stop)
    assert _allowed(bitmask, 0, LETTER if stop != LETTER else BACKTICK)

    # Once the string closes the stop is a valid terminal: kept.
    assert grammar.accept_tokens("req", [QUOTE])
    grammar.fill_bitmask(bitmask, 0)
    assert _allowed(bitmask, 0, stop)
    assert _allowed(bitmask, 0, EOS)
    # The accept/rollback probe leaves the matcher where it was.
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("req", [stop])
    assert grammar.is_terminated()


def _named_tool_request(stop: int, eos: int, ignore_eos: bool = False):
    params = SamplingParams(
        max_tokens=64,
        stop_token_ids=[stop],
        ignore_eos=ignore_eos,
        structured_outputs=StructuredOutputsParams(json=NAMED_TOOL_SCHEMA),
    )
    assert params.structured_outputs is not None
    params.structured_outputs._backend = "xgrammar"
    params.update_from_generation_config({}, eos)
    return Request("named-tool", [0, 1], params, pooling_params=None)


def _wait_for_grammar(manager: StructuredOutputManager, request: Request):
    manager.grammar_init(request)
    assert request.structured_output_request is not None
    while not request.structured_output_request._check_grammar_completion():
        continue
    return request.structured_output_request.grammar


def test_mtp_drafts_through_request_stop(caplog_vllm):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    manager = StructuredOutputManager(
        VllmConfig(
            model_config=ModelConfig(tokenizer=TOKENIZER),
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
            speculative_config=SpeculativeConfig(
                model="[ngram]", num_speculative_tokens=NUM_SPEC_TOKENS
            ),
        )
    )
    request = _named_tool_request(LETTER, EOS)
    grammar = _wait_for_grammar(manager, request)
    body = tokenizer.encode('{"city": "Paris"}')
    assert grammar.accept_tokens(request.request_id, body[:-1])

    # MTP drafts the closing brace, the request-only stop, then more text.
    drafts = [body[-1], LETTER, QUOTE]
    assert grammar.validate_tokens(drafts) == [body[-1], LETTER]
    with caplog_vllm.at_level(logging.ERROR):
        bitmask = manager.grammar_bitmask(
            {request.request_id: request},
            [request.request_id],
            {request.request_id: drafts[:2]},
        )
    assert "Unexpected" not in caplog_vllm.text
    bitmask = torch.from_numpy(bitmask)
    assert not _allowed(bitmask, 0, LETTER)  # object still open
    assert _allowed(bitmask, 1, LETTER)  # object closed: stop is a terminal
    assert not grammar.is_terminated()

    # Verification accepted tokens past the stop: they are ignored, not fatal.
    assert manager.accept_tokens(request, drafts)
    assert grammar.is_terminated()


def _sampled(req_id: str, tokens: list[int]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_scheduler_named_tool_with_mtp_and_request_stop(ignore_eos):
    scheduler = create_scheduler(num_speculative_tokens=NUM_SPEC_TOKENS)
    manager = scheduler.structured_output_manager
    tokenizer = manager.tokenizer
    eos = tokenizer.eos_token_id
    body = tokenizer.encode('{"city": "Paris"}', add_special_tokens=False)
    stop = tokenizer.encode("X", add_special_tokens=False)[0]
    # With ignore_eos the grammar may end on eos while the engine keeps going.
    terminal = eos if ignore_eos else stop
    stream = body + [terminal] + [stop] * NUM_SPEC_TOKENS
    request = _named_tool_request(stop, eos, ignore_eos)
    _wait_for_grammar(manager, request)
    scheduler.add_request(request)
    rid = request.request_id

    scheduler.update_from_output(scheduler.schedule(), _sampled(rid, stream[:1]))
    pos = 1
    while not request.is_finished():
        assert pos < len(stream) - 1
        drafts = stream[pos : pos + NUM_SPEC_TOKENS]
        scheduler.update_draft_token_ids(DraftTokenIds([rid], [drafts]))
        out = scheduler.schedule()
        # The verifier accepts every surviving draft plus a bonus token.
        sent = out.scheduled_spec_decode_tokens.get(rid, [])
        accepted = sent + [stream[pos + len(sent)]]
        scheduler.update_from_output(out, _sampled(rid, accepted))
        pos += len(accepted)

    assert request.structured_output_request.grammar.is_terminated()
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == stop
    expected = body + ([eos, stop] if ignore_eos else [stop])
    assert list(request.output_token_ids) == expected


# Real prod tokenizer + generation_config dirs (tokenizer*.json, config.json,
# generation_config.json), one subdirectory per model.
PROD_TOKENIZERS = os.environ.get("VLLM_TEST_PROD_TOKENIZERS")
PROD_MODELS = [
    pytest.param("ultra", [2], id="ultra3-</s>"),
    pytest.param("bolt-e4b", [106, 50], id="bolt-e4b-<turn|>"),
    pytest.param("laguna", [24], id="laguna-</assistant>"),
]


@pytest.mark.skipif(not PROD_TOKENIZERS, reason="needs VLLM_TEST_PROD_TOKENIZERS")
@pytest.mark.parametrize("model, stops", PROD_MODELS)
def test_prod_generation_config_stops_end_json(model, stops):
    path = os.path.join(PROD_TOKENIZERS, model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    with open(os.path.join(path, "generation_config.json")) as f:
        generation_config = json.load(f)
    params = SamplingParams()
    params.update_from_generation_config(generation_config, tokenizer.eos_token_id)
    assert set(stops) <= params.all_stop_token_ids
    assert tokenizer.eos_token_id not in stops

    backend = XgrammarBackend(
        VllmConfig(
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
        ),
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )
    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, NAMED_TOOL_SCHEMA, params.all_stop_token_ids
    )
    bitmask = backend.allocate_token_bitmask(1)

    # Inside the string value a request stop would truncate the tool call.
    head = tokenizer.encode('{"city": "Pa', add_special_tokens=False)
    assert grammar.accept_tokens("req", head)
    grammar.fill_bitmask(bitmask, 0)
    for stop in stops:
        assert not _allowed(bitmask, 0, stop), stop

    # Once the object closes, every generation_config stop ends the output.
    assert grammar.accept_tokens(
        "req", tokenizer.encode('ris"}', add_special_tokens=False)
    )
    grammar.fill_bitmask(bitmask, 0)
    for stop in stops + [tokenizer.eos_token_id]:
        assert _allowed(bitmask, 0, stop), stop
    assert grammar.accept_tokens("req", stops[:1])
    assert grammar.is_terminated()
