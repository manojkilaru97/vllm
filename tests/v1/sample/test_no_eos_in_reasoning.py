# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request contract of the opt-in stop-token mask while reasoning is open."""

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.no_eos_in_reasoning import (
    BLOCKED_IDS,
    END_IDS,
    get_normalized_token_ids,
    normalize_no_eos_in_reasoning,
    resolve_default_token_ids,
)
from vllm.v1.sample.no_eos_in_reasoning import (
    NO_EOS_IN_REASONING_KEY as KEY,
)

VOCAB = 1000
DEFAULTS = ([13, 14], [11])


def _params(extra_args, **kwargs) -> SamplingParams:
    params = SamplingParams(extra_args=extra_args, **kwargs)
    params.update_from_generation_config({"eos_token_id": [2, 11]}, eos_token_id=2)
    return params


def _normalize(params, *, v2=True, defaults=DEFAULTS):
    normalize_no_eos_in_reasoning(
        params,
        default_token_ids=defaults,
        vocab_size=VOCAB,
        use_v2_model_runner=v2,
    )
    return params.extra_args


def test_absent_flag_is_a_no_op():
    extra_args = {"other": 1}
    assert _normalize(_params(None)) is None
    assert _normalize(_params(extra_args)) is extra_args
    assert get_normalized_token_ids(extra_args, VOCAB) is None


@pytest.mark.parametrize("value", [False, None])
def test_disabled_flag_is_removed_without_mutating_caller(value):
    extra_args = {KEY: value, "other": 1}
    assert _normalize(_params(extra_args)) == {"other": 1}
    assert extra_args == {KEY: value, "other": 1}


def test_true_resolves_default_tokens_and_blocks_all_stop_ids():
    extra_args = {KEY: True}
    params = _params(extra_args, stop_token_ids=[7])
    # Dynamo assigns stop ids after construction; they must still be blocked.
    params.stop_token_ids = [7, 8]
    normalized = _normalize(params, defaults=([13, 14], [5]))
    assert normalized == {KEY: {END_IDS: [13, 14], BLOCKED_IDS: [2, 5, 7, 8, 11]}}
    assert extra_args == {KEY: True}
    assert get_normalized_token_ids(normalized, VOCAB) == ([13, 14], [2, 5, 7, 8, 11])


def test_explicit_ids_are_sorted_and_deduplicated():
    normalized = _normalize(_params({KEY: [14, 13, 14]}))
    assert normalized == {KEY: {END_IDS: [13, 14], BLOCKED_IDS: [2, 11]}}


def test_normalized_form_is_idempotent():
    params = _params({KEY: True})
    once = _normalize(params)
    assert _normalize(params) == once


@pytest.mark.parametrize(
    "value",
    [
        [],
        [True],
        ["13"],
        [13.0],
        1,
        "true",
        {"ids": [13]},
        {END_IDS: []},
        [-1],
        [VOCAB],
    ],
)
def test_malformed_values_are_rejected(value):
    with pytest.raises(VLLMValidationError):
        _normalize(_params({KEY: value}))


def test_end_token_cannot_be_a_stop_token():
    with pytest.raises(VLLMValidationError):
        _normalize(_params({KEY: [11, 13]}))


def test_allowed_tokens_must_leave_a_non_stop_token():
    with pytest.raises(VLLMValidationError, match="allowed_token_ids"):
        _normalize(_params({KEY: True}, allowed_token_ids=[2, 11]))
    assert _normalize(_params({KEY: True}, allowed_token_ids=[2, 11, 99]))


def test_v1_model_runner_ignores_the_flag():
    assert _normalize(_params({KEY: True, "other": 1}), v2=False) == {"other": 1}


def test_true_without_tokenizer_is_rejected():
    with pytest.raises(VLLMValidationError):
        _normalize(_params({KEY: True}), defaults=None)


@pytest.mark.parametrize(
    "value",
    [True, [13, 14], {END_IDS: [13]}, {END_IDS: [13], BLOCKED_IDS: [VOCAB]}],
)
def test_worker_ignores_unnormalized_values(value):
    assert get_normalized_token_ids({KEY: value}, VOCAB) is None


class _Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return self.vocab[text]


def test_default_tokens_must_be_single_tokens():
    tokenizer = _Tokenizer({"</think>": [13], "<tool_call>": [14], "<|im_end|>": [11]})
    assert resolve_default_token_ids(tokenizer) == ([13, 14], [11])
    tokenizer.vocab["<|im_end|>"] = [60, 61]
    assert resolve_default_token_ids(tokenizer) == ([13, 14], [])
    tokenizer.vocab["<tool_call>"] = [60, 61]
    with pytest.raises(VLLMValidationError):
        resolve_default_token_ids(tokenizer)
